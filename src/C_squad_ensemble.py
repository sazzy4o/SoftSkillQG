# %%
import re
import json
import random
import numpy as np
import pandas as pd
import sys

from munkres import Munkres
from pathlib import Path
from tqdm import tqdm
from lib.metrics import metric_map
from lib.deduplication import cluster
from datasets import load_dataset

root = Path(__file__).parent.resolve().parent

args = sys.argv[1:]
# args = [
#     'ghanem_et_al_2022_true', # architecture
#     '1e-5', # learning_rate
#     '0.9', # top_p_value
#     'meteor', # metric
#     'dev', # dataset_subset
#     'squad', # dataset
#     '1000', # take
#     '1', # seed
# ]

decision_variable = 'probs' # 'confidences' or 'probs'
architecture = args[0]
learning_rate = args[1]
top_p_value = args[2] # '0.5'
metric = args[3]
dataset_subset = args[4]
dataset = args[5]
take = int(args[6]) if args[6] != 'None' else None
seed = int(args[7])
random.seed(seed)
deduplication_method = 'none'
deduplication_cutoff = float('nan')

size = 'large'

max_type_count = float('inf')
deduplication_cutoff_float = float(deduplication_cutoff)

architecture_full = architecture
if '_max' in architecture:
    max_type_count = int(architecture.split('_')[-1])
    architecture = '_'.join(architecture.split('_')[:-2])

models_base_name = f'models-squad-{seed}'

model_dir = Path(models_base_name)/f'{architecture}-{take}'/learning_rate
out_dir = root/'src/results_all_metrics_seed_squad'

model_path = str(model_dir.resolve())

remove_type_tokens_to_target_text = True
# if architecture in ['single_model_with_token_v3']:
#     remove_type_tokens_to_target_text = True
remove_tokens_regex = re.compile(r'<\w+>')

metric_func = metric_map[metric]
print({
    # 'decision_variable':decision_variable,
    'top_p_value': top_p_value,
    'learning_rate': learning_rate,
    'architecture':architecture,
    'model_path':model_path,
    'metric':metric,
})
print("Args:", args)

def load_squad_df(split):
    dataset_sq = load_dataset('squad', split=split)
    df = pd.DataFrame(dataset_sq)
    df['question_type'] = 'Unset'
    
    return df

if dataset == 'squad':
    df = load_squad_df('validation')
else:
    raise NotImplementedError('Dataset not currently supported, map to SQUAD format and add another case here...')

models_dir = Path(model_path).resolve()
models_dir_list = list(models_dir.iterdir())
#%%
all_dict = {}
ensemble_dict = {}
for path in tqdm(models_dir_list):
    print(path)
    if not path.is_dir():
        continue
    if path.name in ['decoders','best_model','pretrain','model_files']:
        continue
    if path.name.startswith('checkpoint'):
        continue
    with open(path/f'predictions_nucleus_{dataset_subset}_{top_p_value}.json') as f:
        if path.name == 'All':
            current_dict = all_dict
        else:
            current_dict = ensemble_dict
        data = json.load(f)
        for row in data:
            row['context'] = row['context'].split('</s> ')[-1].split('<Unset9>')[-1]
            context_dict = current_dict.get(row['context'],{
                # 'confidences': [],
                'questions': [],
                'predictions': [],
                'probs': [],
                'question_types': [],
            })
            if remove_type_tokens_to_target_text:
                row['prediction'] = remove_tokens_regex.sub('',row['prediction'].split('<ans_sep>')[0].split('ans_sep>')[0])

            row['prediction'] = row['prediction'].strip()

            # context_dict['confidences'].append(row['confidence'])
            context_dict['predictions'].append(row['prediction'])
            context_dict['question_types'].append(path.name)
            context_dict['probs'].append(row['probs'])

            # print('Row context', row['context'])
            # print('DF columns', df.columns)
            if len(context_dict['questions'])==0:
                context_dict['questions'] = list(df[df['context'].str.strip()==row['context']]['question'])

            assert len(context_dict['questions']) != 0
            
            current_dict[row['context']] = context_dict
            # {
            # 'context_id': context['context_id'],
            # 'context': context['context'],
            # 'prediction': pred,
            # 'confidence': conf.tolist(),
            # 'probs': prob.tolist(),
            # }
#%%        

def get_ensemble_scores(ensemble_dict):
    scores = []
    question_types = []
    labels = []
    predictions = []
    contexts = []
    if metric == 'fbd':
        from eval_metrics.bert_distances import FBD
        def neg_fbd_dist_eval(predictions_raw, references_raw):
            fbd = FBD(references=references_raw, model_name="bert-base-uncased", bert_model_dir=None)
            return -fbd.get_score(sentences=predictions_raw), None, None
        for key,value in tqdm(ensemble_dict.items()):
            questions = value['questions']
            local_predictions = value['predictions']
            local_question_types = value['question_types']
            local_scores = [float(neg_fbd_dist_eval(local_predictions,questions)[0])]
            scores.append(local_scores)
            question_types.append(local_question_types)
            labels.append(questions)
            predictions.append(local_predictions)
            contexts.append(key)
        return scores,question_types,labels,predictions,contexts

    for key,value in tqdm(ensemble_dict.items()):
        # print(key)
        conf_agg = [np.mean(x) for x in value[decision_variable]]
        num_preds = len(value['questions'])
        questions = value['questions']
        question_type_count = len(set(value['question_types']))
        # Make sure we have enough questions selected
        local_max_type_count = np.max([np.ceil(num_preds/question_type_count),max_type_count])
        pred_index = []
        type_counts = {}
        blank_index = None
        for index in np.argsort(conf_agg)[::-1]:
            q_type = value['question_types'][index]
            q_type_count = type_counts.get(q_type,0)
            if q_type_count < local_max_type_count:
                type_counts[q_type] = q_type_count + 1
                pred_index.append(index)
                if q_type == '':
                    blank_index = index

                if len(pred_index) == num_preds:
                    break
        
        # Handle when not enough unique questions were generated
        while len(pred_index) < num_preds:
            pred_index.append(len(pred_index))
            value['predictions'].append(' ')
            value['question_types'].append('')

        # pred_index = np.argpartition(conf_agg, -num_preds)[-num_preds:]
        # pred_index = list(range(num_preds))
        # pred_index = random.choices(range(len(value['predictions'])),k=num_preds)
        selected_predictions = [value['predictions'][i] for i in pred_index]
        selected_question_types = [value['question_types'][i] for i in pred_index]

        if len(selected_question_types) == 1:
            indexes = [(0,0)]
        else:
            matrix = [
                [-metric_func(
                    selected_predictions[i], 
                    [questions[j]]
                )
                for i in range(num_preds)] for j in range(num_preds)
            ]
            m = Munkres()

            # print('selected_predictions', selected_predictions)
            # print('questions', questions)
            # print('matrix', matrix)

            indexes = m.compute(matrix)

        final_questions = [questions[i] for i,j in indexes]
        final_predictions = [selected_predictions[j] for i,j in indexes]
        final_question_types = [selected_question_types[i] for i,j in indexes]

        local_scores = []
        for label,prediction in zip(final_questions,final_predictions):
            local_scores.append(metric_func(
                prediction, 
                [label]
            ))
        scores.append(local_scores)
        question_types.append(final_question_types)
        labels.append(final_questions)
        predictions.append(final_predictions)
        contexts.append(key)
        # print('Q',final_questions)
        # print('P',final_predictions)
        print('C',[(x[1],x[2]) for x in sorted(zip(local_scores,final_questions,final_predictions),reverse=True)])
    return scores,question_types,labels,predictions,contexts

ensemble_scores,all_question_types,all_labels,all_predictions,all_contexts = get_ensemble_scores(ensemble_dict)
mean_score = float(np.mean([np.mean(x) for x in ensemble_scores]))
print('Ensemble Score: ',mean_score)

# %%
with open(out_dir/f'ensemble_{metric}_{architecture_full}_{decision_variable}_{top_p_value}_{deduplication_method}_{deduplication_cutoff}_{learning_rate}_{dataset_subset}_{size}_{take}_{seed}.json','w') as f:
    json.dump({
        'mean_score': mean_score,
        'scores': ensemble_scores,
        'question_types': all_question_types,
        'labels': all_labels,
        'predictions': all_predictions,
        'deduplication_method': deduplication_method,
        'deduplication_cutoff': deduplication_cutoff,
        'contexts': all_contexts,
        # Can now compute t_stat based on the scores
        # 'all_score': float(np.mean(all_scores)),
        # 't_stat': float(t_stat),
        # 'p_value': float(p_value),
        'top_p_value': top_p_value,
        'decision_variable': decision_variable,
        'architecture': architecture_full,
        'learning_rate': learning_rate,
        'metric': metric,
        'dataset': dataset_subset,
        'size': size,
        'take': take,
        'seed': seed,
    },f)
