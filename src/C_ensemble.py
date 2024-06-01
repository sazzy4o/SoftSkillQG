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

root = Path(__file__).parent.resolve().parent

args = sys.argv[1:]
# args = [
#     'single_model_with_token_random_token_init', # architecture
#     '1e-6', # learning_rate
#     '0.9', # top_p_value
#     'meteor', # metric
#     'semantic_cluster', # deduplication_method
#     '0.90', # deduplication_cutoff
#     'dev', # dataset_subset
#     'quail', # dataset
#     'models', # model_dir
#     'results_all_metrics_debug', # out_dir
# ]

decision_variable = 'probs' # 'confidences' or 'probs'
architecture = args[0]
learning_rate = args[1]
top_p_value = args[2] # '0.5'
metric = args[3]
dataset_subset = args[4]
dataset = args[5]
size = args[6]
seed = int(args[7])
random.seed(seed)
deduplication_method = 'whitespace'
deduplication_cutoff = float('nan')

max_type_count = float('inf')
deduplication_cutoff_float = float(deduplication_cutoff)

architecture_full = architecture
if '_max' in architecture:
    max_type_count = int(architecture.split('_')[-1])
    architecture = '_'.join(architecture.split('_')[:-2])

models_base_name = f'models-{size}-{seed}-control'

if len(args) >= 10:
    model_dir = Path(args[8])/architecture/learning_rate
    out_dir = root/'src'/args[9]
else:
    if dataset == 'quail':
        model_dir = Path(models_base_name)/architecture/learning_rate
        out_dir = root/'src/results_all_metrics_seed_control'
    else:
        model_dir = Path(f'{models_base_name}-{dataset}')/architecture/learning_rate
        out_dir = root/f'src/results_all_metrics_seed_{dataset}'

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

def load_dataset_df(path):
    rows = []
    with open(path) as dataset_file:
        for line in dataset_file.readlines():
            rows.append(json.loads(line))
    # target_text     input_text      prefix
    df = pd.DataFrame(rows)
    df['prefix'] = 'ask_question' 
    return df

if dataset == 'quail':
    df = load_dataset_df(root/f'data/quail/quail_v1.3/json/{dataset_subset}.jsonl')
elif dataset == 'dreamscape':
    df = load_dataset_df(root/f'data/dreamscape/{dataset_subset}_v4.jsonl')
elif dataset == 'fairytaleqa':
    df = load_dataset_df(root/f'data/fairytaleqa/{dataset_subset}.jsonl')
else:
    raise NotImplementedError('Dataset not currently supported, map to QUAIL format and add another case here...')

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
            context_dict = current_dict.get(row['context_id'],{
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
            if len(context_dict['questions'])==0:
                context_dict['questions'] = list(df[df['context_id']==row['context_id']]['question'])
            current_dict[row['context_id']] = context_dict
            # {
            # 'context_id': context['context_id'],
            # 'context': context['context'],
            # 'prediction': pred,
            # 'confidence': conf.tolist(),
            # 'probs': prob.tolist(),
            # }
#%% Remove duplicates
dedup_model_type = deduplication_method.split('_')[-1]
for gen_dict in [ensemble_dict]:
    for key,context in tqdm(gen_dict.items()):
        pred_set = set()
        remove_indexes = []
        if deduplication_method == 'whitespace':
            for i,pred in enumerate(context['predictions']):
                if pred not in pred_set:
                    pred_set.add(pred)
                else:
                    remove_indexes.append(i)
        elif deduplication_method.startswith('semantic_cluster'):
            sequence_tuples = [
                (x[0],np.mean(x[1]),x[2]) for x in zip(context['predictions'],context['probs'], range(len(context['predictions'])))
            ]
            clusters = cluster(sequence_tuples, cutoff=deduplication_cutoff_float, model_type=dedup_model_type)
            for cluster_group in clusters:
                if len(cluster_group)>1:
                    cluster_group = sorted(cluster_group, key=lambda x: x[1], reverse=True)
                    for i in range(1,len(cluster_group)):
                        remove_indexes.append(cluster_group[i][2])
        else:
            raise Exception('Unknown deduplication method')
        for i in sorted(remove_indexes,reverse=True):
            del context['predictions'][i]
            del context['question_types'][i]
            # del context['confidences'][i]
            del context['probs'][i]
        if len(context['questions'])>len(context['predictions']):
            for _ in range( len(context['questions'])-len(context['predictions']) ):
                context['predictions'].append(' ')
                # context['confidences'].append(0)
                context['probs'].append(0)
                context['question_types'].append('')
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
        print(key)
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

        matrix = [
            [-metric_func(
                selected_predictions[i], 
                [questions[j]]
            )
            for i in range(num_preds)] for j in range(num_preds)
        ]
        m = Munkres()
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
with open(out_dir/f'ensemble_{metric}_{architecture_full}_{decision_variable}_{top_p_value}_{deduplication_method}_{deduplication_cutoff}_{learning_rate}_{dataset_subset}_{size}_{seed}.json','w') as f:
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
        'seed': seed,
    },f)
