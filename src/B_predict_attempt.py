
#%% Based on https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
from lib.t5 import T5ForConditionalGenerationWithConfidence
import json
import pandas as pd
import sys
import torch

from pathlib import Path
from htawta_true import trainer as true_trainer
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from attempt.dataset import AttemptDataSetClass
from attempt.third_party.models import T5ForConditionalGeneration
from attempt.model import get_soft_prompt_model_and_tokenizer

root_dir = Path(__file__).parent.resolve().parent
args = sys.argv[1:]
# args = ['single_model_with_token_random_token_init','1e-4','Belief_states','0.8','dev']

# model_folder = root_dir/'src/single_models/Belief_states'
architecture = args[0]
learning_rate = args[1]
question_type = args[2]
top_p = float(args[3])
dataset_subset = args[4]
dataset = args[5]
size = args[6]
seed = int(args[7])

base_model = f't5-{size}'

if dataset == 'dreamscape' or dataset == 'fairytaleqa':
    question_type = question_type.replace('_',' ')

models_base_name = f'models-{size}-{seed}-control'
model_dir = Path(models_base_name)/architecture/learning_rate

if dataset == 'quail':
    model_dir = Path(models_base_name)/architecture/learning_rate
    ghanem_et_al_2022_prompt_map = {
        'Belief_states': 'Belief States',
        'Causality': 'Causality',
        'Character_identity': 'Character Identity',
        'Entity_properties': 'Entity Properties',
        'Event_duration': 'Event Duration',
        'Factual': 'Factual',
        'Subsequent_state': 'Subsequent State',
        'Temporal_order': 'Temporal Order',
    }
    t5_wta_patch_prompt_map = {
        'Belief_states': '<SB0><SB1><SB2><SB3>',
        'Causality': '<CA0><CA1><CA2>',
        'Character_identity': '<CI0><CI1>',
        'Entity_properties': '<EP0><EP1><EP2>',
        'Event_duration': '<ED0><ED1>',
        'Factual': '<FA0><FA1>',
        'Subsequent_state': '<SS0><SS1><SS2><SS3><SS4>',
        'Temporal_order': '<TO0><TO1><TO2>',
    }
else:
    model_dir = Path(f'{models_base_name}-{dataset}')/architecture/learning_rate

out_dir = model_dir/question_type
out_dir.mkdir(exist_ok=True)

token_prefix_length = 10

architecture = architecture.split('-pt-')[0]

print('Args: ', ', '.join(args))

#######################################
if 'separate_models' in architecture or architecture == 'soft_attempt':
    model_dir = model_dir/question_type

model_dir = model_dir/'model_files'

def load_dataset_df(path):
    rows = []
    with open(path) as dataset_file:
        for line in dataset_file.readlines():
            rows.append(json.loads(line))
    # target_text     input_text      prefix
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            'question': 'target_text', 
            'context': 'input_text'
        }
    )
    df = df[df['question_type']!='Unanswerable']

    df['prefix'] = '' # We add the prefix transparantly above
    df['task'] = df['question_type']
    return df

root = Path(__file__).parent.resolve().parent
if dataset == 'quail':
    df = load_dataset_df(root/f'data/quail/quail_v1.3/json/{dataset_subset}.jsonl')
elif dataset == 'dreamscape':
    df = load_dataset_df(root/f'data/dreamscape/{dataset_subset}_v4.jsonl')
elif dataset == 'fairytaleqa':
    df = load_dataset_df(root/f'data/fairytaleqa/{dataset_subset}.jsonl')
else:
    raise NotImplementedError('Dataset not currently supported, map to QUAIL format and add another case here...')

max_sequences = 0
contexts = []
for key, group in df.groupby('context_id'):
    contexts.append({
        'context_id': key,
        'context': group['input_text'].values[0],
    })
    if len(group) > max_sequences:
        max_sequences = len(group)

#%%
if 'attempt' in architecture:
    question_types = [question_type]
    if architecture == 'soft_skill_attempt':
        question_types = list(ghanem_et_al_2022_prompt_map.keys())
    model, tokenizer = get_soft_prompt_model_and_tokenizer(
        model_dir, 
        model_dir, 
        'cuda', 
        question_types,
    )
else:
    model = T5ForConditionalGenerationWithConfidence.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
model_params = {
    "OUTPUT_DIR": str(out_dir),
    # "MODEL": "t5-large",
    # "TRAIN_BATCH_SIZE": 8,        
    "VALID_BATCH_SIZE": 16,        
    # "TRAIN_EPOCHS": 100,          
    # "VAL_EPOCHS": 1,              
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 64,
    # "early_stopping_patience": 1,  
    "TOP_P": top_p,
}
print('Model Params', model_params)
context_texts = [x['context'] for x in contexts]
rows = []
for context in context_texts:
    for i in range(max_sequences):
        rows.append({
            'question': '', 
            'context': context,
            'task': question_type,
        })
df = pd.DataFrame(rows)
if 'attempt' in architecture:
    pred_set = AttemptDataSetClass(df, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text="context", target_text="question")
else:
    pred_set = true_trainer.YourDataSetClass(df, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text="context", target_text="question")
val_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
loader = DataLoader(pred_set, **val_params)
device = 'cuda' # if torch.cuda.is_available() else 'cpu'
if size == '11b':
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        1: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
else:
    model.to(device)
outs = true_trainer.generate(tokenizer, model, device, loader, model_params, total=(len(rows)//model_params["VALID_BATCH_SIZE"]))
out_df = pd.DataFrame({'Generated Text':outs[0],'Actual Text':outs[1], 'Token Probs': outs[2]})
preds_1d = out_df['Generated Text'].tolist()
preds = [preds_1d[i*max_sequences:(i+1)*max_sequences] for i in range(len(context_texts))]
probs = out_df['Token Probs'].tolist()

print('Sample input:', context_texts[0])
print('Sample output:', preds[0][0])
print('Sample probs:', probs[0])

probs_2d = [probs[i*max_sequences:(i+1)*max_sequences] for i in range(len(preds))]
#%%
out_rows = []
for pred_list,prob_list,context in zip(preds,probs_2d,contexts):
    for pred,prob in zip(pred_list,prob_list):
        out_rows.append({
            'context_id': context['context_id'],
            'context': context['context'],
            'prediction': pred,
            # 'confidence': conf.tolist(),
            'probs': prob.tolist(),
        })
with open(out_dir/f'predictions_nucleus_{dataset_subset}_{top_p}.json', 'w') as f:
    json.dump(out_rows, f)

# %%
