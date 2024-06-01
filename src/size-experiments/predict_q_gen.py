#%%
import sys
import torch
import numpy as np
import json

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator
from peft import PeftModel, PeftConfig
from pathlib import Path
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader

args = sys.argv[1:]

path = args[0]
# take_param = args[0]
# seed_param = args[1]
# epoch = args[2]

# model_path = str(Path(f'models/squad-q-gen-{take_param}-{seed_param}/epoch_{epoch}').absolute())
model_path = str(Path(path).absolute())

config = PeftConfig.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, model_path)
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

#%%

take = 10

dataset_sq = load_dataset('squad')

dataset_sq = dataset_sq.map(
    lambda x: {"text_label": [y for y in x['question']]},
    batched=True,
    num_proc=1,
)

# Same format as T5 paper see (https://arxiv.org/pdf/1910.10683.pdf page 53)
dataset_sq = dataset_sq.map(
    lambda x: {"input_text": [f'{c}' for q,c in zip(x['question'], x['context'])]},
    batched=True,
    num_proc=1,
)

dataset = dataset_sq

select_indexes = [1,2,3]
dataset['train'] = dataset['train'].select(select_indexes[:take])

dataset["train"][0]


#%%
max_length = 512
target_max_length = 64
input_column = "input_text"
label_column = "text_label"

def preprocess_function(examples):
    inputs = examples[input_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="np")
    labels = tokenizer(
        targets, max_length=target_max_length, padding="max_length", truncation=True, return_tensors="np"
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

eval_dataset = processed_datasets["validation"]

#%%
from nltk.translate.meteor_score import meteor_score
from munkres import Munkres

def meteor(prediction:str,references:list):
    return meteor_score(
        references,
        prediction
    )

def multi_meteor_eval(predictions, references):
    num_preds = len(predictions)
    matrix = [
        [-meteor(
            predictions[i], 
            [references[j]]
        )
        for i in range(num_preds)] for j in range(num_preds)
    ]
    m = Munkres()
    indexes = m.compute(matrix)

    final_references = [references[i] for i,j in indexes]
    final_predictions = [predictions[j] for i,j in indexes]

    local_scores = []
    for label,prediction in zip(final_references, final_predictions):
        local_scores.append(meteor(
            prediction, 
            [label]
        ))
    return np.mean(local_scores)

# %%
device = 'cuda'
batch_size = 32

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

eval_df = dataset['validation'].to_pandas()
print('columns:', eval_df.columns)

preds = []
for row in tqdm(eval_dataloader):
    with torch.inference_mode():
        # input_ids = tokenizer(row["input_text"], return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids=row['input_ids'].cuda(), max_new_tokens=target_max_length)
        pred_texts = tokenizer.batch_decode(outputs.detach().cpu(), skip_special_tokens=True)
        # row['labels'][row['labels'] == -100] = tokenizer.pad_token_id
        # label_texts = tokenizer.batch_decode(row['labels'], skip_special_tokens=True)
        preds.extend(pred_texts)

eval_df['preds'] = preds

multi_meteor_scores = []
for name, group in eval_df.groupby('context'):
    preds = group['preds'].to_list()
    labels = group['question'].to_list()
    multi_meteor_scores.append(multi_meteor_eval(preds,labels))

meteor_mean = float(np.mean(multi_meteor_scores))
meteor_std = float(np.std(multi_meteor_scores))

with open(Path(model_path)/'multi_meteor.json', 'w') as f:
    json.dump({
        'meteor_mean':meteor_mean, 
        'meteor_std':meteor_std,
        # 'seed': seed_param,
        # 'take': take_param,
    }, f)

# %%
