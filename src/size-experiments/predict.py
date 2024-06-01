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

model_path = str(Path(args[0]).absolute())

config = PeftConfig.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, model_path)
model = model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

#%%

take = 10

dataset_sq = load_dataset('squad')

dataset_sq = dataset_sq.map(
    lambda x: {"text_label": [y['text'][0] for y in x['answers']]},
    batched=True,
    num_proc=1,
)

# Same format as T5 paper see (https://arxiv.org/pdf/1910.10683.pdf page 53)
dataset_sq = dataset_sq.map(
    lambda x: {"input_text": [f'question: {q} context: {c}' for q,c in zip(x['question'], x['context'])]},
    batched=True,
    num_proc=1,
)

dataset = dataset_sq

if take:
    rng = np.random.default_rng(seed=42)
    select_indexes = list(range(87599))
    rng.shuffle(select_indexes)
    dataset['train'] = dataset['train'].select(select_indexes[:take])

dataset["train"][0]


#%%
max_length = 512
target_max_length = 76
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
import collections

def f1_score(target, prediction):
    """Computes token f1 score for a single target and prediction."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (collections.Counter(prediction_tokens) &
              collections.Counter(target_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# %%
device = 'cuda'
batch_size = 64

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

f1_scores = []
for row in tqdm(eval_dataloader):
    with torch.inference_mode():
        # input_ids = tokenizer(row["input_text"], return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids=row['input_ids'].cuda(), max_new_tokens=76)
        pred_texts = tokenizer.batch_decode(outputs.detach().cpu(), skip_special_tokens=True)
        row['labels'][row['labels'] == -100] = tokenizer.pad_token_id
        label_texts = tokenizer.batch_decode(row['labels'], skip_special_tokens=True)
        f1_scores.extend(
            [f1_score(l, p) for p,l in zip(pred_texts, label_texts)]
        )

f1_mean = float(np.mean(f1_scores))
f1_std = float(np.std(f1_scores))

with open(Path(model_path)/'f1.json', 'w') as f:
    json.dump({'f1_mean':f1_mean, 'f1_std':f1_std}, f)

# %%
