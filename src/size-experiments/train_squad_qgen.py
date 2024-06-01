# %% Based on https://github.com/huggingface/peft/blob/0ae52fece17a3514116815984444116b75d9c5ca/examples/conditional_generation/peft_prompt_tuning_seq2seq.ipynb (Apache License 2.0)
import os
import numpy as np
import json
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_constant_schedule_with_warmup, Adafactor
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

args = sys.argv[1:]
# args = ['1000', '1']
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print('Args', args)

device = "cuda"
model_name_or_path = "google/t5-large-lm-adapt"
tokenizer_name_or_path = "google/t5-large-lm-adapt"

checkpoint_name = "financial_sentiment_analysis_prompt_tuning_v1.pt"
input_column = "input_text"
label_column = "text_label"
max_length = 512
lr = 0.3
total_steps = 30_000
batch_size = 8
weight_decay = 1e-5
scale_parameter = False
early_stopping_patience = 3
num_warmup_steps = 0 # Constant learning rate
gradient_accumulation_steps = 4
take = int(args[0]) if args[0] != 'None' else None
seed = int(args[1])
save_path = Path(f"./models/squad-q-gen-{take}-{seed}")
save_path.mkdir(parents=True, exist_ok=True)

torch.manual_seed(seed) # pytorch random seed
np.random.seed(seed) # numpy random seed

# %%
# creating model
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=100,
    prompt_tuning_init_text="What is the answer to this question?\n",
    inference_mode=False,
    tokenizer_name_or_path=model_name_or_path,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# model

#%%
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

if take:
    rng = np.random.default_rng(seed=seed)
    select_indexes = list(range(87599))
    rng.shuffle(select_indexes)
    dataset['train'] = dataset['train'].select(select_indexes[:take])

dataset["train"][0]

# %%
# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in dataset["train"]['text_label']])


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

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

# %%
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    collate_fn=default_data_collator, 
    batch_size=batch_size, 
    pin_memory=True,
)
eval_dataloader = DataLoader(
    eval_dataset, 
    collate_fn=default_data_collator, 
    batch_size=batch_size*6, 
    pin_memory=True,
)

# %%
# optimizer and lr scheduler
optimizer = Adafactor(
    model.parameters(),
    lr=lr,
    scale_parameter=scale_parameter,
    relative_step=False,
)
# lr_scheduler = get_constant_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=total_steps,
# )

# %%
# training and evaluation
model = model.to(device)

keep_going = True
epoch = 0
steps = 0
best_eval_loss = float('inf')
early_stoping_wait_time = 0

while keep_going:
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        # print('loss', f_loss)
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            steps += 1

        f_loss = loss.detach().float()
        total_loss += f_loss
        
        if steps >= total_steps:
            keep_going = False
            break

    model.eval()
    eval_loss = 0
    # eval_preds = []
    with torch.inference_mode():
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            # eval_preds.extend(
            #     tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            # )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    model.save_pretrained(save_path/f'epoch_{epoch}')
    (save_path/f'epoch_{epoch}/score.json').write_text(
        json.dumps({
            'train_ppl': train_ppl.item(),
            'train_epoch_loss': train_epoch_loss.item(),
            'eval_ppl': eval_ppl.item(),
            'eval_epoch_loss': eval_epoch_loss.item(),
        })
    )
    epoch += 1
    
    if eval_epoch_loss < best_eval_loss:
        best_eval_loss = eval_epoch_loss
        early_stoping_wait_time = 0
    else:
        early_stoping_wait_time += 1

    if early_stoping_wait_time >= early_stopping_patience:
        break

# %%
