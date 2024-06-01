#%% Based on https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
import re
import numpy as np
import pandas as pd

from pathlib import Path
from htawta_true import trainer as true_trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer

import json
import sys
import torch

from pathlib import Path

# Speed things up
# torch.backends.cuda.matmul.allow_tf32 = True

root = (Path(__file__).parent/'../').resolve()

args = sys.argv[1:]
# args = ['t5_wta_patch','5e-5','quail','large'] # ! TODO: Remove this line
############ Config ############

architecture = args[0]
learning_rate = float(args[1])
token_prefix_length = 10

dataset = args[2]

size = args[3]
# t5-small
# t5-base
# t5-large
# t5-3b
# t5-11b.

seed = int(args[4])

torch.manual_seed(seed) # pytorch random seed
np.random.seed(seed) # numpy random seed
torch.backends.cudnn.deterministic = True

# ! Add batch? (Required unique seeds)
batch_strategy = 'random'

base_model = f't5-{size}'

model_dir_name = architecture

models_base_name = f'models-{size}-{seed}-v2'

if dataset == 'quail':
    out_folder = Path(models_base_name)/model_dir_name/args[1]
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
    optimal_init_map = {
        1: {
            "ghanem_et_al_2022_true": "1e-4",
            "single_model_with_token_random_token_init": "5e-5",
        },
        2: {
            "ghanem_et_al_2022_true": "1e-4",
            "single_model_with_token_random_token_init": "5e-5",
        },
        3: {
            "ghanem_et_al_2022_true": "1e-5",
            "single_model_with_token_random_token_init": "1e-5",
        },
    }
else:
    out_folder = Path(f'{models_base_name}-{dataset}')/model_dir_name/args[1]

epochs = 100

supported_architectures = [
    'ghanem_et_al_2022_true',
    'separate_models',
    'single_model',
    'single_model_with_token_random_token_init',
    'single_model_soft_prompt_patch',
    't5_wta_patch',
]

################################

out_folder.mkdir(parents=True,exist_ok=True)

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
    if architecture == 'single_model_with_token_random_token_init' or architecture == 'single_model_soft_prompt_patch':
        df['input_text'] = df.apply(lambda x: ''.join([f'<{x.question_type}{i}>' for i in range(token_prefix_length)])+ x.input_text, axis=1)
    elif architecture == 'ghanem_et_al_2022_true':
        df['input_text'] = df.apply(lambda row: ghanem_et_al_2022_prompt_map[row['question_type']]+' </s> '+ row['input_text'], axis=1)
    elif architecture == 't5_wta_patch':
        df['input_text'] = df.apply(lambda row: t5_wta_patch_prompt_map[row['question_type']]+' </s> '+ row['input_text'], axis=1)

    df['prefix'] = '' # We add the prefix transparantly above
    return df

if dataset == 'quail':
    df_train = load_dataset_df(root/'data/quail/quail_v1.3/json/train.jsonl')
    df_validation = load_dataset_df(root/'data/quail/quail_v1.3/json/dev.jsonl')
    df_test = load_dataset_df(root/'data/quail/quail_v1.3/json/challenge.jsonl')
    question_types = [
        "Belief_states",
        "Causality",
        "Character_identity",
        "Entity_properties",
        "Event_duration",
        "Factual",
        "Subsequent_state",
        "Temporal_order",
        # "Unanswerable",
    ]
elif dataset == 'dreamscape':
    df_train = load_dataset_df(root/'data/dreamscape/train_v2.jsonl')
    df_validation = load_dataset_df(root/'data/dreamscape/dev_v2.jsonl')
    df_test = load_dataset_df(root/'data/dreamscape/test_v2.jsonl')
    question_types = sorted(list(set(df_train['question_type'])))
else:
    raise NotImplementedError('Dataset not currently supported, map to QUAIL format and add another case here...')
#%%
job_name = f"Quail Question Generation with T5 ({architecture}, lr {learning_rate})"

print('job:',job_name)

model_params = {
    "OUTPUT_PATH": str(out_folder.resolve()),
    "MODEL": base_model,
    # Total batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_BATCH_SIZE = 8
    "TRAIN_BATCH_SIZE": 8,
    "GRADIENT_ACCUMULATION_BATCH_SIZE": 1, # Equivalent to batch size 8 (slower, but lower memory)
    "VALID_BATCH_SIZE": 8,        
    "TRAIN_EPOCHS": epochs,           
    "LEARNING_RATE": learning_rate,
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 64,
    "early_stopping_patience": 3,  
}

if architecture=='separate_models':
    question_type = args[5]
    print('Question type:',question_type)
    out_folder_resolved = (out_folder/question_type).resolve()
    out_folder_resolved.mkdir(parents=True,exist_ok=True)
    model_params["OUTPUT_PATH"] = str(out_folder_resolved)

    df_train = df_train[df_train['question_type']==question_type]
    df_validation = df_validation[df_validation['question_type']==question_type]
    df_test = df_test[df_test['question_type']==question_type]

print('model_params:' ,model_params)

model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])


if "_soft_prompt_patch" in architecture:
    if size == 'large':
        init_lr = optimal_init_map[seed]['single_model_with_token_random_token_init']
        patch_path = root/f'src/models-large-{seed}/single_model_with_token_random_token_init/{init_lr}/model_files'
    else:
        raise NotImplementedError('Need to manually add optimal map')
    patch_model = T5ForConditionalGeneration.from_pretrained(patch_path)
    patch_tokenizer = T5Tokenizer.from_pretrained(patch_path)
    patch_embeddings = patch_model.shared.weight

    tokenizer = patch_tokenizer

    # Resize embeddings
    with torch.no_grad():
        model.resize_token_embeddings(patch_embeddings.shape[0])
        model.shared.weight[32000:] = patch_embeddings[32000:]

    del patch_model
    del patch_tokenizer
    del patch_embeddings
elif architecture == 't5_wta_patch':
    if size == 'large':
        init_lr = optimal_init_map[seed]['ghanem_et_al_2022_true']
        patch_path = root/f'src/models-large-{seed}/ghanem_et_al_2022_true/{init_lr}/model_files'
    else:
        raise NotImplementedError('Need to manually add optimal map')
    patch_model = T5ForConditionalGeneration.from_pretrained(patch_path)
    patch_tokenizer = T5Tokenizer.from_pretrained(patch_path)

    tokens_to_add = []
    for prompt in t5_wta_patch_prompt_map.values():
        for token in re.split(r'(?=<)',prompt):
            if token == '':
                continue
            tokens_to_add.append(token)
    print('tokens_to_add:',tokens_to_add)
    patch_tokenizer.add_tokens(tokens_to_add)

    patch_embeddings = patch_model.shared.weight

    tokenizer = patch_tokenizer

    # Resize embeddings
    with torch.no_grad():
        model.resize_token_embeddings(patch_embeddings.shape[0])
        # model.shared.weight[32000:] = patch_embeddings[32000:]
        for init,new_toks in zip(ghanem_et_al_2022_prompt_map.values(),t5_wta_patch_prompt_map.values()):
            init_tokens = patch_tokenizer.encode(init)[:-1]
            new_tokens = patch_tokenizer.encode(new_toks)[:-1]
            assert len(init_tokens) == len(new_tokens)
            model.shared.weight[new_tokens] = patch_embeddings[init_tokens].detach().clone()

    del patch_model
    del patch_tokenizer
    del patch_embeddings

device_map = None
if size in ['11b']:
    # device_map = {
    #     0: [0,  1,  2,  3,  4,  5],
    #     1: [6,  7,  8,  9,  10, 11],
    #     2: [12, 13, 14, 15, 16, 17],
    #     3: [18, 19, 20, 21, 22, 23]
    # }
    
    device_map = {
        0: [0,  1,  2],
        1: [3,  4,  5,  6,  7,  8,  9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23]
    }
    model_params["TRAIN_BATCH_SIZE"] = 1
    model_params["GRADIENT_ACCUMULATION_BATCH_SIZE"] = 8
    model.parallelize(device_map)
elif size in ['3b']:
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        1: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
else:
    # model_params["TRAIN_BATCH_SIZE"] = 8
    # model_params["GRADIENT_ACCUMULATION_BATCH_SIZE"] = 1
    # device_map = {
    #     0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     1: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    # }
    # model.parallelize(device_map)
    model.cuda()
# model.cuda()

if architecture == 'single_model_with_token_random_token_init':
    new_tokens = [
        tok for i in range(token_prefix_length)
        for tok in
        [
            f"<{x}{i}>" for x in question_types
        ]

    ]

    tokenizer.add_tokens(new_tokens)

    def get_random_regular_token_id(max_id:int):
        while True:
            random_id = np.random.randint(0, max_id)
            if random_id not in tokenizer.all_special_ids:
                return random_id


    # Resize embeddings
    with torch.no_grad():
        old_embeddings = model.get_input_embeddings()
        old_token_count = old_embeddings.weight.shape[0]
        new_embeddings = model._get_resized_embeddings(old_embeddings, old_token_count+len(new_tokens))
        for tok_str in new_tokens:
            new_tok = tokenizer.encode(tok_str)[0]
            new_embeddings.weight[new_tok] = old_embeddings.weight[get_random_regular_token_id(old_token_count)].detach().clone()

        model.set_input_embeddings(new_embeddings)

        old_lm_head = model.get_output_embeddings()
        old_lm_head_count = old_embeddings.weight.shape[0]
        new_lm_head = model._get_resized_lm_head(old_lm_head, old_lm_head_count+len(new_tokens))
        for tok_str in new_tokens:
            new_tok = tokenizer.encode(tok_str)[0]
            new_lm_head.weight[new_tok] = old_lm_head.weight[get_random_regular_token_id(old_token_count)].detach().clone()

        model.set_output_embeddings(new_lm_head)

        model.config.vocab_size = new_embeddings.weight.shape[0]

training_loader, validation_loader, _ = true_trainer.build_data(tokenizer, dataframes=[df_train, df_validation, df_test], source_text="input_text", target_text="target_text", model_params=model_params)
true_trainer.T5Trainer(model, training_loader, validation_loader, tokenizer, model_params=model_params)
# %%
