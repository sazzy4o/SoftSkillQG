#%% Based on https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

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
from attempt.model import get_soft_prompt_model_and_tokenizer
from attempt.dataset import AttemptDataSetClass
from attempt.utils import save_prompts

# Speed things up
# torch.backends.cuda.matmul.allow_tf32 = True

root = (Path(__file__).parent/'../').resolve()

args = sys.argv[1:]
# args = ['soft_skill_attempt_patch','1e-4', 'quail', 'large', '2'] # ! TODO: Remove this line
############ Config ############

architecture = args[0]
learning_rate = float(args[1])
token_prefix_length = 100

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

base_model = 'google/t5-large-lm-adapt'

model_dir_name = architecture

models_base_name = f'models-{size}-{seed}-control'

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
    if dataset == 'dreamscape':
        ghanem_et_al_2022_prompt_map = {
            'Basic Story Elements': 'Basic Story Elements',
            'Summarizing': 'Summarizing',
            'Vocabulary': 'Vocabulary',
            'Figurative Language': 'Figurative Language',
            'Inferring': 'Inferring',
            'Close Reading': 'Close Reading',
            'Predicting': 'Predicting',
            'Character Traits': 'Character Traits',
            'Visualizing': 'Visualizing'
        }
    elif dataset == 'fairytaleqa':
        ghanem_et_al_2022_prompt_map = {
            'character': 'Character',
            'feeling': 'Feeling',
            'action': 'Action',
            'setting': 'Setting',
            'prediction': 'Prediction',
            'outcome resolution': 'Outcome resolution',
            'causal relationship': 'Causal relationship'
        }
    else:
        raise NotImplementedError('Dataset not currently supported, map to QUAIL format and add another case here...')

if architecture == 'control_t5_lm' or architecture == 't5_wta_control_init_striped_t5_lm' or architecture == 't5_wta_control_init_start_t5_lm' or architecture == 'single_model_with_soft_prompt_t5_lm':
    token_prefix_length = 20

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
    df_train = load_dataset_df(root/'data/dreamscape/train_v4.jsonl')
    df_validation = load_dataset_df(root/'data/dreamscape/dev_v4.jsonl')
    df_test = load_dataset_df(root/'data/dreamscape/test_v4.jsonl')
    question_types = sorted(list(set(df_train['question_type'])))
elif dataset == 'fairytaleqa':
    df_train = load_dataset_df(root/'data/fairytaleqa/train.jsonl')
    df_validation = load_dataset_df(root/'data/fairytaleqa/validation.jsonl')
    df_test = load_dataset_df(root/'data/fairytaleqa/test.jsonl')
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
    "TRAIN_BATCH_SIZE": 4,
    "GRADIENT_ACCUMULATION_BATCH_SIZE": 2, # Equivalent to batch size 8 (slower, but lower memory)
    "VALID_BATCH_SIZE": 8,        
    "TRAIN_EPOCHS": epochs,           
    "LEARNING_RATE": learning_rate,
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 64,
    "early_stopping_patience": 3,  
}

if architecture=='separate_models' or architecture=='soft_attempt':
    question_type = args[5]
    print('Question type:',question_type)
    out_folder_resolved = (out_folder/question_type).resolve()
    out_folder_resolved.mkdir(parents=True,exist_ok=True)
    model_params["OUTPUT_PATH"] = str(out_folder_resolved)

    df_train = df_train[df_train['question_type']==question_type]
    df_validation = df_validation[df_validation['question_type']==question_type]
    df_test = df_test[df_test['question_type']==question_type]

print('model_params:' ,model_params)

if architecture == 'soft_attempt':
    model, tokenizer = get_soft_prompt_model_and_tokenizer(
        base_model, 
        model_params["OUTPUT_PATH"], 
        'cuda', 
        [question_type]
    )
elif architecture == 'soft_skill_attempt':
    model, tokenizer = get_soft_prompt_model_and_tokenizer(
        base_model, 
        model_params["OUTPUT_PATH"], 
        'cuda', 
        question_types
    )
    # Unfreeze all weights
    for param in model.parameters():
        param.requires_grad = True
elif architecture == 'soft_skill_attempt_patch':
    model, tokenizer = get_soft_prompt_model_and_tokenizer(
        base_model, 
        model_params["OUTPUT_PATH"], 
        'cuda', 
        question_types
    )
    seed_data = json.load(open(root/'src/soft_attempt_seed_data.json'))[str(seed)]

    for i, q_type in enumerate(question_types):
        prefix_path = root/'src'/seed_data[q_type]
        # prefix_path = root/'src'/f'models-large-2-v2/soft_attempt/2.5e-1/{q_type}/prefix_embeddings.pt'
        model.prefix_shared.data[i] = torch.load(prefix_path)

    # Unfreeze all weights
    for param in model.parameters():
        param.requires_grad = True
else:
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])


if "_soft_prompt_patch" in architecture:
    if size == '3b':
        patch_path = root/'src/models-3b/single_model_with_token_random_token_init/5e-5/model_files'
    elif size == 'large':
        patch_path = root/'src/models-large/single_model_with_token_random_token_init/1e-5/model_files'
    elif size == 'base':
        patch_path = root/'src/models-base/single_model_with_token_random_token_init/1e-5/model_files'
    elif size == 'small':
        patch_path = root/'src/models-small/single_model_with_token_random_token_init/5e-5/model_files'
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
        patch_path = root/'src/models-large/ghanem_et_al_2022_true/5e-5/model_files'
    else:
        raise NotImplementedError('Only large size supported for now')
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

model = model.cuda()

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

if 'attempt' in architecture:
    df_train['task'] = df_train['question_type']
    df_validation['task'] = df_validation['question_type']
    df_test['task'] = df_test['question_type']
    training_loader, validation_loader, _ = true_trainer.build_data(
        tokenizer, 
        dataframes=[df_train, df_validation, df_test], 
        source_text="input_text", 
        target_text="target_text", 
        model_params=model_params,
        dataset_class=AttemptDataSetClass,
    )
else:
    training_loader, validation_loader, _ = true_trainer.build_data(
        tokenizer,
        dataframes=[df_train, df_validation, df_test], 
        source_text="input_text", 
        target_text="target_text", 
        model_params=model_params
    )

true_trainer.T5Trainer(model, training_loader, validation_loader, tokenizer, model_params=model_params)
# %%
if architecture=='soft_attempt' or architecture == 'soft_skill_attempt':
    save_prompts(model, model_params["OUTPUT_PATH"], False, False, 0, None)