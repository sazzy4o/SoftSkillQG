
#%%
import contextlib
import json
import gc
import numpy as np
import pandas as pd
import sys
import torch

from functools import lru_cache
from pathlib import Path
from htawta_true import trainer as true_trainer
from transformers import T5Tokenizer
from types import MethodType
from tqdm.autonotebook import tqdm

from lib.t5 import T5ForConditionalGenerationWithConfidence, T5ForConditionalGeneration
from lib.t5_model import T5Model
from attempt.dataset import AttemptDataSetClass
from attempt.model import get_soft_prompt_model_and_tokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

root_dir = Path(__file__).parent.resolve().parent
args = sys.argv[1:]
# args = ['lora_only_t5_lm','1e-4','dev','quail', 'large','1']

# model_folder = root_dir/'src/single_models/Belief_states'
architecture = args[0]
learning_rate = args[1]
dataset_subset = args[2]
dataset = args[3]
size = args[4]
seed = int(args[5])

base_model_name = f't5-{size}'

model_dir_name = architecture
models_base_name = f'models-{size}-{seed}-control' # ! TODO: Make sure this is updated
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
    out_dir = root_dir/'src/perplexity_loss_results_seed'
else:
    model_dir = Path(f'{models_base_name}-{dataset}')/architecture/learning_rate
    out_dir = root_dir/f'src/perplexity_loss_results_seed_{dataset}'

token_prefix_length = 20

print('Args: ', ', '.join(args))

#######################################

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
    if architecture == 'single_model_with_token_random_token_init' or architecture == 'single_model_soft_prompt_patch' or architecture == 't5_wta_control_init_striped' or architecture == 't5_wta_control_init_start' or architecture == 'control_t5_lm' or architecture == ' t5_wta_control_init_striped_t5_lm' or architecture == 't5_wta_control_init_start_t5_lm':
        df['input_text'] = df.apply(lambda x: ''.join([f'<{x.question_type}{i}>' for i in range(token_prefix_length)])+ x.input_text, axis=1)
    elif architecture == 'ghanem_et_al_2022_true' or architecture == 'ghanem_et_al_2022_t5_lm':
        df['input_text'] = df.apply(lambda row: ghanem_et_al_2022_prompt_map[row['question_type']]+' </s> '+ row['input_text'], axis=1)
    elif architecture == 't5_wta_patch' or architecture == 't5_wta_control_length' or architecture == 't5_wta_control_length_and_init':
        df['input_text'] = df.apply(lambda row: t5_wta_patch_prompt_map[row['question_type']]+' </s> '+ row['input_text'], axis=1)
    elif architecture == 'single_model_with_soft_prompt' or architecture == 'single_model_with_soft_prompt_t5_lm':
        df['input_text'] = df.apply(lambda x: ''.join([f'<SP{i}>' for i in range(token_prefix_length)])+ x.input_text, axis=1)


    df['prefix'] = '' # We add the prefix transparantly above
    return df

root = Path(__file__).parent.resolve().parent
if dataset == 'quail':
    soft_prompt_id_map = {
        'Belief_states': 0,
        'Causality': 1,
        'Character_identity': 2,
        'Entity_properties': 3,
        'Event_duration': 4,
        'Factual': 5,
        'Subsequent_state': 6,
        'Temporal_order': 7,
    }
    df = load_dataset_df(root/f'data/quail/quail_v1.3/json/{dataset_subset}.jsonl')
elif dataset == 'dreamscape':
    df = load_dataset_df(root/f'data/dreamscape/{dataset_subset}_v2.jsonl')
else:
    raise NotImplementedError('Dataset not currently supported, map to QUAIL format and add another case here...')

df = df.sort_values(by=['question_type','input_text'])
context_texts = df.input_text.to_list()
question_types = df.question_type.to_list()
labels = df.target_text.to_list()

#%%
model_params = {
    "OUTPUT_PATH": str(model_dir),
    "MODEL": base_model_name,                  
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 128,
    "TRAIN_BATCH_SIZE": 1,
    "VALID_BATCH_SIZE": 1,
}

print('Model Params', model_params)
#%%
def get_perplexity(base_model, tokenizer, full_context, label, question_type):
    label_ids = tokenizer.encode(label)

    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None): # Way too many logs
        input_df = pd.DataFrame([{'context':full_context,'question':'','task':question_type}])
        _, _, test_loader = true_trainer.build_data(
            tokenizer, 
            dataframes=[input_df, input_df, input_df],
            source_text="context", 
            target_text="question", 
            model_params=model_params,
            dataset_class=AttemptDataSetClass if 'attempt' in architecture else true_trainer.YourDataSetClass,
        )
        input_batch = list(test_loader)[0]
        input_ids = input_batch["source_ids"].to(base_model.device)
        attention_mask = input_batch["source_mask"].to(base_model.device)

    kwargs = {}
    if 'attempt' in architecture:
        kwargs['task'] = question_type
        # kwargs['task_ids'] = data['task_ids'] # ! TODO

    perplexity = base_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache_label_seq_ids=label_ids,
        do_sample=True,
        return_dict_in_generate=True,
        **kwargs
    )
    return perplexity

# This is kind of hacky, but it avoids a lot 
# of easy mistakes preprocessing the input
# (T5 preprocessing has a lot of steps)
def sample(
    self,
    input_ids: torch.LongTensor,
    use_cache_label_seq_ids = None,
    **model_kwargs,
):
    softmax_probs = []
    for tok in use_cache_label_seq_ids:
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True
        )

        next_token_logits = outputs.logits[:, -1, :]

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_tokens = torch.tensor([tok]).to(self.device)
        softmax_probs.append(float(probs[0][tok].detach().cpu()))

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
    
    # We use log probs to avoid underflow and preserve precision
    log_prob = np.log(softmax_probs)
    log_prob_sum = np.sum(log_prob)
    probs_count = len(softmax_probs)

    perplexity = float(np.exp(-1/probs_count * log_prob_sum))

    return perplexity

@lru_cache(maxsize=1)
def get_lora_main_model():
    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    model._validate_model_kwargs = lambda y: None
    return model

@lru_cache(maxsize=1) # Make sure we only load the model once
def get_model(model_dir_base, question_type):
    gc.collect()  # This is so that old models memmory get freed up (including on the GPU)
    model_args = {
        "preprocess_inputs": False,
        "reprocess_input_data": True,
        "max_seq_length": 512,
        "use_multiprocessing": False,
        "use_multiprocessed_decoding": False,

        "do_sample": True,
        "max_length": 50,
    }
    if 'separate_models' in architecture or architecture == 'soft_attempt' or architecture == 'soft_attempt_reverse' or architecture=='separate_models_t5_lm':
        model_dir_base = model_dir_base/question_type

    model_dir_full = str(model_dir_base/'model_files')

    if 'attempt' in architecture:
        question_types = [question_type]
        if architecture == 'soft_skill_attempt':
            question_types = list(ghanem_et_al_2022_prompt_map.keys())
        model, tokenizer = get_soft_prompt_model_and_tokenizer(
            model_dir_full, 
            model_dir_full, 
            'cuda', 
            question_types,
        )
        base_model = model.cuda()
    elif 'lora_only' in architecture:
        base_model_name = 't5-large'
        if 't5_lm' in architecture:
            base_model_name = 'google/t5-large-lm-adapt'
        model = get_lora_main_model()
        tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        lora_dir = model_dir_base/question_type/'model_files/lora_only'
        lora_model = PeftModel.from_pretrained(model, lora_dir)
        base_model = lora_model.merge_and_unload().cuda()
    else:
        model = T5Model('t5c',model_dir_full, args=model_args)
        if size == '11b':
            device_map = {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                1: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            }
            model.model.parallelize(device_map)
        else:
            model.model.cuda()
        base_model = model.model
        tokenizer = model.tokenizer
    base_model.sample = MethodType(sample, base_model)
    base_model._validate_model_kwargs = lambda y: None
    base_model.eval()
    return base_model, tokenizer

base_model = None
with torch.inference_mode():
    perplexities = []
    for question_type, context_text, label in tqdm(zip(question_types, context_texts, labels),total=len(labels)):
        if 'separate_models' in architecture or architecture == 'soft_attempt' or architecture == 'soft_attempt_reverse' or 'lora_only' in architecture or architecture=='separate_models_t5_lm':
            question_type_key = question_type
        else:
            question_type_key = None # Ignore question type when loading model
        del base_model
        base_model, tokenizer = get_model(model_dir, question_type_key)
        perplexities.append(get_perplexity(base_model, tokenizer, context_text, label, question_type))
mean_perplexity = float(np.mean(perplexities))
print('Mean perplexity:',mean_perplexity)
# %%
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir/f'{architecture}_{size}_{dataset}_{dataset_subset}_{learning_rate}_{seed}.json','w') as f:
    json.dump(
        {
            'mean_perplexity':float(np.mean(perplexities)),
            'median_perplexity':float(np.median(perplexities)),
            'std_perplexity':float(np.std(perplexities)),
            'perplexities':perplexities,
            'architecture': architecture,
            'learning_rate': learning_rate,
            'dataset_subset': dataset_subset,
            'dataset': dataset,
            'size': size,
            'seed': seed,
        },
    f)
# %%
