#%%
# import sys
# sys.path.append('..')

import torch
from dataclasses import fields
from transformers import AutoTokenizer

from attempt.adapters import AutoAdapterConfig
from attempt.third_party.models import T5Config, T5ForConditionalGeneration
from attempt.options import AdapterTrainingArguments, ModelArguments
from attempt.utils import freeze_model_params

#### Config ####

# base_model = 't5-base'
# output_dir='outputs/soft_prompt_quail'
# device = torch.device('cuda')
# tasks = ['Belief_states']

################

#%%

def get_adapter_config(adapter_args, config, device, output_dir, tasks):
    if adapter_args.train_task_adapters or adapter_args.prefix_tuning or adapter_args.bitfit:
        adapter_config = AutoAdapterConfig.get(
            adapter_args.adapter_config_name)
        adapter_config.input_dim = config.d_model

        adapter_config.tasks = tasks # ! Might need to wrap this in a list
        adapter_params = [field.name for field in fields(adapter_args)]
        for p in adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p) and\
                    getattr(adapter_args, p) is not None:
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                print(
                    f"Warn: ({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = device
        adapter_config.output_dir = output_dir
        adapter_config.attn_method = config.attn_method
        adapter_config.ignore_target = config.ignore_target
        adapter_config.attn_prefix = config.attn_prefix_tuning
        adapter_config.fix_attention = config.fix_attention
    else:
        adapter_config = None
    return adapter_config

def modify_model_after_init(model, adapter_args, adapter_config):
    # Freezes model parameters.
    freeze_model_params(model, adapter_args, adapter_config)

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(
        "***** Model Trainable Parameters {} *****".format(trainable_params)
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("##### Parameter name %s" % name)
    total_lm_head_params = sum(p.numel()
                                for p in model.lm_head.parameters())
    total_trainable_params = sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)
    total_trainable_bias_params = sum(p.numel(
    ) for n, p in model.named_parameters() if p.requires_grad and n.endswith(".b"))
    total_trainable_layernorm_params = sum(p.numel() for n, p in model.named_parameters(
    ) if p.requires_grad and ".layer_norm.weight" in n)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters %s" % total_trainable_params)
    print("Total trainable bias parameters %s" %
                total_trainable_bias_params)
    print("Total trainable layer norm parameters %s" %
                total_trainable_layernorm_params)
    print("Total parameters %s" % total_params)
    t5_base_params = 222882048
    # total params since we have 8 task, it is Y = 1*BERT + 8*ADAPTERS, and final number is Y/BERT ("1.3x")
    total_params_ratio = ((total_params-t5_base_params)
                            * 8+t5_base_params)/t5_base_params
    total_trainable_params_percent = (
        total_trainable_params/t5_base_params)*100
    total_trainable_bias_params_percent = (
        total_trainable_bias_params/total_trainable_params)*100
    total_trainable_layernorm_params_percent = (
        total_trainable_layernorm_params/total_trainable_params)*100
    total_trainable_lm_head_params_percent = (
        total_lm_head_params/t5_base_params)*100
    print("For adapters/prompt-tuning, total params %s" %
                total_params_ratio)
    print("For intrinsic, total params %f" %
                (total_params/t5_base_params))
    print("Total trainable params %f" %
                total_trainable_params_percent)
    print("Total trainable bias params %f" %
                total_trainable_bias_params_percent)
    print("Total trainable layer norm params %f" %
                total_trainable_layernorm_params_percent)
    print("Total lm_head params %f" %
                total_trainable_lm_head_params_percent)
    return model
# %%
def get_soft_prompt_model_and_tokenizer(base_model, output_dir, device, tasks, prompt_length=10):
    model_args = ModelArguments(
        model_name_or_path=base_model,
        tokenizer_name=base_model,
        save_prefix_only=False,
    )

    adapter_args = AdapterTrainingArguments(train_task_adapters=False,
        non_linearity='gelu_new',
        prefix_tuning=True,
        prefix_dim=prompt_length, # Default value is 100
        init_prefix_from_vocab=True,
    )

    config = T5Config.from_pretrained(
        base_model,
    )
    config.prefix_tuning = adapter_args.prefix_tuning
    config.attn_prefix_tuning = True if len(tasks) > 1 else False
    config.attn_method = "sub" if len(tasks) > 1 else model_args.attn_method
    config.ignore_target = model_args.ignore_target
    config.shared_attn = True if len(tasks) > 1 else False
    config.prefix_num = model_args.prefix_num
    config.num_target = len(tasks)
    config.prefix_num = len(tasks)
    # config.temperature = 2000
    config.learned_temperature = model_args.learned_temperature
    config.fix_attention = model_args.fix_attention

    adapter_config = get_adapter_config(adapter_args, config, device, output_dir, tasks)
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        adapter_config=adapter_config,
    )
    model = modify_model_after_init(
        model, 
        adapter_args, 
        adapter_config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return model, tokenizer
# %%
