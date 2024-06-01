#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import copy
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm

from E_eval_utils import best_set_match_eval, top_reference_eval, cartesian_eval, neg_fbd_dist_eval, jaccard4_dist_eval

################   CONFIG   ################

src_dir = Path(__file__).parent.resolve()
results_dir = src_dir/'results_all_metrics_seed_control'
# results_dir = src_dir/'results_all_metrics_dreamscape.old'
test_set_name = 'challenge'
# test_set_name = 'test'

############################################

def load_json(path):
    with open(path) as f:
        try:
            return json.load(f)
        except:
            print('Error for',path)
            raise

def load_df(prefix,metric='',dataset=''):
    rows = []
    for path in tqdm(list(results_dir.glob(prefix+'*'))):
        # if 'single_model_with_token' not in path.name:
        #     continue
        # if "single_model_with_sentence_prompt" not in path.name:
        #     continue
        if not metric in path.name or not dataset in path.name:
            continue
        data = load_json(path)
        # Delete big data array (this array can be useful to statistical tests)
        # data['scores'] = [np.mean(x) for x in data['scores']]
        rows.append(data)
    # assert len(rows) == 9
    df = pd.DataFrame(rows)
    df = df.sort_values('top_p_value')
    return df

# %%
dataset = test_set_name
metric = ''
# dataset = 'challenge'
df_all_dev = load_df('ensemble',dataset='dev',metric=metric)
df_all_test = load_df('ensemble',dataset=test_set_name,metric=metric)


df_all_dev = df_all_dev[~df_all_dev['architecture'].str.contains('silver')]
# df_all_test = df_all_test[df_all_test['size']=='11b']
#%%
# Filter df_all_test
df_dev_for_join = df_all_dev.sort_values('mean_score', ascending=False).groupby(['architecture', 'seed', 'top_p_value']).head(1)
df_dev_for_join[df_dev_for_join['metric']=='meteor']
# %%
df_dev_for_join = df_all_dev.sort_values('mean_score', ascending=False).groupby(['architecture', 'seed', 'metric']).head(1)
df_dev_for_join = df_dev_for_join[~df_dev_for_join['architecture'].str.contains('max')]
df_dev_for_join = df_dev_for_join[df_dev_for_join['metric']=='meteor']
join_columns = ['top_p_value','decision_variable','architecture','learning_rate','seed']
df_all = df_all_test.merge(df_dev_for_join[join_columns],how='inner',on=join_columns)
df_all['mean_score'][df_all['metric']!='sacrebleu']*=100
#%%
df_out = df_all.groupby('architecture').agg(['mean','std'])['mean_score'].reset_index().sort_values('mean', ascending=False)
df_out
# %%
df_dev_for_join['hyperparameters'] = df_dev_for_join.apply(lambda x: f"\makecell{'{'}p = {x['top_p_value']}\\\\ Learning rate = {x['learning_rate']}{'}'}",axis=1)
out_hyper_init_table = df_dev_for_join[['hyperparameters','architecture','seed','metric']].sort_values(['architecture','seed'])
# out_hyper_table = out_hyper_init_table.drop_duplicates().set_index(["metric", "architecture"]).unstack(level=0)
# out_hyper_table['architecture'] = sorted(out_hyper_init_table['architecture'].unique())
print(out_hyper_init_table.style.hide_index()
    .format(
        precision = 2,
        escape="latex",
    )
    .to_latex(
        environment='table*',
        hrules=True,
        column_format='|c|c|c|',
        index=False
    )
)
#%%
target_architectures = {
    # 'ghanem_et_al_2022': 'T5 WTA',
    'ghanem_et_al_2022_t5_lm': 'T5 WTA',
    # 'separate_models_t5_lm': 'IMoE',
    # 'single_model_t5_lm': 'No Control',
    'control_t5_lm': 'SoftSkillQG',
    't5_wta_control_init_striped': 'SoftSkillQG (with T5 repeating token init)',
    # 'soft_attempt': 'Soft-prompting',
    # 'single_model_with_soft_prompt_t5_lm': 'No Control with soft-prompt',
}
best_param_test_set_df = df_all[df_all['metric']=='meteor']
best_param_test_set_df = best_param_test_set_df.sort_values('architecture')

#%%
# best_param_test_set_df = df_all_test
#%%
def get_score(all_preds,all_labels,score_function):
    return np.mean([score_function(preds,labels)[0] for labels, preds in tqdm(zip(all_labels,all_preds),total=len(row.predictions),leave=False)])

out_rows = []
for row in tqdm(best_param_test_set_df.itertuples(),total=len(best_param_test_set_df)):
    assert row.dataset == test_set_name
    if row.architecture not in target_architectures:
        continue
    arch = target_architectures[row.architecture]
    predictions = row.predictions
    # predictions = [[y if y.strip()!='' else '<blank>' for y in x] for x in row.predictions]
    labels = row.labels
    out_rows.append({
        'architecture': arch,
        'multi_meteor': get_score(predictions,labels,best_set_match_eval),
        'best_reference': get_score(predictions,labels,top_reference_eval),
        'cartesian': get_score(predictions,labels,cartesian_eval),
        'fbd': get_score(predictions,labels,neg_fbd_dist_eval),
        'jaccard4': get_score(predictions,labels,jaccard4_dist_eval),
        'self_similarity': get_score([[y for y in x if y.strip()] for x in predictions],[[y for y in x if y.strip()] for x in predictions],cartesian_eval),
        'seed': row.seed,
    })
out_df = pd.DataFrame(out_rows)
#%%
final_out_df = out_df.groupby('architecture').mean().reset_index()
final_out_df = final_out_df[
    [
        'architecture', 
        'multi_meteor', 
        'best_reference', 
        'cartesian', 
        'jaccard4',
        'fbd',
        'self_similarity',
    ]
]
for column in final_out_df.columns:
    if column in ['architecture']:
        continue
    if column in ['fbd']:
        final_out_df[column] = final_out_df[column]*-1
        continue
    final_out_df[column] = final_out_df[column]*100
print(final_out_df.style
    .hide_index()
    .format(
        precision = 2,
        escape="latex",
    )
    .to_latex(
        environment='table*',
        hrules=True,
        column_format=len(out_df.columns)*'|c'+'|',
    )
)

#%%
# out_init_table = df_all[['mean_score','architecture','metric']].sort_values('architecture')
# out_init_table.drop_duplicates(['mean_score','architecture','metric'],inplace=True)
# out_table = out_init_table.set_index(["metric", "architecture"]).unstack(level=0)
# out_table['architecture'] = sorted(out_init_table['architecture'].unique())
# print(out_table.style
#     .hide_index()
#     .format(
#         precision = 2,
#         escape="latex",
#     )
#     .to_latex(
#         environment='table*',
#         hrules=True,
#         column_format='|l|l|l',
#     )
# )
#%%
out_df.drop(1).groupby('architecture').mean().reset_index()
# %%
