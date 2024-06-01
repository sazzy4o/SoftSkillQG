#%%
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
#%%
def load_jsonl_df(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f.readlines()]
    return pd.DataFrame(rows)
df_test = load_jsonl_df(src_dir.parent/'data/quail/quail_v1.3/json/challenge.jsonl')
#%%
out_dir = src_dir/'human_annotation_files'

human_eval_csv_paths = out_dir.glob('Batch_51727*.csv')

df_human_eval_raw = pd.concat([pd.read_csv(path) for path in human_eval_csv_paths])
# %%
df_human_eval_rows = []
for _,row in df_human_eval_raw.iterrows():
    for i in range(1, 8):
        q = 'q'+str(i)

        fluency_list = [
            row[f'Answer.{q}_fluency_1.context_specific'],
            row[f'Answer.{q}_fluency_2.not_context_specific'],
            row[f'Answer.{q}_fluency_3.not_context_specific'],
            row[f'Answer.{q}_fluency_4.not_context_specific'],
            row[f'Answer.{q}_fluency_5.not_context_specific'],
        ]

        fluency = fluency_list.index(True) + 1

        df_human_eval_rows.append({
            'HITId': row.HITId,
            'HITTypeId': row.HITTypeId,
            'WorkerId': row.WorkerId,
            'context': row[f'Input.context'],
            'context_id': row[f'Input.context_id'],
            'row_id': row[f'Input.row_id'],
            'comment': row[f'Answer.{q}_comment'],
            'answerable': row[f'Answer.{q}_answerable.answerable'],
            'fluency': fluency,
            'context_specific': row[f'Answer.{q}_context_specific.context_specific'],
            'architecture': row[f'Input.{q}_id'],
            'question': row[f'Input.{q}'],
        })
df_human_eval = pd.DataFrame(df_human_eval_rows)
df_human_eval.to_csv(out_dir/'human_eval.csv', index=False)
#%%
from statsmodels.stats import inter_rater as irr
fleiss_rows = []
for name, group in df_human_eval.groupby(['architecture', 'HITId', 'row_id','question','context_id']):
    answerable_true_count = sum(group['answerable'] == True)
    answerable_false_count = sum(group['answerable'] == False)
    fleiss_rows.append([
        answerable_true_count,
        answerable_false_count
    ])
fleiss_val = irr.fleiss_kappa(np.array(fleiss_rows), method='fleiss')
print('Fleiss Kappa:', fleiss_val)
all_agree = len([x for x in fleiss_rows if max(x)==3])
all_agree_percent = all_agree / len(fleiss_rows)*100
print('All Agree:', all_agree_percent)
#%%
from statsmodels.stats import inter_rater as irr
fleiss_rows = []
for name, group in df_human_eval.groupby(['architecture', 'HITId', 'row_id','question','context_id']):
    context_specific_true_count = sum(group['context_specific'] == True)
    context_specific_false_count = sum(group['context_specific'] == False)
    fleiss_rows.append([
        context_specific_true_count,
        context_specific_false_count
    ])
irr.fleiss_kappa(np.array(fleiss_rows), method='fleiss')
fleiss_val = irr.fleiss_kappa(np.array(fleiss_rows), method='fleiss')
print('Fleiss Kappa:', fleiss_val)
all_agree = len([x for x in fleiss_rows if max(x)==3])
all_agree_percent = all_agree / len(fleiss_rows)*100
print('All Agree:', all_agree_percent)
#%%
from statsmodels.stats import inter_rater as irr
fleiss_rows = []
for name, group in df_human_eval.groupby(['architecture', 'HITId', 'row_id','question','context_id']):
    context_specific_1 = sum(group['fluency'] == 1)
    context_specific_2 = sum(group['fluency'] == 2)
    context_specific_3 = sum(group['fluency'] == 3)
    context_specific_4 = sum(group['fluency'] == 4)
    context_specific_5 = sum(group['fluency'] == 5)

    fleiss_rows.append([
        context_specific_1,
        context_specific_2,
        context_specific_3,
        context_specific_4,
        context_specific_5,
    ])
fleiss_val = irr.fleiss_kappa(np.array(fleiss_rows), method='fleiss')
print('Fleiss Kappa:', fleiss_val)
all_agree = len([x for x in fleiss_rows if max(x)==3])
all_agree_percent = all_agree / len(fleiss_rows)*100
print('All Agree:', all_agree_percent)
# %%
agg_var = 'mean'
df_human_eval_out = df_human_eval.copy()
df_human_eval_out['context_specific'] = df_human_eval_out['context_specific'].astype(int)*100
df_human_eval_out['answerable'] = df_human_eval_out['answerable'].astype(int)*100
df_human_eval_out = df_human_eval_out.groupby(['architecture', 'HITId', 'row_id','question','context_id']).agg(
    {'fluency': 'mean', 'answerable': 'median', 'context_specific': 'median'}
).reset_index()
df_human_eval_out.groupby('architecture').agg({'fluency': agg_var, 'answerable': agg_var, 'context_specific': agg_var}).reset_index()
df_human_eval_out.to_csv(out_dir/'human_eval_out.csv', index=False)
# %%
# t-test
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(
    df_human_eval_out[df_human_eval_out['architecture']=='single_model_t5_lm']['context_specific'], 
    df_human_eval_out[df_human_eval_out['architecture']=='soft_attempt']['context_specific']
)
p_val

# %%
target_architectures = {
    # 'ghanem_et_al_2022': 'T5 WTA',
    'ghanem_et_al_2022_t5_lm': 'T5 WTA',
    'separate_models_t5_lm': 'IMoE',
    'soft_attempt': 'Soft-prompting',
    'single_model_t5_lm': 'No Control',
    'single_model_with_soft_prompt_t5_lm': 'Constant soft-prompt',
    'control_t5_lm': 'SoftSkillQG',
    'reference_question': 'Human Created'
}
order = {x:i for i,x in enumerate(target_architectures.values())}

df_human_eval_out_arch = df_human_eval_out.copy()
df_human_eval_out_arch['architecture'] = df_human_eval_out_arch['architecture'].apply(lambda x: target_architectures[x])

df_plot = df_human_eval_out_arch.groupby('architecture').agg({'fluency': agg_var, 'answerable': agg_var, 'context_specific': agg_var}).reset_index()
df_plot['order'] = df_plot['architecture'].apply(lambda x: order[x])
df_plot = df_plot.sort_values('order')

# %%
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

ax1.set_xlabel('Model')
ax1.set_ylabel('Fluency')
ax1.set_ylim(0, 5)
ax1.set_yticks([1,2,3,4,5])
plt.xticks(rotation=90)

ax1.bar(df_plot['architecture'], df_plot['fluency'], color='tab:blue')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Answerability/Context Specificity (%)')
ax2.set_ylim(0, 100)

ax2.bar(df_plot['architecture'], df_plot['answerable'], color='tab:orange')
# %%
df_plot[['answerable','fluency','context_specific']].plot.bar(rot=90)
# %%
df_plot_2 = df_plot[['architecture','answerable','fluency','context_specific']].copy().set_index('architecture')

df_plot_2['answerable'] = df_plot_2['answerable'] / 20
df_plot_2['context_specific'] = df_plot_2['context_specific'] / 20

df_plot_2.rename(
    columns={
        'answerable': 'Answerability',
        'context_specific': 'Context Specificity',
        'fluency': 'Fluency'
    }, 
    inplace=True,
)

fig, ax1 = plt.subplots()

transpose_order = ['Fluency', 'Answerability','Context Specificity']

df_plot_2[transpose_order].transpose().plot.bar(rot=90, ax=ax1)

# ax1.set_xlabel('Model')
ax1.set_ylabel('Fluency')
ax1.set_ylim(0, 5)
ax1.set_yticks([1,2,3,4,5])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Answerability/Context Specificity (%)')
ax2.set_ylim(0, 100)

plt.savefig(out_dir/'human_eval.svg', bbox_inches='tight')

# %%
