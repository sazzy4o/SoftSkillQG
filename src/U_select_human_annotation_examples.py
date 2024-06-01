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
df_dev_for_join = df_all_dev.sort_values('mean_score', ascending=False).groupby(['architecture', 'metric']).head(1)
df_dev_for_join = df_dev_for_join[~df_dev_for_join['architecture'].str.contains('max')]
df_dev_for_join = df_dev_for_join[df_dev_for_join['metric']=='meteor']
join_columns = ['top_p_value','decision_variable','architecture','learning_rate','seed']
df_all = df_all_test.merge(df_dev_for_join[join_columns],how='inner',on=join_columns)
df_all['mean_score'][df_all['metric']!='sacrebleu']*=100
#%%
out_dir = src_dir/'human_annotation_files'
out_dir.mkdir(exist_ok=True)
#%%
df_all.to_csv(out_dir/'best_model_scores.csv', index=False)
# %%
pred_rows = []
for row in df_all.itertuples():
    sub_rows = zip(row.scores,row.predictions,row.contexts)
    for score_list, prediction_list, context_id in sub_rows:
        for score, prediction in sorted(zip(score_list,prediction_list),reverse=True):
            pred_rows.append({
                'architecture': row.architecture,
                'prediction': prediction,
                'context_id': context_id,
                'score': score,
            })
df_pred = pd.DataFrame(pred_rows)

df_pred.to_csv(out_dir/'best_model_predictions.csv', index=False)
#%%
def load_jsonl_df(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f.readlines()]
    return pd.DataFrame(rows)
df_test = load_jsonl_df(src_dir.parent/'data/quail/quail_v1.3/json/challenge.jsonl')

extend_rows = []
for row in df_test.itertuples():
    extend_rows.append({
        'architecture': 'context_id',
        'context_id': row.context_id,
        'prediction': row.context_id,
        'score': 0,
    })
    extend_rows.append({
        'architecture': 'context',
        'context_id': row.context_id,
        'prediction': row.context,
        'score': 0,
    })
    extend_rows.append({
        'architecture': 'reference_question',
        'context_id': row.context_id,
        'prediction': row.question,
        'score': 0,
    })
df_pred_extended = pd.concat([df_pred,pd.DataFrame(extend_rows)])
df_pred_extended.to_csv(out_dir/'best_model_predictions_extended.csv', index=False)
# %%
from itertools import product
# Pivot so the `architecture` names are columns and the `prediction`s are row
groups = [(k,v.reset_index(drop=True)) for k,v in df_pred_extended.groupby(['architecture'])]
for group_a, group_b in product(groups, groups):
    if group_a[0] == group_b[0]:
        continue
    assert group_a[1]['context_id'].equals(group_b[1]['context_id'])

df_all_questions = pd.DataFrame(data={k:v['prediction'].values for k,v in groups}, index=groups[0][1].index)
# Add id to all rows
df_all_questions['row_id'] = df_all_questions.index
df_all_questions.to_csv(out_dir/'best_questions_all.csv', index=False)
# %%
base = (df_all_questions['single_model_t5_lm'].str.strip() == '')
for col in df_all_questions.columns:
    if col == 'row_id':
        continue
    base = base | (df_all_questions[col].str.strip() == '')
df_all_questions_not_empty = df_all_questions[~base]
df_all_questions_not_empty.to_csv(out_dir/'best_questions_all_not_empty.csv', index=False)
# %%
df_selected = df_all_questions_not_empty.sample(
    frac=1, 
    random_state=314
).groupby('context_id').head(3)
assert len(df_selected) == 90 # 30 contexts * 3 questions

df_selected.to_csv(out_dir/'best_questions_selected.csv', index=False)
# %%
# Notes: Sampled such that:
# - each context_id has 3 questions
# - there are no blank questions
# - questions are sampled from the same rank across all architectures
# %% Shuffle the columns so that they are number from q1 to q7
import random
random.seed(314)

columns_to_shuffle = sorted(df_pred['architecture'].unique()) + ['reference_question']
shuffled_rows = []
for row in df_selected.itertuples():
    shuffled_columns = random.sample(columns_to_shuffle, len(columns_to_shuffle))
    assert len(set(shuffled_columns)) == len(shuffled_columns)
    shuffled_rows.append({
        'context': row.context,
        'context_id': row.context_id,
        'row_id': row.row_id,
        **{f'q{i+1}': getattr(row,col) for i,col in enumerate(shuffled_columns)},
        **{f'q{i+1}_id': col for i,col in enumerate(shuffled_columns)},
    })
df_shuffled = pd.DataFrame(shuffled_rows)

consent_string = (out_dir/'consent.txt').read_text()
df_shuffled['consent'] = consent_string

df_shuffled.to_csv(out_dir/'best_questions_ready_for_human_annotations.csv', index=False)
# %%
# Break shuffled questions into 3 files of length 30

for i in range(3):
    df_shuffled.iloc[i*30:(i+1)*30].to_csv(out_dir/f'best_questions_ready_for_human_annotations_part_{i+1}_of_3.csv', index=False)
# %%
