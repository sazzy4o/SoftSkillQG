#%%
import itertools
import numpy as np
import pandas as pd

from munkres import Munkres
from pathlib import Path
from scipy.stats import stats
from tqdm import tqdm

from lib.metrics import metric_map

#%%
root_dir = Path(__file__).parent.parent.resolve()

def cartesian_eval(predictions_raw, references_raw, metric_name='meteor'):
    metric_func = metric_map[metric_name]
    predictions_product = list(itertools.product(predictions_raw, references_raw))
    predictions = [x[0] for x in predictions_product]
    references = [x[1] for x in predictions_product]
    local_scores = []
    for prediction, reference in zip(predictions, references):
        local_scores.append(metric_func(
            prediction, 
            [reference]
        ))

    return np.mean(local_scores)
# %%
predictions_folder = root_dir / 'src/arch_predictions_full_0_4'
# Iterate through all files and merge into one csv
all_predictions = []
df = None
for file in predictions_folder.glob('*.csv'):
    if df is None:
        df = pd.read_csv(file)
        df['source'] = file.name.replace('.csv','')
    else:
        df_new = pd.read_csv(file)
        df_new['source'] = file.name.replace('.csv','')
        df = pd.concat([df,df_new],ignore_index=True)
# del df['context']
# %%
df.head()
# %%
eval_eval_rows = []
for context, df_group in tqdm(df.groupby(['context','source'])):
    predictions = df_group['question'].tolist()
    predictions_product = list(itertools.product(predictions, predictions))
    eval_eval_rows.append({
        'context': context[0],
        'self_similarity': cartesian_eval(predictions, predictions),
        'model': context[1],
    })
df_eval_eval = pd.DataFrame(eval_eval_rows)
# %%
df_eval_eval.groupby('model').mean()
# %%
output_data_rows = []
for method in ['matching_eval', 'top_reference_eval', 'single_eval']:
    output_data_rows.append({
        'method': method,
        'pearson': stats.pearsonr(df_eval_eval[method], df_eval_eval['self_similarity'])[0],
        'spearman': stats.spearmanr(df_eval_eval[method], df_eval_eval['self_similarity'])[0],
    })
df_output_data = pd.DataFrame(output_data_rows)
# %%
df_output_data
# %%
