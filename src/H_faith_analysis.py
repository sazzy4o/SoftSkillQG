# %%
import json
import pandas as pd

from pathlib import Path
from tqdm import tqdm

results_dir = Path('./faithfulness_results_seed')

def load_json(path):
    with open(path) as f:
        try:
            return json.load(f)
        except:
            print('Error for',path)
            raise


def load_df(prefix='',subset='',suffix='.json'):
    rows = []
    for path in tqdm(list(results_dir.glob(prefix+'*'+suffix))):
        if subset not in path.name:
            continue 
        data = load_json(path)
        # Delete big data array (this array can be useful to statistical tests)
        # data['scores'] = [np.mean(x) for x in data['scores']]
        rows.append(data)
    # assert len(rows) == 9
    df = pd.DataFrame(rows)
    df = df.sort_values('accuracy', ascending=False)
    return df
#%%
df_dev = load_df(subset='-dev-')
df_test = load_df(subset='-challenge-')
# %%
# df.head()
# %%
df_top = df.sort_values('accuracy', ascending=False).groupby(['arch','size']).agg(['mean','std'])['accuracy'].reset_index().sort_values('mean', ascending=False).round(4)
# %%
df_top
# df_top = df_dev.sort_values('accuracy', ascending=False).groupby(['seed','arch','lr']).head(1)
# df_top_test = df_top[['seed','arch','lr']].merge(df_test)
df_test
# %%
# df_top[[
#     'architecture',
#     'learning_rate',
#     'median_perplexity',
#     'mean_perplexity',
#     'size'
# ]]
# %%
df_out = df_test.groupby('arch').agg(['mean','std'])['accuracy'].reset_index().sort_values('mean', ascending=False)
# %%
