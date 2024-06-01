# %%
import json
import pandas as pd

from pathlib import Path
from tqdm import tqdm

results_dir = Path('./results_all_metrics_seed_squad')

def load_json(path):
    with open(path) as f:
        try:
            return json.load(f)
        except:
            print('Error for',path)
            raise


def load_df(prefix='', subset='',suffix='.json'):
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
    df = df.sort_values('mean_perplexity')
    return df
#%%
df_dev = load_df(subset='dev')
# df_test = load_df(subset='challenge')

df_dev = df_dev[~df_dev.architecture.str.contains('hlr')]
#%%
df_dev_for_join = df_dev.sort_values('mean_perplexity', ascending=True).groupby(['architecture','size','seed']).head(1)
join_columns = ['architecture','learning_rate','size','seed']
df_all = df_test.merge(df_dev_for_join[join_columns],how='inner',on=join_columns)
# %%
df_all.head()
# %%
df_all.sort_values('mean_perplexity', ascending=True)
# %%
df_all.sort_values('mean_perplexity', ascending=True)[['mean_perplexity','architecture','seed']]
# %%
df_out = df_all.groupby('architecture').agg(['mean','std'])['mean_perplexity'].reset_index().sort_values('mean', ascending=True)
# %%
list(y for x in df_out.round(4).itertuples() for y in tuple(x[1:]))
# %% Show results for dev set
df_dev_for_join.groupby('architecture').agg(['mean','std'])['mean_perplexity'].reset_index().sort_values('mean', ascending=True)
# %%
