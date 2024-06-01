#%%
import json
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from simpletransformers.classification import ClassificationModel

def load_json(path):
    with open(path) as f:
        try:
            return json.load(f)
        except:
            print('Error for',path)
            raise

root = (Path(__file__).parent/'../').resolve()
results_dir = root/'src/results_all_metrics'
plot_dir = results_dir.parent/'results_distribution_plots'
plot_dir.mkdir(exist_ok=True)

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

line_styles_base = [
    '-',
    '--',
    # '-.',
    # ':',
    # (0, (3, 5, 1, 5, 1, 5)),
    # (0, (3, 7, 1, 5)),
    # (0, (3, 5)),
]
colors_base = [
    'blue','red','green','orange'
]
# %%
dev_set = 'dev'
test_set = 'challenge'
select_metric = 'meteor'
# select_metric = ''
# dataset = 'challenge'
df_all_dev = load_df('ensemble',dataset=dev_set,metric=select_metric)
df_all_dev = df_all_dev[df_all_dev['metric']==select_metric]
df_all_test = load_df('ensemble',dataset=test_set,metric=select_metric)
df_all_test = df_all_dev[df_all_dev['metric']==select_metric]
#%%
best_dev_df = df_all_dev.sort_values('mean_score', ascending=False).groupby(['architecture','metric','size']).head(1)
best_dev_df = best_dev_df[best_dev_df['architecture']!='single_model_soft_prompt_patch']
best_dev_df.head()
# %%
best_test_df = df_all_test.merge(
    best_dev_df[['architecture','size','learning_rate','metric','top_p_value','decision_variable','deduplication_method']], 
    on=['architecture','size','learning_rate','metric','top_p_value','decision_variable','deduplication_method'], 
    suffixes=['_test','_dev'],
)
assert len(best_test_df) == len(best_dev_df)
# %%
model_size = '3b'
df_select = best_test_df[best_test_df['size']==model_size]
# %%
type_map = {
    'Causality': 0,
    'Entity_properties': 1,
    'Temporal_order': 2,
    'Belief_states': 3,
    'Factual': 4,
    'Event_duration': 5,
    'Character_identity': 6,
    'Subsequent_state': 7
}
num_map = {v:k for k,v in type_map.items()}

batch_size = 64

model = ClassificationModel(
    "roberta", 
    root/"src/q_type_classifier_tuned6", 
    args = {
        'eval_batch_size': batch_size,
        'silent': True
    }
)

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

#%%

flatten = lambda l: [item for sublist in l for item in sublist]

df_init = {}
for arch in sorted(df_select['architecture'].unique()):
    predictions_flat = flatten(df_select[df_select['architecture']==arch]['predictions'].iloc[0])

    predicted_q_types_num = []
    for pred in tqdm(divide_chunks(predictions_flat,batch_size),total=len(predictions_flat)/batch_size,mininterval=5.0,leave=False):
        predicted_q_types_num_next, _ = model.predict(pred)
        predicted_q_types_num.extend(predicted_q_types_num_next)

    predicted_q_types = [num_map[x] for x in predicted_q_types_num]

    df_init[arch] = pd.Series(predicted_q_types).value_counts().sort_index()

# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(df_init)

colors = [
#  '#f97306', #'xkcd:orange',
#  '#029386', #'xkcd:teal'
#  '#95d0fc', #'xkcd:light blue'
 '#e50000', #'xkcd:red',
 '#0343df', #'xkcd:blue',
 '#15b01a', #'xkcd:green',
 '#7e1e9c', #'xkcd:purple',
]

df.plot.bar(
    color=colors, 
    # rot=0, /
    title=f"Question Type Distribution ({model_size})",
)
plt.savefig(plot_dir/f'{model_size}.png',bbox_inches='tight')
plt.show()
# %%
