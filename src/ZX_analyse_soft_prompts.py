#%%
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_json(path):
    with open(path) as f:
        try:
            return json.load(f)
        except:
            print('Error for',path)
            raise

results_dir = Path(__file__).parent.resolve()/'results_all_metrics_seed_control'

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
# %%
dataset = 'challenge'
select_metric = 'meteor'
# select_metric = ''
# dataset = 'challenge'
df_all = load_df('ensemble',dataset=dataset,metric=select_metric)
df_all = df_all[df_all['metric']==select_metric]
#%%
df_all.architecture.unique()
#%%
df_lester = df_all[df_all['architecture']=='soft_attempt']
#%%
df_lester[df_lester['top_p_value']=='0.5'].sort_values('mean_score', ascending=False).groupby(['architecture','metric','seed']).head(1)
#%%
df_res = df_all.sort_values('mean_score', ascending=False).groupby(['architecture','metric','size','seed']).head(1)
df_res.groupby('architecture').mean()
#%%
grouped_df = df_all.groupby(['architecture','metric','size'])
testset_name = "challenge"
for name, group in grouped_df:
    top = group.sort_values('mean_score', ascending=False).iloc[0]
    architecture = top['architecture']
    learning_rate = top['learning_rate']
    size = top['size']
    top_p = top['top_p_value']
    for question_type in ["Belief_states", "Causality", "Character_identity", "Entity_properties", "Event_duration", "Factual", "Subsequent_state", "Temporal_order"]:
        file_path=f"./models-{size}/{architecture}/{learning_rate}/{question_type}/predictions_nucleus_{testset_name}_{top_p}.json"
        if size == '11b':
            print(
                f"if [ ! -f {file_path} ]; then "
                "sbatch --time=2:00:00 --mem-per-cpu=80000M --mincpus=1 --account def-afyshe-ab --output ./logs/slurm-%j.out "
                "--gres=gpu:v100l:2 run_python.sh ./B_predict.py "
                f"{architecture} {learning_rate} {question_type} {top_p} {testset_name} quail {size}"
                "; fi"
            )
        else:
            print(
                f"if [ ! -f {file_path} ]; then "
                "sbatch --time=1:00:00 --mem-per-cpu=24000M --mincpus=1 --account def-afyshe-ab --output ./logs/slurm-%j.out "
                "--gpus=1 run_python.sh ./B_predict.py "
                f"{architecture} {learning_rate} {question_type} {top_p} {testset_name} quail {size}"
                "; fi"
            )
# %%
metrics = [
    # 'sacrebleu',
    'meteor',
    # 'rouge_l',
    # 'gleu',
    # 'bleu_4',
    # 'bleu_3',
    # 'bleu_2',
    # 'bleu_1',
]
for metric in metrics:
    grouped_df = df_all.groupby(['architecture','size'])
    testset_name = "challenge"
    for name, group in grouped_df:
        top = group.sort_values('mean_score', ascending=False).iloc[0]
        architecture = top['architecture']
        size = top['size']
        learning_rate = top['learning_rate']
        deduplication_cutoff = top['deduplication_cutoff']
        top_p = top['top_p_value']
        deduplication_method = top['deduplication_method']
        # metric = top['metric']
        file_path = f"./results_all_metrics/ensemble_{metric}_{architecture}_probs_{top_p}_{deduplication_method}_{deduplication_cutoff}_{learning_rate}_{testset_name}_{size}.json"
        print(
            f"if [ ! -f {file_path} ]; then "
            "sbatch --time=0:20:00 --mem-per-cpu=4000M --mincpus=1 --account def-afyshe-ab --output ./logs/slurm-%j.out "
            f"run_python.sh ./C_ensemble.py {architecture} {learning_rate} {top_p} {metric} {testset_name} quail {size}"
            "; fi"
        )
# ./C_ensemble.py $architecture $learning_rate $top_p $metric whitespace nan $1 quail models results_all_metrics_dups
# %%
grouped_df = df_all.groupby(['architecture','metric','size'])
testset_name = "challenge"
for name, group in grouped_df:
    top = group.sort_values('mean_score', ascending=False).iloc[0]
    architecture = top['architecture']
    learning_rate = top['learning_rate']
    size = top['size']
    top_p = top['top_p_value']
    file_path = f"./faithfulness_results/{architecture}-{learning_rate}-{top_p}-{testset_name}-{size}.json"
    print(
        f"if [ ! -f {file_path} ]; then "
        "sbatch --time=0:20:00 --mem-per-cpu=12000M --mincpus=1 --account def-afyshe-ab --exclude=cdr2614,cdr2486 --gpus=1 --output ./logs/slurm-%j.out "
        f"run_python.sh ./H_faith_eval.py {architecture} {learning_rate} {top_p} {testset_name} {size}"
        "; fi"
    )
# %%
