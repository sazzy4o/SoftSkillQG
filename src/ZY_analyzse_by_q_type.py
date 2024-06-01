#%%
import pandas as pd
import json
import numpy as np

from pathlib import Path
from tqdm import tqdm
# %%
rows = []
for path in tqdm(list(Path('results_match_counts_seed_control').glob('ensemble*'))):
    data = json.load(open(path))
    rows.extend(data)
df = pd.DataFrame(rows)
# %%
df_dev = df[df['dataset']=='dev']
df_test = df[df['dataset']=='challenge']

df_best = df_dev.sort_values('mean_score', ascending=False).groupby(['architecture','question_type','seed']).head(1)
# %%
plot_df = df_best.groupby(['architecture','question_type']).mean().reset_index().sort_values(['question_type','architecture'])

# %%
# Plot bar chart grouped by architecture
import matplotlib.pyplot as plt
ax = plot_df.pivot(index='question_type', columns='architecture', values='mean_score').plot.bar()
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_title('Mean score by question type')
ax.set_ylabel('Mean score')
ax.set_xlabel('Question type')
# Plot legend
import matplotlib.pyplot as plt
plt.legend(loc='center left', bbox_to_anchor=(1.1, 1.05))

# %%
grouped_df = df_best.groupby(['architecture','metric','size','seed','question_type'])
testset_name = "challenge"
for name, group in grouped_df:
    top = group.sort_values('mean_score', ascending=False).iloc[0]
    architecture = top['architecture']
    learning_rate = top['learning_rate']
    size = top['size']
    seed = top['seed']
    top_p = top['top_p_value']
    for question_type in ["Belief_states", "Causality", "Character_identity", "Entity_properties", "Event_duration", "Factual", "Subsequent_state", "Temporal_order"]:
        file_path=f"./models-{size}-{seed}-control/{architecture}/{learning_rate}/{question_type}/predictions_nucleus_{testset_name}_{top_p}.json"
        print(
            f"if [ ! -f {file_path} ]; then "
            "sbatch --time=20:00 --mem-per-cpu=10000M --mincpus=1 --account def-afyshe-ab --output ./logs/slurm-%j.out "
            "--gpus=1 run_python.sh ./B_predict.py "
            f"{architecture} {learning_rate} {question_type} {top_p} {testset_name} quail {size} {seed}"
            "; fi"
        )
# %%
metrics = [
    'meteor',
]
for metric in metrics:
    grouped_df = df_best.groupby(['architecture','metric','size','seed','question_type'])
    testset_name = "challenge"
    for name, group in grouped_df:
        top = group.sort_values('mean_score', ascending=False).iloc[0]
        architecture = top['architecture']
        size = top['size']
        seed = top['seed']
        learning_rate = top['learning_rate']
        deduplication_cutoff = top['deduplication_cutoff']
        top_p = top['top_p_value']
        deduplication_method = top['deduplication_method']
        # metric = top['metric']
        file_path = f"./results_match_counts_seed_control/ensemble_{metric}_{architecture}_probs_{top_p}_{deduplication_method}_{deduplication_cutoff}_{learning_rate}_{testset_name}_{size}_{seed}.json"
        print(
            f"if [ ! -f {file_path} ]; then "
            "sbatch --time=0:20:00 --mem-per-cpu=4000M --mincpus=1 --account def-afyshe-ab --output ./logs/slurm-%j.out "
            f"run_python.sh ./C_ensemble_match_count.py {architecture} {learning_rate} {top_p} {metric} {testset_name} quail {size} {seed}"
            "; fi"
        )
# %%
# Merge df_best with df_test

df_test_best = df_test.merge(df_best, on=['architecture','question_type','seed','learning_rate','top_p_value'], suffixes=['_test','_best'])
assert len(df_test_best) == len(df_best)
# %%
plot_df = df_test_best.groupby(['architecture','question_type']).mean().reset_index().sort_values(['question_type','architecture'])
arch_map = {
    'control_t5_lm': 'SoftSkillQG',
    'ghanem_et_al_2022_t5_lm': 'T5 WTA',
}
plot_df['architecture'] = plot_df['architecture'].apply(lambda x: arch_map[x])
# %%
# Plot bar chart grouped by architecture
import matplotlib.pyplot as plt
ax = plot_df.pivot(index='question_type', columns='architecture', values='mean_score_test').plot.bar()
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
# ax.set_title('Mean Multi-METEOR Score by Question Type')
ax.set_ylabel('Mean Multi-METEOR score')
ax.set_xlabel('Question Type')
# Plot legend
plt.xticks(rotation=90)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45))
plt.savefig('mean_score_by_q_type.svg', bbox_inches='tight')
plt.show()

# %%
