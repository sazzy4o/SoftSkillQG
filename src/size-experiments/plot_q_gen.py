#%%
import re
import json
import pandas as pd
from pathlib import Path
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = cycler('color', reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']))
# %%
data_dir = Path("./models_data_narval/models")
#%%
rows = []
for score_path in list(data_dir.glob("**/multi_meteor.json")):
    score_data = json.loads(score_path.read_text())
    score_path_string = str(score_path)
    take_match = re.search(r"squad-q-gen-(\w+)-(\d+)", score_path_string)
    take = int(take_match[1]) if take_match[1] != 'None' else 87599
    seed = int(take_match[2])

    epoch_match = re.search(r"epoch_(\d+)", score_path_string)
    epoch = int(epoch_match[1])
    rows.append({
        "take": take,
        "step": take * (epoch+1),
        "epoch": epoch,
        'seed': seed,
        "meteor_mean": score_data["meteor_mean"],
        "meteor_std": score_data["meteor_std"],
    })

df_score = pd.DataFrame(rows)
df_score.sort_values(by="step", inplace=True)
# %%
# Plot F1 grouped by take
# Y-axis: F1
# X-axis: step
# Color: take
# %%
# CONFIG
# plot_var = 'eval_epoch_loss'
plot_var = 'meteor_mean'

# plot_df = df_score
plot_df = df_score.groupby(['take', 'step']).max().reset_index()

# y_label = "Loss"
y_label = "Mulit-METEOR Mean Score"

fig, ax = plt.subplots(figsize=(7, 7))

grouped = plot_df.groupby("take")
sorted_group = sorted(grouped, key=lambda x: x[0], reverse=True)

for name, group in sorted_group:
    group.plot(x="step", y=plot_var, ax=ax)

ax.legend([f'Dataset size: {name}' for name, _ in sorted_group])# %%

plt.ylabel(y_label)
plt.xlabel("Training Steps")
ax.set_xscale('log')

plt.savefig("multi_meteor_question_generation.svg", bbox_inches='tight')
plt.show()

# %%
