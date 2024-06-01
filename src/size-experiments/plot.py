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
data_dir = Path("./models_data_cedar/models")
# %%
rows = []
for f1_path in data_dir.glob("**/f1.json"):
    f1_data = json.loads(f1_path.read_text())
    f1_path_string = str(f1_path)
    take_match = re.search(r"squad-(\w+)", f1_path_string)
    take = int(take_match[1]) if take_match[1] != 'None' else 87599

    epoch_match = re.search(r"epoch_(\d+)", f1_path_string)
    epoch = int(epoch_match[1])
    rows.append({
        "take": take,
        "step": take * (epoch+1),
        "epoch": epoch,
        "f1_mean": f1_data["f1_mean"],
        "f1_std": f1_data["f1_std"],
    })

df_f1 = pd.DataFrame(rows)
df_f1.sort_values(by="step", inplace=True)
#%%
rows = []
for score_path in list(data_dir.glob("**/score.json")):
    score_data = json.loads(score_path.read_text())
    score_path_string = str(score_path)
    take_match = re.search(r"squad-(\w+)", score_path_string)
    take = int(take_match[1]) if take_match[1] != 'None' else 87599

    epoch_match = re.search(r"epoch_(\d+)", score_path_string)
    epoch = int(epoch_match[1])
    rows.append({
        "take": take,
        "step": take * (epoch+1),
        "epoch": epoch,
        "eval_epoch_loss": score_data["eval_epoch_loss"],
        "eval_ppl": score_data["eval_ppl"],
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
plot_var = 'f1_mean'

# plot_df = df_score
plot_df = df_f1

# y_label = "Loss"
y_label = "F1 Score"

fig, ax = plt.subplots(figsize=(7, 7))

grouped = plot_df.groupby("take")
sorted_group = sorted(grouped, key=lambda x: x[0], reverse=True)

for name, group in sorted_group:
    group.plot(x="step", y=plot_var, ax=ax)

ax.legend([f'Dataset size: {name}' for name, _ in sorted_group])# %%

plt.ylabel(y_label)
plt.xlabel("Training Steps")
ax.set_xscale('log')

plt.savefig("f1_question_answering.svg", bbox_inches='tight')
plt.show()

# %%
