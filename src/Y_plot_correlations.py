#%%
import pandas as pd
from pathlib import Path
from scipy import stats

df = pd.read_csv('./human_eval_raw_scores.csv')
df = pd.read_csv('./auto_eval_scores.csv') # with errors
# %%
df.head()
df['Neg Number of Errors'] = 5 - df['number_of_errors']
#%%
# fluency	grammaticality	
# df['Fluency Grammar'] = (df['fluency']+df['grammaticality'])/2
#%%
# df_select = df[df['architecture']=='ghanem_et_al_2022'][['answerability','best_set_match_question_score']]
out_dir = Path('./corr_plots')
out_dir.mkdir(exist_ok=True)
# %%
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

arch_config = {
    'ghanem_et_al_2022': {
        'name': 'T5 WTA',
        'colors': [
            'tab:blue',
            'b',
        ],
        'symbol': '.',
    },
    'separate_models': {
        'name': 'IMoE',
        'colors': [
            'r',
            'tab:red',
        ],
        'symbol': 'x',
    },
    'single_model': {
        'name': 'No Control',
        'colors': [
            'tab:green',
            'g',
        ],
        'symbol': '+',
    },
    'single_model_with_token_random_token_init':{
        'name': 'SoftSkillQG',
        'colors': [
            'tab:pink',
            'm',
        ],
        'symbol': '^',
    }
}
eval_methods = [
    'best_set_match',
    'top_reference',
    'cartesian',
    'fbd',
    'jaccard',
]
# human_eval_method = 'answerability'
human_eval_method = 'Neg Number of Errors'

corrs = []
for eval_method in eval_methods:
    for all in [False, True]:
        for arch in sorted(df['architecture'].unique()):
            if arch == 'reference':
                continue
            df_select = df[df['architecture']==arch][[human_eval_method,f'{eval_method}_context_score']]

            arch_data = arch_config[arch]
            arch_colors = arch_data['colors']

            # Sample data
            x = df_select[human_eval_method]
            y = df_select[f'{eval_method}_context_score']

            # Fit with polyfit
            b, m = polyfit(x, y, 1)

            plt.title(f'{arch_data["name"]}')
            plt.ylabel(f"Automatic Evaluation Score ({eval_method})")
            plt.xlabel(f"{human_eval_method.capitalize()} Score")
            plt.plot(x, y, arch_data['symbol'], color=arch_colors[0])
            corr = round(stats.spearmanr(x,y).correlation,4)
            corrs.append(f'{arch_data["name"]}: {corr}')
            plt.plot(x, b + m * x, '-', color=arch_colors[1], label=f"{arch_data['name']} Corr: {corr}")
            plt.xlim([0.95, 5.05])
            if eval_method == 'fbd' or eval_method=='jaccard':
                plt.ylim([min(y)-0.02, max(y)+0.02])
            else:
                plt.ylim([-0.02, 1.02])
            plt.legend()
            if not all:
                plt.savefig(out_dir/f'human_eval_{eval_method}_{arch}.png')
                plt.show()
        if all:
            plt.title('All Architectures')
            x = df[human_eval_method]
            y = df[f'{eval_method}_context_score']
            b, m = polyfit(x, y, 1)
            corr = round(stats.spearmanr(x,y).correlation,4)
            plt.plot(x, b + m * x, '-', color='k', label=f"All Corr: {corr}")
            plt.legend()
            plt.savefig(out_dir/f'human_eval_{eval_method}_all.png')
            plt.show()
# plt.show()
# %%
corrs
# %%
