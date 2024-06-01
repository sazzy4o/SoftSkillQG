#%%
import pandas as pd

from pathlib import Path

root = Path(__file__).parent.parent

df_ghanem = pd.read_csv(root / "data/annotations/Ghanem_Batch_4986034_batch_results.csv")
df_soft_prompt = pd.read_csv(root / "data/annotations/QA Evaluation - single_model_with_token_random_token_init.csv")
df_reference = pd.read_csv(root / "data/annotations/reference_Batch_4990162_batch_results.csv")
df_independent = pd.read_csv(root / "data/annotations/QA Evaluation - separate_models_results.csv")
df_no_control = pd.read_csv(root / "data/annotations/Batch_4992258_batch_single_model.csv")
# %%
dfs_to_eval = {
    'Manually Created': df_reference,
    'T5 WTA': df_ghanem,
    'Independent Models': df_independent,
    'No Control': df_no_control,
    'Soft Prompt': df_soft_prompt
}

rows = []
for name, current_df in dfs_to_eval.items():
    df_grouped = current_df.groupby('HITId').median()
    answerability_scores = list(df_grouped['Answer.answerability1'].values)+list(df_grouped['Answer.answerability2'].values)
    fluency_scores = list(df_grouped['Answer.fluency1'].values)+list(df_grouped['Answer.fluency2'].values)
    grammatically_scores = list(df_grouped['Answer.grammatically1'].values)+list(df_grouped['Answer.grammatically2'].values)
    overall_scores = [sum(x)/3 for x in zip(answerability_scores, fluency_scores, grammatically_scores)]
    rows.append({
        'Method': name,
        'Answerability': sum(answerability_scores)/len(answerability_scores),
        'Fluency': sum(fluency_scores)/len(fluency_scores),
        'Grammaticality': sum(grammatically_scores)/len(grammatically_scores),
        'Overall': sum(overall_scores)/len(overall_scores)
    })
df_out = pd.DataFrame(rows)
# %%
print(df_out.style
    .hide_index()
    .format(
        precision = 2,
        escape="latex",
    )
    .to_latex(
        environment='table*',
        hrules=True,
        column_format='|l|l|l|l|l|',
    )
)
# %%
