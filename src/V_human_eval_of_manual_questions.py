#%%
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import stats
from tqdm.autonotebook import tqdm

from E_eval_utils import best_set_match_eval, top_reference_eval, cartesian_eval, multi_meteor_eval, neg_fbd_dist_eval, jaccard4_dist_eval


root = Path(__file__).parent.parent

df_human = pd.read_csv(root / "data/annotations/reference_Batch_4990162_batch_results.csv")
# %%
input_dfs = [
    (df_human, 'human_created_questions'),
]
transformered_rows = []
for input_df, arch in input_dfs:
    for index, row in input_df.iterrows():
        transformered_rows.append({
            'hITId': row.HITId,
            'workerId': row.WorkerId,
            'context': row['Input.c1'],
            'question': row['Input.q1'],
            'answerability': row['Answer.answerability1'],
            'fluency': row['Answer.fluency1'],
            'grammaticality': row['Answer.grammatically1'],
            'architecture': arch,
        })
        transformered_rows.append({
            'hITId': row.HITId,
            # 'WorkerId': row.WorkerId,
            'context': row['Input.c2'],
            'question': row['Input.q2'],
            'answerability': row['Answer.answerability2'],
            'fluency': row['Answer.fluency2'],
            'grammaticality': row['Answer.grammatically2'],
            'architecture': arch,
        })
df_transformed = pd.DataFrame(transformered_rows)
df_agg = df_transformed.groupby([
    'architecture',
    'context',
    'question',
    'hITId',
]).mean().reset_index()
df_agg['overall'] = df_agg[['answerability', 'fluency', 'grammaticality']].mean(axis=1)

# %%
# Columns
# context	question	architecture
df_full_set = pd.read_csv(root / "data/quail/quail_v1.3/csv/challenge.csv")
df_full_set['architecture'] = 'human_created_questions'
# %%

method_funcs = {
    'best_set_match': best_set_match_eval,
    'top_reference': top_reference_eval,
    'cartesian': cartesian_eval,
    'multi_meteor': multi_meteor_eval,
    'fbd': neg_fbd_dist_eval,
    'jaccard': jaccard4_dist_eval,
}
def list_duplicates(seq):
  seen = set()
  seen_add = seen.add
  # adds all elements it doesn't know yet to seen and all other to seen_twice
  seen_twice = set( x for x in seq if x in seen or seen_add(x) )
  # turn the set into a list (as requested)
  return list( seen_twice )

scores_rows = []
for arch, arch_df in tqdm(df_full_set.groupby(['architecture'])):
    for context, context_df in arch_df.groupby(['context']):
        ref_group_df = df_full_set[(df_full_set['architecture']=='reference')&(df_full_set['context']==context)]
        df_full_set[(df_full_set['architecture']=='reference')&(df_full_set['context']==context)]
        predictions = context_df['question'].tolist()
        for dup in list_duplicates(predictions):
            assert dup not in df_agg['question'].tolist()
        labels = ref_group_df['question'].tolist()
        context_scores = {}
        prediction_scores = {}
        for method in method_funcs:
            context_agg_score, scores, predictions_sorted = method_funcs[method](predictions, labels)
            context_scores[method] = context_agg_score
            if predictions_sorted is None:
                for pred in predictions:
                    prediction_scores[(pred,method)] = method_funcs[method]([pred], labels)[0]
                continue
            for score, pred in zip(scores, predictions_sorted):
                prediction_scores[(pred,method)] = score
            
        for prediction in predictions:
            scores_rows.append({
                'architecture': arch,
                'context': context,
                'question': prediction,
                **{f'{method}_context_score': context_scores[method] for method in method_funcs},
                **{f'{method}_question_score': prediction_scores[(prediction, method)] for method in method_funcs if (prediction, method) in prediction_scores},
            })
df_scores = pd.DataFrame(scores_rows)
#%%
# ! TODO: Filter by model
df_agg_filtered = df_agg.copy()
# df_agg_filtered = df_agg[df_agg['architecture']=='ghanem_et_al_2022']
# df_agg_filtered = df_agg[df_agg['architecture']=='single_model']
# df_agg_filtered = df_agg[df_agg['architecture']=='separate_models']
# df_agg_filtered = df_agg[df_agg['architecture']=='single_model_with_token_random_token_init']
df_merged = df_agg_filtered.merge(df_scores, on=['architecture', 'context', 'question']).drop_duplicates()
eval_res_dir = root / "src/eval_results"
eval_res_dir.mkdir(exist_ok=True)
df_merged.to_csv(eval_res_dir / "human_eval_raw_scores.csv", index=False)
# %%
agg_len = len(df_agg_filtered)
assert len(df_merged) == agg_len
assert len(df_merged[df_merged['best_set_match_context_score'].notnull()]) == agg_len
# %%
# var = 'answerability'
# var = 'fluency'
# var = 'grammaticality'
# var = 'overall'

def get_corrs(var,method, score_type='question_score'):
    pearson = stats.pearsonr(df_merged.get(f'{method}_{score_type}',np.zeros(len(df_merged[var]))), df_merged[var])[0]
    spearman = stats.spearmanr(df_merged.get(f'{method}_{score_type}',np.zeros(len(df_merged[var]))), df_merged[var])[0]
    return f'{pearson:.4f} / {spearman:.4f}'

output_data_rows = []
for method in method_funcs:
    output_data_rows.append({
        'method': method,
        'fluency': get_corrs('fluency',method),
        'grammaticality': get_corrs('grammaticality',method),
        'answerability': get_corrs('answerability',method),
        # 'overall': get_corrs('overall',method),
    })
df_output_data = pd.DataFrame(output_data_rows)

df_output_data
# %%
# print latex table
print(df_output_data.to_latex(index=False, escape=False))
# %%
df_eval_eval = df_merged.copy()
color_map = {
    'ghanem_et_al_2022': 'red',
    'separate_models': 'blue',
    'single_model': 'green',
    'reference': 'black',
}
df_eval_eval['color'] = df_eval_eval['architecture'].apply(lambda x: color_map[x])
# %%
aspect = 'fluency'
df_eval_eval[df_eval_eval['architecture']!='reference'].plot.scatter(x='matching_question_score',y=aspect,c='color')
# %%
df_eval_eval[df_eval_eval['architecture']!='reference'].plot.scatter(x='top_reference_question_score',y=aspect,c='color')
#%%
df_eval_eval[df_eval_eval['architecture']!='reference'].plot.scatter(x='cartesian_question_score',y=aspect,c='color')
# %%
# Generate Excel file
writer = pd.ExcelWriter(eval_res_dir/'Human Eval Correlations.xlsx',engine='xlsxwriter')   
workbook=writer.book
sheets = [
    'all',
    'single_model',
    'separate_models',
    'ghanem_et_al_2022',
    'single_model_with_token_random_token_init',
]
for sheet in sheets:
    if sheet == 'all':
        df_agg_filtered = df_agg.copy()
    else:
        df_agg_filtered = df_agg[df_agg['architecture']==sheet]
    df_merged = df_agg_filtered.merge(df_scores, on=['architecture', 'context', 'question']).drop_duplicates()
    def get_corrs(var,method, score_type='question_score'):
        pearson = stats.pearsonr(df_merged.get(f'{method}_{score_type}',np.zeros(len(df_merged[var]))), df_merged[var])[0]
        spearman = stats.spearmanr(df_merged.get(f'{method}_{score_type}',np.zeros(len(df_merged[var]))), df_merged[var])[0]
        return float(f'{pearson:.4f}'), float(f'{spearman:.4f}')
    output_data_rows = []
    for method in method_funcs:
        fluency_pearson_question, fluency_spearman_question = get_corrs('fluency',method, score_type='question_score')
        grammaticality_pearson_question, grammaticality_spearman_question = get_corrs('grammaticality',method, score_type='question_score')
        answerability_pearson_question, answerability_spearman_question = get_corrs('answerability',method, score_type='question_score')
        fluency_pearson_context, fluency_spearman_context = get_corrs('fluency',method, score_type='context_score')
        grammaticality_pearson_context, grammaticality_spearman_context = get_corrs('grammaticality',method, score_type='context_score')
        answerability_pearson_context, answerability_spearman_context = get_corrs('answerability',method, score_type='context_score')
        output_data_rows.append({
            'method': method,
            'fluency_pearson_question': fluency_pearson_question,
            'fluency_spearman_question': fluency_spearman_question,
            'grammaticality_pearson_question': grammaticality_pearson_question,
            'grammaticality_spearman_question': grammaticality_spearman_question,
            'answerability_pearson_question': answerability_pearson_question,
            'answerability_spearman_question': answerability_spearman_question,
            'fluency_pearson_context': fluency_pearson_context,
            'fluency_spearman_context': fluency_spearman_context,
            'grammaticality_pearson_context': grammaticality_pearson_context,
            'grammaticality_spearman_context': grammaticality_spearman_context,
            'answerability_pearson_context': answerability_pearson_context,
            'answerability_spearman_context': answerability_spearman_context,
        })
    df_output_data = pd.DataFrame(output_data_rows)
    sheet_name = 'Human Eval - '+sheet.replace('single_model_with_token_random_token_init','soft_prompt')
    worksheet=workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    df_output_data.to_excel(writer, sheet_name=sheet_name, startrow=0 , startcol=0) 
writer.save()
# %%
