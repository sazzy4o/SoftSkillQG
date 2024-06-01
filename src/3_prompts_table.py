#%% 
import pandas as pd
# %%
quail_path = '../data/quail/quail_v1.3/json/train.jsonl'
quail_df = pd.read_json(quail_path, lines=True)
quail_ghanem_et_al_2022_prompt_map = {
    'Belief_states': 'Belief States',
    'Causality': 'Causality',
    'Character_identity': 'Character Identity',
    'Entity_properties': 'Entity Properties',
    'Event_duration': 'Event Duration',
    'Factual': 'Factual',
    'Subsequent_state': 'Subsequent State',
    'Temporal_order': 'Temporal Order',
}
# %%
sbrcs_path = '../data/dreamscape/train_v4.jsonl'
sbrcs_df = pd.read_json(sbrcs_path, lines=True)
sbrcs_ghanem_et_al_2022_prompt_map = {
    'Basic Story Elements': 'Basic Story Elements',
    'Summarizing': 'Summarizing',
    'Vocabulary': 'Vocabulary',
    'Figurative Language': 'Figurative Language',
    'Inferring': 'Inferring',
    'Close Reading': 'Close Reading',
    'Predicting': 'Predicting',
    'Character Traits': 'Character Traits',
    'Visualizing': 'Visualizing'
}
# %%
output_rows = []
for q_type in sorted(quail_df.question_type.unique()):
    if q_type == 'Unanswerable':
        continue
    output_rows.append({
        'Dataset': 'QuAIL',
        'Prompt': quail_ghanem_et_al_2022_prompt_map[q_type] + ' </s> ',
    })
for q_type in sorted(sbrcs_df.question_type.unique()):
    output_rows.append({
        'Dataset': 'SB-RCS',
        'Prompt': sbrcs_ghanem_et_al_2022_prompt_map[q_type] + ' </s> ',
    })
out_df = pd.DataFrame(output_rows)
out_df
# %%
