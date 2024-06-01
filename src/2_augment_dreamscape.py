#%%
import pandas as pd
import json
from pathlib import Path

#%%
df_train = pd.read_csv('../data/dreamscape/training_dream_question_type.csv')
df_dev = pd.read_csv('../data/dreamscape/validation_dream_question_type.csv')
df_test = pd.read_csv('../data/dreamscape/test_dream_question_type.csv')

df_context = pd.read_csv('../data/dreamscape/passages.csv')
df_context['text'] = df_context['text'].apply(lambda x: x.strip())
# %%
out_folder = Path('../data/dreamscape')
out_folder.mkdir(exist_ok=True)

df_map = {
    'train': df_train,
    'dev': df_dev,
    'test': df_test,
}

for split in ['train', 'dev', 'test']:
    current_df = df_map[split].copy()
    current_df['context'] = current_df['context'].apply(lambda x: x.split('</s>')[-1].strip())

    df_merged = current_df.merge(df_context, left_on='context', right_on='text')

    assert len(df_merged) == len(current_df)

    data = []
    with open(out_folder / f'{split}_v4.jsonl','w') as out_file:
        for _, row in df_merged.iterrows():
            json.dump({
                'question': row['question'],
                'answer': row['answer'],
                'question_type': row['skillName'],
                'context': row['text'],
                'context_id': row['id'],
            }, out_file)
            out_file.write('\n')
    
# %%