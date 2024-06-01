#%%
import json
from pathlib import Path

from datasets import load_dataset

dataset = load_dataset('WorkInTheDark/FairytaleQA')
# %%
out_folder = Path('../data/fairytaleqa')
out_folder.mkdir(exist_ok=True)

for split in ['train', 'validation', 'test']:
    with open(out_folder / f'{split}.jsonl', 'w') as out_file:
        data = dataset[split].to_list()
        for row in data:
            json.dump({
                'question': row['question'],
                'answer': row['answer1'],
                'question_type': row['attribute'],
                'context': row['story_section'],
                'context_id': row['story_name'],
            }, out_file)
            out_file.write('\n')
    
# %%
