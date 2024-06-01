#%%
import json
import pandas

from pathlib import Path

def jsonl_to_csv(path_jsonl, path_csv):
    with open(path_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pandas.DataFrame(data)
    df.to_csv(path_csv, index=False)

root_dir = Path(__file__).parent.resolve().parent
output_dir = root_dir/'data/quail/quail_v1.3/csv'
output_dir.mkdir(exist_ok=True)

input_dir = root_dir/'data/quail/quail_v1.3/json'

for path_jsonl in input_dir.glob('*.jsonl'):
    path_csv = output_dir/path_jsonl.name.replace('.jsonl', '.csv')
    jsonl_to_csv(path_jsonl, path_csv)
# %%
