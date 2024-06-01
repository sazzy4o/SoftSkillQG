#%%
import json
import pandas as pd
from pathlib import Path

rows = []
for folder in Path('./models').iterdir():
    if not folder.is_dir():
        continue
    
    f_name = folder.name
    # if f_name == 'squad-None':
    #     continue

    for sub_folder in folder.iterdir():
        sf_name = sub_folder.name
        score_path = sub_folder/'score.json'
        # print(str(score_path.absolute()))
        with open(score_path) as f:
            try:
                rows.append({**json.load(f),'fname':f_name,'sfname':sf_name})
            except:
                print('Error with', score_path)

df = pd.DataFrame(rows)
df.to_csv('./scores.csv')