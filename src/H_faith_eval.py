#%%
import json
import pandas as pd
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn import metrics

from simpletransformers.classification import ClassificationModel

#%%
args = sys.argv[1:]
# args = ['single_model_with_token_random_token_init','1e-5','0.4']

architecture = args[0]
learning_rate = args[1]
p_value = args[2]
data_subset = args[3]
size = args[4]
seed = int(args[5])
#%%
type_map = {
    'Causality': 0,
    'Entity_properties': 1,
    'Temporal_order': 2,
    'Belief_states': 3,
    'Factual': 4,
    'Event_duration': 5,
    'Character_identity': 6,
    'Subsequent_state': 7
}
num_map = {v:k for k,v in type_map.items()}

def load_dataset_df(path):
    rows = []
    with open(path) as dataset_file:
        for line in dataset_file.readlines():
            rows.append(json.loads(line))
    # target_text     input_text      prefix
    df = pd.DataFrame(rows)
    df = df[df['question_type']!='Unanswerable']
    df['text'] = df['question']
    df['labels'] = df['question_type'].map(type_map)
    return df

root = (Path(__file__).parent/'../').resolve()

#%%
model = ClassificationModel(
    "roberta", 
    root/"src/q_type_classifier_tuned6", 
    args = {
        "use_multiprocessing": False, # Can cause crash if enabled (but, slower)
        "use_multiprocessing_for_evaluation": False, # Can cause crash if enabled (but, slower)
    }
)

# %%
import json
from pathlib import Path

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_json(path):
    with open(path) as f:
        return json.load(f)

base_path = root/f'src/models-{size}-{seed}-control'
out_folder = root/'src/faithfulness_results_seed'
out_folder.mkdir(exist_ok=True, parents=True)
out_path = root/f'src/faithfulness_results_seed/{architecture}-{learning_rate}-{p_value}-{data_subset}-{size}-{seed}.json'

model_info = {
    'path': f'{architecture}/{learning_rate}',
    'arch': architecture,
    'lr': learning_rate,
    'p': p_value,
}

model_path = base_path/model_info['path']
predictions_flat = []
question_types_flat = []
for q_type_dir in model_path.iterdir():
    q_type = q_type_dir.name
    if q_type in ['best_model','model_files'] or 'checkpoint' in q_type or not q_type_dir.is_dir():
        continue
    pred_data = load_json(q_type_dir/f'predictions_nucleus_{data_subset}_{model_info["p"]}.json')
    for row in pred_data:
        predictions_flat.append(row['prediction'])
        question_types_flat.append(q_type)
model.args.eval_batch_size=4
model.args.silent=True
predicted_q_types_num = []
for pred in tqdm(divide_chunks(predictions_flat,32),total=len(predictions_flat)/32,mininterval=5.0,leave=False):
    predicted_q_types_num_next, _ = model.predict(pred)
    predicted_q_types_num.extend(predicted_q_types_num_next)

predicted_q_types = [num_map[x] for x in predicted_q_types_num]
select_indexes = [i for i in range(len(question_types_flat)) if question_types_flat[i]]
mpl.rcParams['figure.dpi'] = 300
cm = metrics.confusion_matrix([question_types_flat[i] for i in select_indexes], [predicted_q_types[i] for i in select_indexes], normalize='all', labels = list(type_map.keys()))
cmd = metrics.ConfusionMatrixDisplay(cm, display_labels=type_map.keys())
fig, ax = plt.subplots(figsize=(16,16))
ax.set_ylabel('Requested Question Type') 
ax.set_xlabel('Predicted Question Type') 
cmd.plot(ax=ax)
ax.set_ylabel('Requested Question Type') 
ax.set_xlabel('Predicted Question Type') 
# cmd.figure_.savefig(confusion_dir/f'confusion_{best_model["arch"]}.png')
# %%
true_labels = question_types_flat
predicted_labels = predicted_q_types
precision_dict = {

}
recall_dict = {

}
f1_dict = {

}
main_dict = {

}

for classname in type_map:
    tp = sum(1 for x,y in zip(true_labels,predicted_labels) if x == classname and y == classname)
    fn = sum(1 for x,y in zip(true_labels,predicted_labels) if x == classname and y != classname)
    fp = sum(1 for x,y in zip(true_labels,predicted_labels) if x != classname and y == classname)

    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    precision_dict[classname] = f"{precision:.3f}"
    recall_dict[classname] = f"{recall:.3f}"
    f1_dict[classname] = f"{f1:.3f}"
    main_dict[classname] = f"{precision:.3f} / {recall:.3f} / {f1:.3f}"


accuracy = sum(1 for x,y in zip(true_labels,predicted_labels) if x == y)/len(true_labels)

#%%
with open(out_path, "w") as f:
    json.dump({
        'precision': precision_dict,
        'recall': recall_dict,
        'f1': f1_dict,
        'accuracy': accuracy,
        'main': main_dict,
        'size': size,
        'seed': seed,
        **model_info,
    }, f, indent=4)
# %%
