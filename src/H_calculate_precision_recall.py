#%%
import json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
from tqdm import tqdm
from sklearn import metrics

from simpletransformers.classification import ClassificationModel

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

df_train = load_dataset_df(root/'data/quail/quail_v1.3/json/train.jsonl')
df_validation = load_dataset_df(root/'data/quail/quail_v1.3/json/dev.jsonl')
df_test = load_dataset_df(root/'data/quail/quail_v1.3/json/challenge.jsonl')

#%%
model = ClassificationModel(
    "roberta", 
    "/home/vonderoh/Github/EnsembleGeneration/src/q_type_classifier_tuned4", 
    args = {
        "use_multiprocessing": False, # Can cause crash if enabled (but, slower)
        "use_multiprocessing_for_evaluation": False, # Can cause crash if enabled (but, slower)
    }
)

# %%
import json
from pathlib import Path
from tqdm import tqdm

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
    from tqdm import tqdm

def load_json(path):
    with open(path) as f:
        return json.load(f)

base_path = Path('/home/vonderoh/Github/EnsembleGeneration/src/models')
confusion_dir = Path('/home/vonderoh/Github/EnsembleGeneration/src/confusion')

out_map = {}

best_models = [
    {
        'path':'single_model_with_token_random_token_init/1e-5',
        'arch':'single_model_with_token_random_token_init',
        'lr':'1e-5',
        'p': '0.4',
    },
    {
        'path':'ghanem_et_al_2022/1e-6',
        'arch':'ghanem_et_al_2022',
        'lr':'1e-6',
        'p': '0.8',
    },
    {
        'path':'separate_models/1e-4',
        'arch':'separate_models',
        'lr':'1e-4',
        'p': '0.4',
    }
]

for best_model in best_models:
    model_path = base_path/best_model['path']
    predictions_flat = []
    question_types_flat = []
    for q_type_dir in model_path.iterdir():
        q_type = q_type_dir.name
        if q_type in ['best_model','model_files'] or 'checkpoint' in q_type or not q_type_dir.is_dir():
            continue
        pred_data = load_json(q_type_dir/f'predictions_nucleus_challenge_{best_model["p"]}.json')
        # pred_data = load_json(q_type_dir/'predictions_nucleus_dev_0.5.json')
        # pred_data = load_json(q_type_dir/'predictions_nucleus_dev_0.8.json')
        for row in pred_data:
            predictions_flat.append(row['prediction'])
            question_types_flat.append(q_type)
    model.args.eval_batch_size=4
    predicted_q_types_num = []
    for pred in tqdm(divide_chunks(predictions_flat,32),total=len(predictions_flat)/32,mininterval=5.0):
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
    cmd.figure_.savefig(confusion_dir/f'confusion_{best_model["arch"]}.png')

    out_map[best_model["arch"]] = {
        'predicted_types': predicted_q_types,
        'requested_types': question_types_flat,
    }
# %%
precision_rows = []
recall_rows = []
f1_rows = []
acc_rows = []
main_rows = []
for method, data in out_map.items():
    true_labels = data['requested_types']
    predicted_labels = data['predicted_types']
    precision_dict = {
        'Method': method,
    }
    recall_dict = {
        'Method': method,
    }
    f1_dict = {
        'Method': method,
    }
    main_dict = {
        'Method': method,
    }

    for classname in type_map:
        tp = sum(1 for x,y in zip(true_labels,predicted_labels) if x == classname and y == classname)
        fn = sum(1 for x,y in zip(true_labels,predicted_labels) if x == classname and y != classname)
        fp = sum(1 for x,y in zip(true_labels,predicted_labels) if x != classname and y == classname)

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        precision_dict[classname] = f"{precision:.3f}"
        recall_dict[classname] = f"{recall:.3f}"
        f1_dict[classname] = f"{f1:.3f}"
        main_dict[classname] = f"{precision:.3f} / {recall:.3f} / {f1:.3f}"
    precision_rows.append(precision_dict)
    recall_rows.append(recall_dict)
    f1_rows.append(f1_dict)
    main_rows.append(main_dict)

    accuracy = sum(1 for x,y in zip(true_labels,predicted_labels) if x == y)/len(true_labels)
    acc_rows.append({
        'Method': method,
        'Accuracy': f"{accuracy:.3f}",
    })
df_precision_out = pd.DataFrame(precision_rows)
df_recall_out = pd.DataFrame(recall_rows)
df_f1_out = pd.DataFrame(f1_rows)
df_acc = pd.DataFrame(acc_rows)
df_main = pd.DataFrame(main_rows)
# %%
df_out = df_main
Path('./tmp.txt').write_text(df_out.T.style
    # .hide_index()
    .format(
        precision = 2,
        escape="latex",
    )
    .to_latex(
        environment='table*',
        hrules=True,
        column_format='|c|c|c|c|c|c|c|c|c|',
    )
)
# %%
