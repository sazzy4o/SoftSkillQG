#%%
import json
import pandas as pd
#%%
from pathlib import Path
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
out_folder = Path(root/'src/q_type_classifier_untuned')

num_labels = len(df_train['question_type'].unique())

model = ClassificationModel(
    "roberta", 
    "deepset/roberta-large-squad2", 
    num_labels=num_labels,
    args = {
        "output_dir": str(out_folder.resolve()),
        "best_model_dir": str((out_folder/'best_model').resolve()),        
        # "save_eval_checkpoints": True,
        # "save_steps": -1,
        "use_multiprocessing": False, # Can cause crash if enabled (but, slower)
        "use_multiprocessing_for_evaluation": False, # Can cause crash if enabled (but, slower)
        
        # "num_train_epochs": 2,
        # "train_batch_size": 32,
        # "early_stopping_patience": 1,
        # "use_early_stopping": True,
        # "fp16": False,
        # "learning_rate": 1e-5,
        # # "evaluate_during_training": True,
        # # "evaluate_during_training_verbose": True,

        # "manual_seed": 314,

        # "overwrite_output_dir": True,
    }
)

model.train_model(
    # df_train,
    # eval_df=df_validation,
    pd.concat([df_train, df_validation]),
    # eval_df=df_test,
)
#%%

# result, model_outputs, wrong_predictions = model.eval_model(df_validation)
# # %%
# result
# # %%
# wrong_predictions
# %%
df_eval = df_test
predictions, raw_outputs = model.predict(df_eval['text'].tolist())
df_eval['pred'] = predictions
# %%

print('Accuracy:',metrics.accuracy_score(df_eval['pred'],df_eval['labels']))
# %%
import json

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Func from: https://stackoverflow.com/a/952952/3558475
def flatten(l):
    return [item for sublist in l for item in sublist]

data = load_json('/scratch/vonderoh/EG-clean/src/results_all_metrics/ensemble_sacrebleu_single_model_with_token_probs_0.8_5e-5_dev.json')
# %%
predictions_flat = flatten(data['predictions'])
question_types_flat = flatten(data['question_types'])
# %%
predicted_q_types_num, _ = model.predict(predictions_flat)
# %%
predicted_q_types = [num_map[x] for x in predicted_q_types_num]
# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
# %%
select_indexes = [i for i in range(len(question_types_flat)) if question_types_flat[i]]
mpl.rcParams['figure.dpi'] = 300
cm = metrics.confusion_matrix([question_types_flat[i] for i in select_indexes], [predicted_q_types[i] for i in select_indexes], normalize='all', labels = list(type_map.keys()))
cmd = metrics.ConfusionMatrixDisplay(cm, display_labels=type_map.keys())
fig, ax = plt.subplots(figsize=(16,16))
cmd.plot(ax=ax)
cmd.figure_.savefig(root/'src/confusion_bilal.png')


# %%
series_pred_types = pd.Series(predicted_q_types)
series_req_types = pd.Series(question_types_flat)
# %%
pred_types_df = (series_pred_types.value_counts()/len(series_pred_types)).to_frame()
pred_types_df = pred_types_df.reset_index(level=0)
pred_types_df = pred_types_df.rename(columns={0:'type'})
pred_types_df['num_type'] = pred_types_df['index'].map(type_map)
pred_types_df = pred_types_df.sort_values('num_type')
del pred_types_df['num_type']
pred_types_df = pred_types_df.set_index('index')
# %%
req_types_df = (series_req_types.value_counts()/len(series_req_types)).to_frame()
req_types_df = req_types_df.reset_index(level=0)
req_types_df = req_types_df.rename(columns={0:'type'})
req_types_df['num_type'] = req_types_df['index'].map(type_map)
req_types_df = req_types_df.sort_values('num_type')
del req_types_df['num_type']
req_types_df = req_types_df.set_index('index')
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

def load_json(path):
    with open(path) as f:
        return json.load(f)

# ensemble_sacrebleu_separate_models_probs_0.5_1e-5_dev.json
# ensemble_sacrebleu_single_model_probs_0.8_1e-5_dev.json
# model_path = Path('/scratch/vonderoh/EG-clean/src/models/single_model/1e-5')
model_path = Path('/scratch/vonderoh/EG-clean/src/models/separate_models/1e-5')
# model_path = Path('/scratch/vonderoh/EG-clean/src/models/single_model_with_end_token/5e-5')
# model_path = Path('/scratch/vonderoh/EG-clean/src/models/single_model_with_token/5e-5')
predictions_flat = []
question_types_flat = []
for q_type_dir in model_path.iterdir():
    q_type = q_type_dir.name
    if q_type == 'best_model' or 'checkpoint' in q_type or not q_type_dir.is_dir():
        continue
    pred_data = load_json(q_type_dir/'predictions_nucleus_dev_0.5.json')
    # pred_data = load_json(q_type_dir/'predictions_nucleus_dev_0.8.json')
    for row in pred_data:
        predictions_flat.append(row['prediction'])
        question_types_flat.append(q_type)
#%%
df_pred = pd.read_csv('/home/vonderoh/Github/EnsembleGeneration/hta_wta/output/quail/predictions.csv')
df = pd.merge(df_pred, df_test, left_on='Actual Text', right_on='question')
predictions_flat = df['Generated Text'].tolist()
question_types_flat = df['question_type'].tolist()
# %%
def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
from tqdm import tqdm
model.args.eval_batch_size=4
predicted_q_types_num = []
for pred in tqdm(divide_chunks(predictions_flat,32)):
    predicted_q_types_num_next, _ = model.predict(pred)
    predicted_q_types_num.extend(predicted_q_types_num_next)

# %%
predicted_q_types = [num_map[x] for x in predicted_q_types_num]
# %% I single model without token

# question_types_flat = ['Subsequent_state' for _ in predicted_q_types]

# %%
