# Importing libraries
import enum
import os, time, torch, datetime, ast, re, collections
from numpy.core.fromnumeric import sort
from re import I
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import EarlyStopping
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console=Console(record=True)

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(0) # pytorch random seed
np.random.seed(0) # numpy random seed
torch.backends.cudnn.deterministic = True

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

training_logger = Table(Column("Epoch", justify="center" ), Column("train_loss", justify="center"),
                        Column("val_loss", justify="center"), Column("Epoch Time", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)


class YourDataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

  def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.summ_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }


def train(tokenizer, model, device, loader, optimizer):
    """
    Function to be called for training with the parameters passed from main function
    """
    train_losses = []
    model.train()
    for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc='Processing batches..'):
        y = data['target_ids'].to(device, dtype = torch.long)
        lm_labels = y.clone()#[:, 1:].clone().detach()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def validate(tokenizer, model, device, loader):
    """
    Function to be called for validating the trainner with the parameters passed from main function
    """
    validate_losses = []
    model.eval()
    for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc='Validating batches..'):
        y = data['target_ids'].to(device, dtype = torch.long)
        lm_labels = y.clone()#[:, 1:].clone().detach()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels)
        loss = outputs[0]
        validate_losses.append(loss.item())
    return validate_losses

def build_data(dataframes, source_text, target_text, model_params):
    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_tokens('<sep>')
    tokenizer.add_tokens('<ans>')
    # special_tokens_dict = {'additional_special_tokens': ['<sep>', '<ans>']}
    # tokenizer.add_special_tokens(special_tokens_dict)
    # model_params['new_tokens_size'] = len(tokenizer)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataframes[0].sample(frac=1, random_state = 0).reset_index(drop=True)
    val_dataset = dataframes[1].reset_index(drop=True)
    test_dataset = dataframes[2].reset_index(drop=True)
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"VALIDATION Dataset: {val_dataset.shape}")
    console.print(f"TEST Dataset: {test_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    test_set = YourDataSetClass(test_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)


    # Defining the parameters for creation of dataloaders
    train_params = {'batch_size': model_params["TRAIN_BATCH_SIZE"], 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
    test_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, validation_loader, test_loader, tokenizer

def generate(tokenizer, model, device, loader, model_params):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(input_ids = ids, attention_mask = mask, 
                max_length=256, do_sample=True, top_p=0.9, top_k=0, num_return_sequences=1)
                # max_length=256, num_beams=6, length_penalty=1.5, no_repeat_ngram_size=3, early_stopping=True)
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
          if _%10==0:
              console.print(f'Completed {_}')
          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


def T5Trainer(training_loader, validation_loader, tokenizer, model_params):
    """
    T5 trainer
    """
    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)
    # for name, param in model.named_parameters():
    #     if 'decoder' in name:
    #         print(name)
    # exit(1)
    # model.resize_token_embeddings(model_params['new_tokens_size'])
    
    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = model_params["LEARNING_RATE"])
    # optimizer = Adafactor(params = model.parameters(), relative_step=True, lr = model_params["LEARNING_RATE"])

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=model_params["early_stopping_patience"], verbose=False, 
                                    path=f'{model_params["OUTPUT_PATH"]}/best_model_checkpoint.pt')

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        start_time = time.time()
        train_losses = train(tokenizer, model, device, training_loader, optimizer)
        valid_losses = validate(tokenizer, model, device, validation_loader)
        epoch_time = round(time.time()-start_time)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        # preparing the processing time for the epoch and est. the total.
        epoch_time_ = str(datetime.timedelta(seconds=epoch_time))
        total_time_estimated_ = str(datetime.timedelta(seconds=(epoch_time * (model_params["TRAIN_EPOCHS"]-epoch-1))))
        training_logger.add_row(f'{epoch+1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}', f'{valid_loss:.5f}', f'{epoch_time_} (Total est. {total_time_estimated_})')
        console.print(training_logger)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    console.log(f"[Saving Model]...\n")
    #Saving the model after training
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def T5Generator(validation_loader, model_params, save_outputs=True):
    console.log(f"[Loading Model]...\n")
    
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = model.to(device)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = generate(tokenizer, model, device, validation_loader, model_params)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        if save_outputs: final_df.to_csv(os.path.join(model_params["OUTPUT_PATH"],'predictions.csv'))

    if save_outputs:
        console.save_text(os.path.join(model_params["OUTPUT_PATH"],'logs.txt'))
        console.log(f"[Validation Completed.]\n")
        console.print(f"""[Model] Model saved @ {os.path.join(model_params["OUTPUT_PATH"], "model_files")}\n""")
        console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(model_params["OUTPUT_PATH"],'predictions.csv')}\n""")
        console.print(f"""[Logs] Logs saved @ {os.path.join(model_params["OUTPUT_PATH"],'logs.txt')}\n""")
    
    return final_df

#========== utils
def getting_one_story_with_all_skills(test, idx):
    print(test.iloc[idx, :]['context'])
    print(test.iloc[idx, :]['question'])
    print(test.iloc[idx, :]['skillName'])
    
    skills_ = test['skillName'].unique().tolist()
    test = pd.DataFrame({'context': [test.iloc[idx, :]['context'].split(' </s> ')[1] for _ in skills_],
                            'question': [test.iloc[idx, :]['question'] for _ in skills_],
                            'answer': [test.iloc[idx, :]['answer'] if _ == 0 else _ for _ in skills_],
                            'skillName': [_ for _ in skills_]})
    test['context'] = test.apply(lambda row: str(row['skillName']) + ' </s> ' + str(row['context']), axis=1)
    # print(test)
    # exit(1)
    return test

def question_words_counter():
    data = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/demo_experiments/outputs_new/dream_withAnswers_using_squad_cosmos_withAnswers/predictions.csv',
                            names=['id', 'question_', 'question'], skiprows=1)
    QUESTION_TYPES = ["Who", "Where", "When", "Why", "Which", "What", "How", "Other",
                        "Am", "Is", "Was", "Were", "Are", "Does", "Do", 
                        "Did", "Have", "Had", "Has", "Could", "Can", "Shall", 
                        "Should", "Will", "Would", "May", "Might"]
    question_words = []
    for question in data['question'].tolist():
        question_words.append(re.sub(r"['\"“.,!]", "", str(question).title().split()[0]))
    
    question_words_dict = {w: question_words.count(w) for w in QUESTION_TYPES}
    question_words = [q_w for q_w in question_words if q_w not in QUESTION_TYPES]
    question_words_dict['Other'] = len(question_words)
    question_words_dict = sorted(question_words_dict.items(), key=lambda x: x[1], reverse=True)
    question_words_dict = [pair for pair in question_words_dict if pair[1] != 0]
    print(*question_words_dict, sep='\n')
    exit(1)

def switch_skill_name_and_context(df):
    df['context'] = df['context'].map(lambda x: x.split('</s>')[1] + ' </s> ' + x.split('</s>')[0])

def move_skill_name_to_decoder(df):
    df['skillName'] = df['context'].map(lambda x: x.split(' </s> ')[0])
    df['context'] = df['context'].map(lambda x: x.split(' </s> ')[1])
    df['question'] = df.apply(lambda x: x['skillName'] + ' <sep> ' + x['question'], axis=1)
    del df['skillName']

def few_shot_generation(df, size=1):
    if size == 'all':
        return df
    print(f'Few shot generation..: {size}')
    skills_ = test['skillName'].unique().tolist()

    data = []
    for _ in skills_:
        tmp = df[df['skillName'] == _].reset_index(drop=True)
        if isinstance(size, float):
            data.append(tmp.sample(frac=size, random_state=0))
        else:
            data.append(tmp.sample(n=size, random_state=0))
    data = pd.concat(data, axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    return data

if __name__ == '__main__':
    root_dir = Path(__file__).parent.resolve().parent
    training =   pd.read_csv(root_dir/'data/quail/quail_v1.3/csv/train.csv')
    validation = pd.read_csv(root_dir/'data/quail/quail_v1.3/csv/dev.csv')
    test =       pd.read_csv(root_dir/'data/quail/quail_v1.3/csv/challenge.csv')
    
    output_dir = (root_dir/f"hta_wta/output/t5_base").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_params = update_param_automatically({
        "OUTPUT_PATH": str(output_dir), # output path
        "MODEL": "t5-base", # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 8,          # without answers in Decoder 8 | with 4 | large 2
        "VALID_BATCH_SIZE": 8,          # without answers in Decoder 8 | with 4 | large 2
        "TRAIN_EPOCHS": 100,            #
        "VAL_EPOCHS": 1,                #
        "LEARNING_RATE": 1e-4,          # 
        "MAX_SOURCE_TEXT_LENGTH": 512,  #
        "MAX_TARGET_TEXT_LENGTH": 128,  # without answers in Decoder 128 | with 512
        "early_stopping_patience": 1,   #
    })
    training_loader, validation_loader, test_loader, tokenizer = build_data(dataframes=[training, validation, test], source_text="context", target_text="question", model_params=model_params)
    
    T5Trainer(training_loader, validation_loader, tokenizer, model_params=model_params)
    T5Generator(test_loader, model_params=model_params)