# Importing libraries
import os, time, torch, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

from .utils import EarlyStopping
from lib.t5 import T5ForConditionalGenerationWithConfidence

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(0) # pytorch random seed
np.random.seed(0) # numpy random seed
torch.backends.cudnn.deterministic = True

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

prompt_map = {
    'Belief_states': 'Belief States',
    'Causality': 'Causality',
    'Character_identity': 'Character Identity',
    'Entity_properties': 'Entity Properties',
    'Event_duration': 'Event Duration',
    'Factual': 'Factual',
    'Subsequent_state': 'Subsequent State',
    'Temporal_order': 'Temporal Order',
}

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
    # target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        # 'target_ids_y': target_ids.to(dtype=torch.long)
    }


def train(tokenizer, model, device, loader, optimizer, gradient_accumulation_steps = 1):
    """
    Function to be called for training with the parameters passed from main function
    """
    train_losses = []
    model.train()

    for step, data in tqdm(enumerate(loader, 0), total=len(loader), desc='Processing batches..'):
        y = data['target_ids'].to(device, dtype = torch.long)
        lm_labels = y.clone()#[:, 1:].clone().detach()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        kwargs = {}
        if 'task' in data:
            kwargs['task'] = data['task']
            kwargs['task_ids'] = data['task_ids'].to(device)

        outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels, **kwargs)
        loss = outputs[0]
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_losses.append(loss.item())
    return train_losses

def validate(tokenizer, model, device, loader):
    """
    Function to be called for validating the trainer with the parameters passed from main function
    """
    validate_losses = []
    model.eval()
    with torch.inference_mode():
        for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc='Validating batches..'):
            lm_labels = data['target_ids'].clone().to(device, dtype = torch.long)
            lm_labels[lm_labels == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels)
            loss = outputs[0]
            validate_losses.append(loss.item())
    return validate_losses

def build_data(tokenizer, dataframes, source_text, target_text, model_params, dataset_class=YourDataSetClass):
    # make sure we are not affecting the original
    dataframes = [dataframe.copy() for dataframe in dataframes]

    # logging
    print(f"[Data]: Reading data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataframes[0].sample(frac=1, random_state = 0).reset_index(drop=True)
    val_dataset = dataframes[1].reset_index(drop=True)
    test_dataset = dataframes[2].reset_index(drop=True)
    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"VALIDATION Dataset: {val_dataset.shape}")
    print(f"TEST Dataset: {test_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = dataset_class(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = dataset_class(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    test_set = dataset_class(test_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)


    # Defining the parameters for creation of dataloaders
    train_params = {'batch_size': model_params["TRAIN_BATCH_SIZE"], 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 0}
    test_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 0}

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, validation_loader, test_loader

def generate(tokenizer, model, device, loader, model_params, total=0):

    """
    Function to evaluate model for predictions

    """
    model.eval()
    probs = []
    predictions = []
    actuals = []
    with torch.inference_mode():
        for _, data in tqdm(enumerate(loader, 0),total=total):
            y = data['target_ids']
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            kwargs = {}
            if 'task' in data:
                kwargs['task'] = data['task']
                kwargs['task_ids'] = data['task_ids']

            full_outputs = model.generate(
                input_ids = ids, 
                attention_mask = mask, 
                max_length=256, 
                do_sample=True, 
                top_p=model_params['TOP_P'], 
                top_k=0, 
                num_return_sequences=1,
                return_dict_in_generate=True,
                **kwargs,
            )
            generated_ids=full_outputs.sequences
                # conf_out=full_outputs.confidence
            prob_out=full_outputs.probs
            local_probs = []
            for prob, toks in zip(prob_out, generated_ids):
                local_probs.append(prob[toks!=0].cpu().numpy())
                    # max_length=256, num_beams=6, length_penalty=1.5, no_repeat_ngram_size=3, early_stopping=True)
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            #   if _%10==0:
            #       print(f'Completed {_}')
            predictions.extend(preds)
            actuals.extend(target)
            probs.extend(local_probs)
    return predictions, actuals, probs


def T5Trainer(model, training_loader, validation_loader, tokenizer, model_params):
    """
    T5 trainer
    """
    # logging
    print(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    gradient_accumulation_steps = model_params['GRADIENT_ACCUMULATION_BATCH_SIZE']
    
    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = model_params["LEARNING_RATE"])
    # optimizer = Adafactor(params = model.parameters(), relative_step=True, lr = model_params["LEARNING_RATE"])

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=model_params["early_stopping_patience"], verbose=False, 
                                    path=f'{model_params["OUTPUT_PATH"]}/best_model_checkpoint.pt')

    best_valid_loss = float('inf')

    # Training loop
    print(f'[Initiating Fine Tuning]...\n')
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        start_time = time.time()
        train_losses = train(tokenizer, model, device, training_loader, optimizer, gradient_accumulation_steps=gradient_accumulation_steps)
        valid_losses = validate(tokenizer, model, device, validation_loader)
        epoch_time = round(time.time()-start_time)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        # preparing the processing time for the epoch and est. the total.
        epoch_time_ = str(datetime.timedelta(seconds=epoch_time))
        total_time_estimated_ = str(datetime.timedelta(seconds=(epoch_time * (model_params["TRAIN_EPOCHS"]-epoch-1))))
        print(f'{epoch+1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}', f'{valid_loss:.5f}', f'{epoch_time_} (Total est. {total_time_estimated_})')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{model_params["OUTPUT_PATH"]}/best_model_checkpoint.pt')
            print(f'[Model] Model saved @ {model_params["OUTPUT_PATH"]}/best_model_checkpoint.pt\n')

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f'{model_params["OUTPUT_PATH"]}/best_model_checkpoint.pt'))

    print(f"[Saving Model]...\n")
    #Saving the model after training
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def T5Generator(validation_loader, model_params, save_outputs=True):
    print(f"[Loading Model]...\n")
    
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model = T5ForConditionalGenerationWithConfidence.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = model.to(device)

    # evaluating test dataset
    print(f"[Initiating Validation]...\n")
    # for epoch in range(model_params["VAL_EPOCHS"]):
    predictions, actuals, probs = generate(tokenizer, model, device, validation_loader, model_params)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals, 'Token Probs': probs})
    if save_outputs: 
        final_df.to_csv(os.path.join(model_params["OUTPUT_PATH"],'predictions.csv'))
        # console.save_text(os.path.join(model_params["OUTPUT_PATH"],'logs.txt'))
        print(f"[Validation Completed.]\n")
        print(f"""[Model] Model saved @ {os.path.join(model_params["OUTPUT_PATH"], "model_files")}\n""")
        print(f"""[Validation] Generation on Validation data saved @ {os.path.join(model_params["OUTPUT_PATH"],'predictions.csv')}\n""")
        print(f"""[Logs] Logs saved @ {os.path.join(model_params["OUTPUT_PATH"],'logs.txt')}\n""")
    
    return final_df