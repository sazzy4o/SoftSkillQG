import pickle, torch, re, random, string, collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from bleurt import score
from datasets import load_metric
bleurt_metric = load_metric("bleurt")
# pd.set_option('display.max_colwidth', -1)


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def corpus_f1_em(source, prediction):
    # Lower-case everything
    source = [[snt.lower() for snt in ref] for ref in source]
    prediction = [snt.lower() for snt in prediction]
    
    f1 = []
    em = []
    for i in range(len(source)):
        f1.append(max(compute_f1(gold_references, prediction[i]) for gold_references in source[i]))
        em.append(max(compute_exact(gold_references, prediction[i]) for gold_references in source[i]))
    print('f1: ', np.mean(f1), ', EM: ', np.mean(em))

def corpus_BLEU(source, prediction):
    # Lower-case everything
    source = [[snt.lower() for snt in ref] for ref in source]
    prediction = [snt.lower() for snt in prediction]
    source_splited = [[snt.split() for snt in refs] for refs in source]
    prediction_splited = [snt.split() for snt in prediction]
    
    print('BLEU-1:', corpus_bleu(source_splited, prediction_splited, weights=(1.0, )))
    print('BLEU-2:', corpus_bleu(source_splited, prediction_splited, weights=(0.5, 0.5)))
    print('BLEU-3:', corpus_bleu(source_splited, prediction_splited, weights=(1/3, 1/3, 1/3)))
    print('BLEU-4:', corpus_bleu(source_splited, prediction_splited, weights=(0.25, 0.25, 0.25, 0.25)))

def STS_roberta(source, prediction):
  # Lower-case everything
  source = [[snt.lower() for snt in ref] for ref in source][5:15]
  prediction = [snt.lower() for snt in prediction][5:15]

  model = SentenceTransformer('stsb-roberta-large')
  scores = []
  for i in range(len(prediction)):
    # encode sentences to get their embeddings
    source_embeddings = model.encode(source[i], convert_to_tensor=True)
    predcition_embedding = model.encode([prediction[i]], convert_to_tensor=True)

    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(source_embeddings, predcition_embedding)

    for j in range(len(source_embeddings)):
      print(f'{i}-{j}')
      print("Sentence 1:", prediction[i])
      print("Sentence 2:", source[i][j])
      print("Similarity Score:", cosine_scores[j].item(), '\n')
    print('\n---------------\n')

def NLI_roberta(source, prediction):
  model_name = "roberta-large-mnli"
  tokenizer = RobertaTokenizer.from_pretrained(model_name)
  model = RobertaForSequenceClassification.from_pretrained(model_name)
  model = model.to('cuda')

  scores = []
  for i in tqdm(range(len(prediction)), desc='Calculating NLI using RoBerta'):
    tmp_scores = []
    for j in range(len(source[i])):
      tokens = tokenizer(source[i][j], prediction[i], return_tensors="pt")
      tokens = {key: val.to('cuda', dtype = torch.long) for key, val in tokens.items()}
      logits = model(**tokens).logits
      entain_score = logits[:,[0, 2]].softmax(dim=1).cpu().detach().numpy()[:, 1][0]
      tmp_scores.append(entain_score)
    scores.append(max(tmp_scores))
  print('NLI avg Score: ', np.mean(scores))

def BLEURT_(source, prediction):
  checkpoint = "bleurt/BLEURT-20-D12" # BLEURT-20-D12 bleurt-base-128
  scorer = score.BleurtScorer(checkpoint)

  scores = []
  for i in tqdm(range(len(prediction)), desc='Calculating BLEURT'):
    tmp_scores = []
    for j in range(len(source[i])):
      pair_scores = scorer.score(references=[source[i][j]], candidates=[prediction[i]], batch_size=1024)
      # pair_scores = bleurt_metric.compute(predictions=[prediction[i]], references=[source[i][j]])['scores']
      # print([prediction[i]], '\n', [source[i][j]], '\n', pair_scores)
      # exit(1)
      tmp_scores.append(pair_scores[0])
    scores.append(max(tmp_scores))
  print('BLEURT: ', np.mean(scores))

def BLEURT_per_skill(source, prediction):
  test = pd.read_csv('/home/bghanem/projects/EyeRead_QA/data/tmp_csv/test_dream_question_type.csv')
  assert test.shape[0] == len(prediction)
  checkpoint = "/home/bghanem/projects/EyeRead_QA/demo_experiments/bleurt/bleurt-base-128"
  scorer = score.BleurtScorer(checkpoint)
  tmp_source = source.copy()
  tmp_prediction = prediction.copy()
  results = {}
  skills = test['skillName'].unique().tolist()
  sorted(skills)
  for skillName in skills:
    skillName_indices = np.where(test['skillName'].values == skillName)
    source = tmp_source[skillName_indices]
    # QT = test['Question Type'].values[skillName_indices].tolist()
    # print(skillName, ': ', max(set(QT), key=QT.count))
    # continue
    prediction = tmp_prediction[skillName_indices]
    scores = []
    for i in tqdm(range(len(prediction)), desc='Calculating BLEURT'):
      tmp_scores = []
      for j in range(len(source[i])):
        pair_scores = scorer.score(references=[source[i][j]], candidates=[prediction[i]])
        tmp_scores.append(pair_scores[0])
      scores.append(max(tmp_scores))
    results[skillName] = round(np.mean(scores), 6)
  print('--------------')
  print(*results.items(), sep='\n')

def BLEURT_per_question_type(source, prediction):
  test = pd.read_csv('/home/bghanem/projects/EyeRead_QA/data/tmp_csv/test_dream_question_type.csv')
  # source, prediction = source[:100], prediction[:100]
  # test = test.loc[:99, :]
  checkpoint = "/home/bghanem/projects/EyeRead_QA/demo_experiments/bleurt/BLEURT-20-D12" # BLEURT-20-D12 bleurt-base-128
  scorer = score.BleurtScorer(checkpoint)
  test['score'] = 0
  
  tmp_source = source.copy()
  tmp_prediction = prediction.copy()
  results = {}
  res = []
  question_types = test['Question Type'].unique().tolist()
  sorted(question_types)
  for question_type in question_types:
    question_type_indices = np.where(test['Question Type'].values == question_type)
    source = tmp_source[question_type_indices[0]]
    prediction = tmp_prediction[question_type_indices[0]]
    
    scores = []
    for i in tqdm(range(len(prediction)), desc='Calculating BLEURT'):
      tmp_scores = []
      for j in range(len(source[i])):
        pair_scores = scorer.score(references=[source[i][j]], candidates=[prediction[i]])
        # pair_scores = bleurt_metric.compute(predictions=[prediction[i]], references=[source[i][j]])['scores']
        tmp_scores.append(pair_scores[0])
      scores.append(max(tmp_scores))
    results[question_type] = round(np.mean(scores), 6)
    res.extend(scores)
  print(round(np.mean(res), 6))
  print('--------------')
  print(*results.items(), sep='\n')

if __name__ == '__main__':
  # MODEL_PREDICTIONS = ['dream_withAnswers_using_squad_cosmos_withAnswers', 'dream_withAnswers_using_t5_base', 'squad_cosmos_dream_using_t5_base']
  MODEL_PREDICTIONS = ['dream_withAnswers_using_t5_base_unskilled']
  
  for MODEL_PREDICTION in MODEL_PREDICTIONS:
    print('\n\n\n', MODEL_PREDICTION)
    predictions = pd.read_csv(f'/home/bghanem/projects/EyeRead_QA/demo_experiments/outputs_new/{MODEL_PREDICTION}/predictions.csv', names=['id', 'prediction', 'gold'], skiprows=1)
    del predictions['id']
    
    if not MODEL_PREDICTION.__contains__('dream'):
      predictions = predictions.applymap(lambda txt: [itm.strip() for itm in txt.split('<sep>')])
      predictions = predictions.explode('prediction')
      predictions.fillna('empty question', inplace=True)
      if MODEL_PREDICTION.__contains__('withAnswers'):
        predictions['prediction'] = predictions['prediction'].map(lambda x: x.split('<ans>')[0].strip())
        predictions['gold'] = predictions['gold'].map(lambda x: [itm.split('<ans>')[0].strip() for itm in x])
    else:
      if MODEL_PREDICTION.__contains__('withAnswers'):
        predictions = predictions.applymap(lambda x: x.split('<ans>')[0].strip())
      predictions['gold'] = predictions['gold'].map(lambda x: [x])

    corpus_BLEU(predictions['gold'].tolist(), predictions['prediction'].tolist())
    BLEURT_(predictions['gold'].tolist(), predictions['prediction'].tolist())
    # BLEURT_per_skill(predictions['gold'].values, predictions['prediction'].values)
    # BLEURT_per_question_type(predictions['gold'].values, predictions['prediction'].values)