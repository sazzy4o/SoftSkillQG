#%%
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer
from statistics import mean
from typing import List
#%%
# from datasets import load_metric
from sacrebleu.metrics import BLEU
sacrebleu_metric = BLEU(effective_order=True)
# meteor_metric = load_metric("meteor")
# bleu_metric = load_metric("bleu")
# gleu_metric = load_metric("google_bleu")
rouge_l_metric = rouge_scorer.RougeScorer(rouge_types=['rougeL'])
#%%
# Standardized
def sacrebleu(prediction:str,references:List[str]):
    return sacrebleu_metric.sentence_score(
        prediction,references
    ).score
# Seems standardized, but havn't checked
def meteor(prediction:str,references:List[str]):
    return meteor_score(
        references,
        prediction
    )
def rouge_l(prediction:str,references:List[str]):
    return mean([
        rouge_l_metric.score(reference, prediction)['rougeL'].fmeasure
        for reference in references
    ])
# Lacks standardization
# def gleu(prediction:str,references:List[str]):
#     return gleu_metric.compute(
#         predictions=[word_tokenize(prediction)],
#         references=[[word_tokenize(x) for x in references]],
#     )['google_bleu']
def gleu(prediction:str,references:List[str]):
    return sentence_gleu(
        [word_tokenize(x) for x in references], 
        word_tokenize(prediction)
    )
def bleu_4(prediction:str,references:List[str]):
    return sentence_bleu(
        [word_tokenize(x) for x in references], 
        word_tokenize(prediction), 
        weights=(1/4, 1/4, 1/4, 1/4)
    )
def bleu_3(prediction:str,references:List[str]):
    return sentence_bleu(
        [word_tokenize(x) for x in references], 
        word_tokenize(prediction), 
        weights=(1/3, 1/3, 1/3)
    )
def bleu_2(prediction:str,references:List[str]):
    return sentence_bleu(
        [word_tokenize(x) for x in references], 
        word_tokenize(prediction), 
        weights=(1/2, 1/2)
    )
def bleu_1(prediction:str,references:List[str]):
    return sentence_bleu(
        [word_tokenize(x) for x in references], 
        word_tokenize(prediction), 
        weights=(1,)
    )

# %%
metric_map={
    'sacrebleu':sacrebleu,
    'meteor':meteor,
    'rouge_l':rouge_l,
    'gleu':gleu,
    'bleu_4':bleu_4,
    'bleu_3':bleu_3,
    'bleu_2':bleu_2,
    'bleu_1':bleu_1,
    'fbd': None,
}