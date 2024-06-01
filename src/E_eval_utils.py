#%%
import copy
import itertools
import numpy as np

from munkres import Munkres

from eval_metrics.bert_distances import FBD
from eval_metrics.multiset_distances import MultisetDistances
from nlgeval import NLGEval

from lib.metrics import metric_map

# New eval
def best_set_match_eval(predictions, references, metric_name='meteor'):
    # assert len(predictions) == len(references)
    num_preds = len(predictions)
    metric_func = metric_map[metric_name]
    matrix = [
        [-metric_func(
            predictions[i], 
            [references[j]]
        )
        for i in range(num_preds)] for j in range(num_preds)
    ]
    m = Munkres()
    indexes = m.compute(matrix)

    final_references = [references[i] for i,j in indexes]
    final_predictions = [predictions[j] for i,j in indexes]

    local_scores = []
    for label,prediction in zip(final_references, final_predictions):
        local_scores.append(metric_func(
            prediction, 
            [label]
        ))
    return np.mean(local_scores), local_scores, final_predictions

# Same as Bilal's eval (he would just generated 1 for each type)
# Some other papers use this (need to double check which)
def top_reference_eval(predictions, references, metric_name='meteor'):
    metric_func = metric_map[metric_name]
    local_scores = []
    for prediction in predictions:
        local_scores.append(metric_func(
            prediction, 
            references
        ))

    return np.mean(local_scores), local_scores, predictions


def cartesian_eval(predictions_raw, references_raw, metric_name='meteor'):
    metric_func = metric_map[metric_name]
    predictions_product = list(itertools.product(predictions_raw, references_raw))
    predictions = [x[0] for x in predictions_product]
    references = [x[1] for x in predictions_product]
    local_scores = []
    for prediction, reference in zip(predictions, references):
        local_scores.append(metric_func(
            prediction, 
            [reference]
        ))

    return np.mean(local_scores), local_scores, predictions


def neg_fbd_dist_eval(predictions_raw, references_raw):
    fbd = FBD(references=references_raw, model_name="bert-base-uncased", bert_model_dir=None)
    return -fbd.get_score(sentences=predictions_raw), None, None


def jaccard4_dist_eval(predictions_raw, references_raw):
    msd = MultisetDistances(references=references_raw, min_n=4, max_n=4)
    return msd.get_jaccard_score(sentences=predictions_raw)[4], None, None

metrics_to_omit = [x for x in [*NLGEval.valid_metrics,*NLGEval.glove_metrics] if x!='METEOR']
m = Munkres()
nlgeval = NLGEval(metrics_to_omit=metrics_to_omit)
def multi_meteor_eval(predictions_raw, references_raw):
    pred_num = len(predictions_raw)
    ref_num = len(references_raw)
    profit_matrix = {}
    for pred in predictions_raw:
        row = {}
        for ref in references_raw:
            metrics = nlgeval.compute_individual_metrics([ref], pred)
            for key in metrics:
                if key in row:
                    row[key].append(metrics[key])
                else:
                    row[key] = [metrics[key]]
        for key in row:
            if key in profit_matrix:
                profit_matrix[key].append(row[key])
            else:
                profit_matrix[key] = [row[key]]
    ranked_idx = [id for id in range(len(predictions_raw))]
    optimal_indexes = {}
    for key in profit_matrix:
        profit_matrix[key] = [profit_matrix[key][row] for row in ranked_idx]
        optimal_indexes[key] = compute_optimal_assignment(profit_matrix[key])
    individual_metrics = {}
    for key in optimal_indexes:
        vals = []
        for row, column in optimal_indexes[key]:
            vals.append(profit_matrix[key][row][column])

        individual_metrics["multi_" + key + "_prec"] = sum(vals) / pred_num if pred_num != 0 else 0
        individual_metrics["multi_" + key + "_rec"] = sum(vals) / ref_num if ref_num != 0 else 0
        p = individual_metrics["multi_" + key + "_prec"]
        r = individual_metrics["multi_" + key + "_rec"]
        individual_metrics["multi_" + key + "_F1"] = (2 * p * r) / (p + r) if (p + r) != 0 else 0
    local_scores = []
    local_predictions = []
    for row_id, column_id in optimal_indexes['METEOR']:
        local_scores.append(profit_matrix[key][row][column_id])
        local_predictions.append(predictions_raw[row_id])

    return individual_metrics["multi_METEOR_F1"], local_scores, local_predictions

def compute_optimal_assignment(profit_matrix):
    profit_matrix_ = copy.deepcopy(profit_matrix)
    np_profit_matrix = np.asarray(profit_matrix_)
    # converting profit matrix to meet the pre-conditions of the hungarian algorithm code
    np_positive_profit_matrix = np_profit_matrix - max(0, np.min(np_profit_matrix))
    np_rounded_profit_matrix = np.around(np_positive_profit_matrix * 100000).astype(int)
    np_cost_matrix = (np.max(np_rounded_profit_matrix) + 1) - np_rounded_profit_matrix
    input_cost_matrix = np_cost_matrix.tolist()
    # print("modified cost matrix: ", input_cost_matrix)
    optimal_indexes = m.compute(input_cost_matrix)
    return optimal_indexes
# %%