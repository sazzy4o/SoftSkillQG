import torch

from scipy.sparse.csgraph import connected_components
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, util

#Load the model
models_cache = {}

def cluster(sequence_tuples,cutoff=0.9,model_type='cos'):
    if model_type in models_cache:
        model = models_cache[model_type]
    else:
        model = SentenceTransformer(f'sentence-transformers/multi-qa-mpnet-base-{model_type}-v1')
        models_cache[model_type] = model
    emb = model.encode([x[0] for x in sequence_tuples])
    if model_type == 'cos':
        scores = util.cos_sim(emb, emb)
    elif model_type == 'dot':
        scores = util.dot_score(emb, emb)
        scores = scores/scores.max() # Normalize scores to [0,1]
    else:
        raise NotImplementedError('Unknown `model_type`')
    merge_elements = (torch.triu(scores).fill_diagonal_(0) > cutoff).cpu().numpy()
    _, groups = connected_components(csgraph=merge_elements, directed=False, return_labels=True)
    group_dict = {x:[] for x in range(max(groups)+1)}
    for index, group in enumerate(groups):
        group_dict[group].append(sequence_tuples[index])
    return list(group_dict.values())

if __name__ == "__main__":
    q_tuples = [
        ("How many people live in London?", 1.02),
        ("How many people live in the city of London?", 0.92),
        ("Where is London?", 2.01),
        ("Why do people live in London?", 1.01),
        ("How many people live in London?", 0.88),
    ]
    print(
        cluster(
            q_tuples,
            cutoff=0.9
        )
    )