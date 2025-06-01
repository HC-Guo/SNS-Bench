import os
import re
import json
import torch
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from sentence_transformers import SentenceTransformer, util


@LOAD_DATASET.register_module()
class NoteQueryGenDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset
        
        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            dataset.append({
                "content": item["content"],
                "answer": item["answer"],
            })
        return Dataset.from_list(dataset)



def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    # 2d dp matrix to store the edit distance
    # f[i][j]: edit dist between word1_[:i) and word2_[:j) , final prediction is f[m][n]
    f = [[0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                f[i][j] = j
            elif j == 0:
                f[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
    return f[m][n]

def calculate_anls_score(prediction, reference):
    max_len = max(len(prediction), len(reference))
    nld = edit_distance(prediction, reference) / max_len
    anls = 1.0 - nld
    return anls * 100


class NoteQueryGenEvaluator(BaseEvaluator):
    def __init__(self, embed_model_path):
        super().__init__()
        self.embed_model_path = embed_model_path
    def score(self, predictions, references):
        assert len(predictions) == len(references), (
            "Predictions and references should have the same length."
        )
        
        similarity_scores = []
        scores = []        
        # calculate embedding similarity
        embed_model = SentenceTransformer(
            self.embed_model_path,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        for pred, ref in zip(predictions, references):
            truncated_pred = re.split(r'[\n]|<|endofresponse|>', pred.strip(), 1)[0]
            embed_pred = embed_model.encode(truncated_pred, convert_to_tensor=True)
            embed_ref = embed_model.encode(ref, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embed_pred, embed_ref).item() * 100
            anls = calculate_anls_score(pred, ref)
            scores.append((similarity + anls) / 2)
            
        return { "final_scores": np.mean(scores) }
