import os
import json
import re
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class NoteQueryCorrTopicDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset
        
        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            dataset.append({
                "content"   : item["content"],
                "answer"    : item["answer"],
            })
        dataset = Dataset.from_list(dataset)
        return dataset
    


LEVEL = ['-1', '0', '1', '2', '3']

class NoteQueryCorrTopicEvaluator(BaseEvaluator):
    def score(self, predictions, references) -> dict:
        assert len(predictions) == len(references), (
            "The number of predictions is not equal to the number of references")
        
        counter = {level : {"tp": 0, "fp": 0, "fn": 0} for level in LEVEL}
        pattern = r"(-1|0|1|2|3)"
        error_cnt = 0
        for pred, ref in zip(predictions, references):
            match = re.findall(pattern, pred)
            print(f"match:\n{match}\n")
            if len(match) == 0:
                error_cnt += 1
                continue
            pred = match[0]
            if pred == ref:
                counter[ref]["tp"] += 1
            else:
                counter[ref]["fn"] += 1
                counter[pred]["fp"] += 1
        
        scores = {"success-ratio" : (1 - error_cnt / len(predictions)) * 100}
        for level in LEVEL:
            scores[f"level({level})-precision"] =  counter[level]["tp"] / (counter[level]["tp"] + counter[level]["fp"] + 1e-5) * 100
            scores[f"level({level})-recall"] =  counter[level]["tp"] / (counter[level]["tp"] + counter[level]["fn"] + 1e-5) * 100
            scores[f"level({level})-f1"] =  2 * scores[f"level({level})-precision"] * scores[f"level({level})-recall"] / (scores[f"level({level})-precision"] + scores[f"level({level})-recall"] + 1e-5)
        scores["success-macro-f1"] = np.mean([scores[f"level({level})-f1"] for level in LEVEL]) * (1 - error_cnt / len(predictions))
        return scores