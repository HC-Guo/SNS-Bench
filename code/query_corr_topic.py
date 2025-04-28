import os
import json
import re
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class QueryCorrTopicDataset(BaseDataset):
    """ custom dataset """
    @staticmethod
    def load(path: str, name: str, version: str = None):
        dataset = []
        if version: 
            name = f"{name}_{version}"
        with open(os.path.join(path, f"{name}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            dataset.append({
                "content"   : item["content"],
                "answer"    : item["answer"],
            })

        dataset = Dataset.from_list(dataset)
        return dataset
    

LEVEL = ['-1', '0', '1', '2', '3']

class QueryCorrTopicEvaluator(BaseEvaluator):
    """ custom evaluator """
    def score(self, predictions, references, **kwargs) -> dict:
        assert len(predictions) == len(references), (
            "The number of predictions is not equal to the number of references")
        
        counter = {level : {"tp": 0, "fp": 0, "fn": 0} for level in LEVEL}
        pattern = r"(-1|0|1|2|3)"
        error_cnt = 0
        
        prompts = kwargs["origin_prompt"]
        details = []
        for pred, ref, prompt in zip(predictions, references, prompts):
            match = re.findall(pattern, pred)
            print(f"match:\n{match}\n")
            if len(match) == 0:
                error_cnt += 1
                details.append({
                    "prompt"    : prompt,
                    "pred"      : pred,
                    "ref"       : ref,
                    "correct"   : 0,
                })
                continue
            pred = match[0]
            if pred == ref:
                counter[ref]["tp"] += 1
                details.append({
                    "prompt"    : prompt,
                    "pred"      : pred,
                    "ref"       : ref,
                    "correct"   : 1,
                })
            else:
                counter[ref]["fn"] += 1
                counter[pred]["fp"] += 1
                details.append({
                    "prompt"    : prompt,
                    "pred"      : pred,
                    "ref"       : ref,
                    "correct"   : 0,
                })
        
        scores = {"success-ratio" : (1 - error_cnt / len(predictions)) * 100}
        for level in LEVEL:
            scores[f"level({level})-precision"] =  counter[level]["tp"] / (counter[level]["tp"] + counter[level]["fp"] + 1e-5) * 100
            scores[f"level({level})-recall"] =  counter[level]["tp"] / (counter[level]["tp"] + counter[level]["fn"] + 1e-5) * 100
            scores[f"level({level})-f1"] =  2 * scores[f"level({level})-precision"] * scores[f"level({level})-recall"] / (scores[f"level({level})-precision"] + scores[f"level({level})-recall"] + 1e-5)
            
            
        scores["success-macro-f1"] = np.mean([scores[f"level({level})-f1"] for level in LEVEL]) * (1 - error_cnt / len(predictions))
        scores["details"] = details
        return scores