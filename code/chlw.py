import os
import re
import ast
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class CHLWDataset(BaseDataset):
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


class CHLWEvaluator(BaseEvaluator):
    """ custom evaluator """
    def score(self, predictions, references, **kwargs) -> dict:
        print("\n\n[Note-CHLW Evaluator]\n\n")
        assert len(predictions) == len(references), (
            "predictions and references should have the same length") 
        prompts = kwargs["origin_prompt"]
        F1 = 0
        P = 0   # precision
        R = 0   # recall
        error_cnt = 0
        details = []
        for prediction, reference, prompt in zip(predictions, references, prompts):
            print("[Test]")
            try:
                reference = eval(reference)
                if len(reference) == 0:
                    reference = [""]
                # match list format
                match = re.compile(r'\[[^\[\]]*\]', re.DOTALL).search(prediction)
                if match:
                    prediction = match.group(0)
                prediction = eval(prediction)
                assert isinstance(prediction, list), (
                    "prediction and reference must be list")
                if len(prediction) == 0: 
                    prediction = [""]
                assert all(isinstance(pred, str) for pred in prediction), (
                    "prediction and reference must be list of string")
            except Exception as e:
                print(f"Literal-Eval Error ({e}):\nPrediction: {prediction}\nReference: {reference}\n")
                error_cnt += 1
                details.append({
                    "prompt"        : prompt,
                    "pred"          : prediction,
                    "ref"           : reference,
                    "correct"       : {"tp": 0, "fp": 0, "fn": 0},
                })
                continue

            print(f"Literal-Eval Success:\nprediction: {prediction}\nreference: {reference}\n")
            if prediction == [""] and reference == [""]:
                print(f"prediction and reference are all empty\n\n")
                
            results = set(prediction)
            labels = set(reference)
            F1 += len(labels & results)
            P += len(results)
            R += len(labels)
            details.append({
                "prompt"        : prompt,
                "pred"          : prediction,
                "ref"           : reference,
                "correct"       : {"tp": len(labels & results), 
                                   "fp": len(results) - len(labels & results), 
                                   "fn": len(labels) - len(labels & results)},
            })
        
        f1_score = round(F1 * 2 / (P + R + 0.0001), 4)
        precision = round(F1 / (P + 0.0001), 4)
        recall = round(F1 / (R + 0.0001), 4)
        
        success_ratio = (1 - error_cnt / len(predictions)) * 100

        return {
                'success-ratio'     : success_ratio,
                'f1'                : f1_score * 100,
                'precision'         : precision * 100,
                'recall'            : recall * 100,

                'success-f1'        : success_ratio * f1_score,
                'success-precision' : success_ratio * precision,
                'success-recall'    : success_ratio * recall,
                'details'           : details,
            }