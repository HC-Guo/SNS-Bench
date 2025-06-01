import os
import re
import ast
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class NoteCommentHLWDataset(BaseDataset):
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


class NoteCommentHLWEvaluator(BaseEvaluator):
    def score(self, predictions, references) -> dict:
        assert len(predictions) == len(references), (
            "predictions and references should have the same length") 
        
        F1 = 0
        P = 0   # precision
        R = 0   # recall
        error_cnt = 0
        for prediction, reference in zip(predictions, references):
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
                continue
                
            results = set(prediction)
            labels = set(reference)
            F1 += len(labels & results)
            P += len(results)
            R += len(labels)
        
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
            }