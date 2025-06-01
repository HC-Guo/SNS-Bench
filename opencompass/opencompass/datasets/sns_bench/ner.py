import os
import re
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class NoteNERDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset
        
        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            dataset.append({
                "question"  : item["question"],
                "content"   : item["content"],
                "answer"    : item["answer"],
            })

        dataset = Dataset.from_list(dataset)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def note_ner_postprocess(text: str, **kwargs):
    text = text.replace("Answer:", "")
    delimiter = kwargs.get("delimiter", ",")
    text = text.strip()
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n]|<|endofresponse|>', text, 1)[0]
    truncated_text = truncated_text.split(delimiter)
    final_text = []
    for t in truncated_text:
        if t:
            final_text.append(t.strip())
    text = final_text
    return text




def round_function(x):
    return round(x, 4)

def do_calculate(F1, P, R):
    f1_score = round_function(F1 * 2 / (P + R + 0.0001))
    precision = round_function(F1 / (P + 0.0001))
    recall = round_function(F1 / (R + 0.0001))
    return f1_score, precision, recall

class NoteNEREvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                         'length'
            }
        F1 = 0
        P = 0  # 准确率
        R = 0  # 召回率
        for prediction, reference in zip(predictions, references):
            results = set(prediction)
            labels = set(reference)
            F1 += len(labels & results)
            P += len(results)
            R += len(labels)
        f1_score, precision, recall = do_calculate(F1, P, R)
        return {'f1': f1_score * 100}
