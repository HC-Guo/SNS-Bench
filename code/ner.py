import os
import re
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator

@LOAD_DATASET.register_module()
class NERDataset(BaseDataset):
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
                "question"  : item["question"],
                "content"   : item["content"],
                "answer"    : item["answer"],
            })

        dataset = Dataset.from_list(dataset)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def ner_postprocess(text: str, **kwargs):
    """ custom postprocess funtion """
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

