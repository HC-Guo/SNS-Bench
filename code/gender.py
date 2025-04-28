import os
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class GenderDataset(BaseDataset):
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


