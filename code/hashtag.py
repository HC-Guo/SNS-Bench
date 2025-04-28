import os
import re
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class HashtagDataset(BaseDataset):
    """ custom dataset """
    @staticmethod
    def load(path: str, name: str, subset: str, version: str = None):
        dataset = []
        if version: 
            name = f"{name}_{version}"
        with open(os.path.join(path, f"{name}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if subset == "single":
                dataset.append({
                    "content"   : item["content"],
                    "candidates": item["candidates"],
                    "answer"    : item["answer"],
                })
            elif subset == "multi":
                dataset.append({
                    "content"       : item["content"],
                    "candidates"    : item["candidates"],
                    "answer"        : item["answer"],
                })
            else:
                raise ValueError(f"Unknown dataset name: {name}")

        dataset = Dataset.from_list(dataset)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def hashtag_postprocess(text: str, **kwargs):
    """ custom postprocess function """
    text = text.replace("答案：", "")
    text = text.replace("Answer:", "")
    subset = kwargs.get("subset", None)
    if subset == "single":
        text = text.strip()
        truncated_text = re.split(r'[\n.,]|<|endofresponse|>|<|end|>|<|im_end|>', text, 1)
        if len(truncated_text) == 0:
            text = ''
        text = truncated_text[0].strip()
    elif subset == "multi":
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


