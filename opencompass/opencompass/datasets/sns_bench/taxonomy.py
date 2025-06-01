import os
import re
import json


from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator



@LOAD_DATASET.register_module()
class NoteTaxonomyDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset
        
        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            if "single" in name:
                dataset.append({
                    "content"   : item["content"],
                    "candidates": item["candidates"],
                    "answer"    : item["answer"],
                })
            elif "multi" in name:
                dataset.append({
                    "content"               : item["content"],
                    "candidates_primary"    : item["candidates_primary"],
                    "candidates_secondary"  : item["candidates_secondary"],
                    "candidates_tertiary"   : item["candidates_tertiary"],
                    "answer"                : [item["answer"]],
                })
            else:
                raise ValueError(f"Unknown dataset name: {name}")
        dataset = Dataset.from_list(dataset)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def note_taxonomy_postprocess(text: str):
    text = text.replace("Answer:", "")
    text = text.strip()
    truncated_text = re.split(r'[\n.,]|<|endofresponse|>|<|end|>|<|im_end|>', text, 1)
    if len(truncated_text) == 0:
        return ''
    return truncated_text[0].strip()