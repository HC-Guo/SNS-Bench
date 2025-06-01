import os
import re
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator



@LOAD_DATASET.register_module()
class NoteQueryCorrExplainDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset
        
        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            dataset.append({
                "sentence1"   : item["sentence1"],
                "sentence2"   : item["sentence2"],
                "answer"      : item["answer"],
            })
        dataset = Dataset.from_list(dataset)
        return dataset



@TEXT_POSTPROCESSORS.register_module()
def note_querycorr_explain_postprocess(text: str):
    text = text.replace("Answer:", "")
    text = text.strip()
    truncated_text = re.split(r'[\n.,]|<|endofresponse|>', text, 1)
    if len(truncated_text) == 0:
        return ''
    return truncated_text[0].strip()
