import os
import re
import json
import numpy as np

from sacrebleu.metrics import CHRF, BLEU
from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator
from .base import BaseDataset


@LOAD_DATASET.register_module()
class RedTransDataset(BaseDataset):
    
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset

        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            if "zh2en" in name:
                dataset.append({
                    "zh"    : item["zh"],
                    "en"    : item["en"],
                    "label" : {
                        "src"               : item["zh"],
                        "ref"               : item["en"],
                        "dst_lang"          : "en",
                    }
                })
            elif "en2zh" in name:
                dataset.append({
                    "en"    : item["en"],
                    "zh"    : item["zh"],
                    "label" : {
                        "src"               : item["en"],
                        "ref"               : item["zh"],
                        "dst_lang"          : "zh",
                    }
                })
            else: 
                raise ValueError(f"Invalid name: {name}")
        dataset = Dataset.from_list(dataset)
        return dataset



"""
    Evaluator
"""
def cal_sacre_blue(reference, hypothesis, lang='zh'):
    bleu = BLEU(tokenize=lang,
                effective_order=True,)   #  stop including n-gram orders for which precision is 0.
    sentence_score = bleu.sentence_score(hypothesis, [reference])
    sentence_score = sentence_score.score / 100
    return sentence_score

def cal_chrf(reference, hypothesis):
    chrf = CHRF(beta=2)
    score = chrf.sentence_score(hypothesis, [reference])
    score = score.score / 100
    return score

    
class RedTransEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        assert len(predictions) == len(references), "predictions and references have different length"
        
        all_bleu_score = []
        all_chrf_score = []
        details = []
        for pred, ref in zip(predictions, references):            
            dst_lang = ref["dst_lang"]
            src_sent = ref["src"].strip()
            ref_sent = ref["ref"].strip()
            pred_sent = pred.strip()

            bleu_score = cal_sacre_blue(ref_sent, pred_sent, 
                                        dst_lang if dst_lang == "zh" else None)
            chrf_score = cal_chrf(ref_sent, pred_sent)
            all_bleu_score.append(bleu_score)
            all_chrf_score.append(chrf_score)
            details.append({
                "dst_lang"      : dst_lang,
                "src_sent"      : src_sent,
                "transd_sent"   : pred_sent,
                "ref_sent"      : ref_sent,
                "bleu_score"    : bleu_score,
                "chrf_score"    : chrf_score,
            })
            
        return {
            "bleu_score"        : np.mean(all_bleu_score) * 100,
            "chrf_score"        : np.mean(all_chrf_score) * 100,
            "details"           : details,
        }
            
