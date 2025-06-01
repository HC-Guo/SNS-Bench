import os
import re
import json
import jieba
import demoji
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


@LOAD_DATASET.register_module()
class NoteMRCSimpleDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset
        
        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            dataset.append({
                "query"     : item["query"],
                "content"   : item["content"],
                "answer"    : ''.join(item["answer"]),
            })
        dataset = Dataset.from_list(dataset)
        return dataset



class NoteMRCSimpleEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        assert len(predictions) == len(references), (
            "predictions and references should have the same length") 
               
        tp, tn, fp, fn = [], [], [], []
        error_cnt = 0
        for pred, ref in zip(predictions, references):
            try:
                # add reg to extract list
                match = re.compile(r'\[[^\[\]]*\]', re.DOTALL).search(pred)
                if match:
                    pred = match.group(0)
                pred = '\n'.join(eval(pred)).strip()
            except:
                print(f"Eval Error:\nPrediction: {pred}\nReference: {ref}\n")
                error_cnt += 1
                continue
            
            if pred == "" and ref == "":
                tn.append([pred, ref])

            elif pred == "" and ref != "":
                fn.append([pred, ref])
                
            elif pred != "" and ref == "":
                fp.append([pred, ref])

            else:
                tp.append([pred, ref])
        
        precision = len(tp) / (len(tp) + len(fp)+ 0.0001)
        recall = len(tp) / (len(tp) + len(fn)+ 0.0001)
        f1 = 2 * precision * recall / (precision + recall + 0.0001)
        
        # evaluate consistency
        def preprocess_sentence(text):
            text = text.replace('\n', '').replace('�', '')
            text = demoji.replace(text, '')
            return text
        
        blue_res, rouge1_res, rouge2_res, rougeL_res = [], [], [], []
        rouge = Rouge()
        for pred, ref in tp:
            hypothesis = preprocess_sentence(pred)
            reference = preprocess_sentence(ref)

            blue_score = sentence_bleu([list(jieba.cut(reference))], list(jieba.cut(hypothesis)))
            blue_res.append(blue_score)
            
            try:
                rouge_score = rouge.get_scores(' '.join(jieba.cut(hypothesis)), ' '.join(jieba.cut(reference)))
            except:
                raise Exception("Rouge Error")
            rouge1_res.append(rouge_score[0]['rouge-1']['f'])
            rouge2_res.append(rouge_score[0]['rouge-2']['f'])
            rougeL_res.append(rouge_score[0]['rouge-l']['f'])
        
        success_ratio = (1 - error_cnt / len(predictions)) * 100
        
        return {
                'success-ratio'     : success_ratio,
                "total-tp"          : len(tp) / len(predictions) * 100,
                "total-fp"          : len(fp) / len(predictions) * 100,
                "total-fn"          : len(fn) / len(predictions) * 100,
                "total-tn"          : len(tn) / len(predictions) * 100,
                "total-precision"   : precision * 100,
                "total-recall"      : recall * 100,
                "total-f1"          : f1 * 100,
                "blue"              : np.mean(blue_res) * 100 if blue_res else 0,
                "rouge-1"           : np.mean(rouge1_res) * 100 if rouge1_res else 0,
                "rouge-2"           : np.mean(rouge2_res) * 100 if rouge2_res else 0,
                "rouge-L"           : np.mean(rougeL_res) * 100 if rougeL_res else 0,
                
                "success-f1"        : success_ratio * f1,
                "success-blue"      : success_ratio * np.mean(blue_res)   if blue_res else 0,
                "success-rouge-1"   : success_ratio * np.mean(rouge1_res) if rouge1_res else 0,
                "success-rouge-2"   : success_ratio * np.mean(rouge2_res) if rouge2_res else 0,
                "success-rouge-L"   : success_ratio * np.mean(rougeL_res) if rougeL_res else 0,
            }
