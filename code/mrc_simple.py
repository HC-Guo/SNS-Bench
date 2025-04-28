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
class MRCSimpleDataset(BaseDataset):
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
                "query"  : item["query"],
                "content"   : item["content"],
                "answer"    : ''.join(item["answer"]),
            })

        dataset = Dataset.from_list(dataset)
        return dataset



class MRCSimpleEvaluator(BaseEvaluator):
    """ custom evaluator """
    def score(self, predictions, references, **kwargs):
        print("\n\n[Note-MRC-Simple Evaluator]\n\n")
        
        assert len(predictions) == len(references), (
            "predictions and references should have the same length")        
        tp, tn, fp, fn = [], [], [], []
        error_cnt = 0

        prompts = kwargs["origin_prompt"]
        details = []
        # evaluate existence
        for pred, ref, prompt in zip(predictions, references, prompts):
            # FIXME:
            try:
                # FIXME: add reg to extract list
                match = re.compile(r'\[[^\[\]]*\]', re.DOTALL).search(pred)
                if match:
                    pred = match.group(0)
                pred = '\n'.join(eval(pred)).strip()
            except:
                print(f"Eval Error:\nPrediction: {pred}\nReference: {ref}\n")
                error_cnt += 1
                details.append({
                    "prompt"        : prompt,
                    "pred"          : pred,
                    "ref"           : ref,
                    "correct"       : 0,
                })
                continue
            if pred == "" and ref == "":
                tn.append([pred, ref])
                details.append({
                    "prompt"        : prompt,
                    "pred"          : pred,
                    "ref"           : ref,
                    "correct"       : 0,
                })
            elif pred == "" and ref != "":
                fn.append([pred, ref])
                details.append({
                    "prompt"        : prompt,
                    "pred"          : pred,
                    "ref"           : ref,
                    "correct"       : 0,
                })
            elif pred != "" and ref == "":
                fp.append([pred, ref])
                details.append({
                    "prompt"        : prompt,
                    "pred"          : pred,
                    "ref"           : ref,
                    "correct"       : 0,
                })
            else:
                tp.append([pred, ref, prompt])
        
        precision = len(tp) / (len(tp) + len(fp)+ 0.0001)
        recall = len(tp) / (len(tp) + len(fn)+ 0.0001)
        f1 = 2 * precision * recall / (precision + recall + 0.0001)
        # hallucination = len(fp) / len(predictions)
        
        print('***** Existence Evaluation *****')
        print(f'Totally {len(predictions)} pairs')
        print(f'tp/tn/fp/fn: {len(tp)}/{len(tn)}/{len(fp)}/{len(fn)}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')
        # print(f'hallucination ratio: {hallucination}')
        
        # evaluate consistency
        def preprocess_sentence(text):
            text = text.replace('\n', '').replace('�', '')
            text = demoji.replace(text, '')
            # text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。：；～？、~@#￥%……&*（）]+", "", text)
            return text
        
        blue_res, rouge1_res, rouge2_res, rougeL_res = [], [], [], []
        rouge = Rouge()
        for pred, ref, prompt in tp:
            hypothesis = preprocess_sentence(pred)
            reference = preprocess_sentence(ref)
            # print(hypothesis)
            # print(reference)

            blue_score = sentence_bleu([list(jieba.cut(reference))], list(jieba.cut(hypothesis)))
            blue_res.append(blue_score)
            
            try:
                rouge_score = rouge.get_scores(' '.join(jieba.cut(hypothesis)), ' '.join(jieba.cut(reference)))
            except:
                raise Exception("Rouge Error")
            rouge1_res.append(rouge_score[0]['rouge-1']['f'])
            rouge2_res.append(rouge_score[0]['rouge-2']['f'])
            rougeL_res.append(rouge_score[0]['rouge-l']['f'])
            
            details.append({
                "prompt"        : prompt,
                "pred"          : pred,
                "ref"           : ref,
                "correct"       : (blue_score + rouge_score[0]['rouge-1']['f'] + 
                                   rouge_score[0]['rouge-2']['f'] + rouge_score[0]['rouge-l']['f']) / 4
            })
            
            print(f"[TEST]\n" \
                  f"Pred: {hypothesis}\n" \
                  f"Ref:  {reference}\n" \
                  f"Blue: {np.round(blue_score, 2)}\n" \
                  f"Rouge-1: {np.round(rouge_score[0]['rouge-1']['f'], 2)}\n" \
                  f"Rouge-2: {np.round(rouge_score[0]['rouge-2']['f'], 2)}\n" \
                  f"Rouge-L: {np.round(rouge_score[0]['rouge-l']['f'], 2)}")
        
            print('***** Consistency Evaluation *****')
            print(f'Totally {len(blue_res)} tp pairs')
            print(f'BLEU: {np.mean(blue_res)}')
            print(f'ROUGE-1: {np.mean(rouge1_res)}')
            print(f'ROUGE-2: {np.mean(rouge2_res)}')
            print(f'ROUGE-L: {np.mean(rougeL_res)}')
        
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
                
                "details"           : details,
            }

