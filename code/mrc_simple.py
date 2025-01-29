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
from rouge_chinese import Rouge


@LOAD_DATASET.register_module()
class MRCSimpleDataset(BaseDataset):
    """Custom dataset for MRC simple tasks."""

    @staticmethod
    def load(path: str, name: str, version: str = None) -> Dataset:
        """Load the MRC simple dataset from a JSON file.

        Args:
            path: The directory path containing the dataset file.
            name: The name of the dataset file (without extension).
            version: Optional version string to append to the filename.

        Returns:
            A Dataset object containing the loaded data.
        """
        dataset = []
        if version:
            name = f"{name}_{version}"
        
        file_path = os.path.join(path, f"{name}.json")
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        for item in data:
            dataset.append({
                "query": item["query"],
                "content": item["content"],
                "answer": ''.join(item["answer"]),
            })

        return Dataset.from_list(dataset)


class MRCSimpleEvaluator(BaseEvaluator):
    """Custom evaluator for MRC simple tasks."""

    def score(self, predictions, references, **kwargs):
        """Evaluate the predictions against the ground truth references.

        Args:
            predictions: List of predicted answers.
            references: List of ground truth answers.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary of evaluation metrics.
        """
        print("\n\n[Note-MRC-Simple Evaluator]\n\n")

        assert len(predictions) == len(references), (
            "Predictions and references should have the same length."
        )
        
        tp, tn, fp, fn = [], [], [], []
        error_count = 0

        prompts = kwargs["origin_prompt"]
        
        # Evaluate existence
        for pred, ref, prompt in zip(predictions, references, prompts):
            try:
                match = re.compile(r'\[[^\[\]]*\]', re.DOTALL).search(pred)
                if match:
                    pred = match.group(0)
                pred = '\n'.join(eval(pred)).strip()
            except Exception:
                print(f"Eval Error:\nPrediction: {pred}\nReference: {ref}\n")
                error_count += 1
                continue

            if pred == "" and ref == "":
                tn.append([pred, ref])
            elif pred == "" and ref != "":
                fn.append([pred, ref])
            elif pred != "" and ref == "":
                fp.append([pred, ref])
            else:
                tp.append([pred, ref, prompt])
        
        precision = len(tp) / (len(tp) + len(fp) + 0.0001)
        recall = len(tp) / (len(tp) + len(fn) + 0.0001)
        f1 = 2 * precision * recall / (precision + recall + 0.0001)

        print('***** Existence Evaluation *****')
        print(f'Total {len(predictions)} pairs')
        print(f'TP/TN/FP/FN: {len(tp)}/{len(tn)}/{len(fp)}/{len(fn)}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')

        # Evaluate consistency
        def preprocess_sentence(text: str) -> str:
            """Preprocess the input sentence by cleaning unwanted characters.

            Args:
                text: The input text to preprocess.

            Returns:
                The cleaned text.
            """
            text = text.replace('\n', '').replace('�', '')
            text = demoji.replace(text, '')
            return text
        
        bleu_results, rouge1_results, rouge2_results, rougeL_results = [], [], [], []
        rouge = Rouge()
        
        for pred, ref, prompt in tp:
            hypothesis = preprocess_sentence(pred)
            reference = preprocess_sentence(ref)

            bleu_score = sentence_bleu([list(jieba.cut(reference))], list(jieba.cut(hypothesis)))
            bleu_results.append(bleu_score)
            
            try:
                rouge_score = rouge.get_scores(' '.join(jieba.cut(hypothesis)), ' '.join(jieba.cut(reference)))
            except Exception:
                raise Exception("Rouge Error")

            rouge1_results.append(rouge_score[0]['rouge-1']['f'])
            rouge2_results.append(rouge_score[0]['rouge-2']['f'])
            rougeL_results.append(rouge_score[0]['rouge-l']['f'])
            
            print(f"[TEST]\n" \
                  f"Pred: {hypothesis}\n" \
                  f"Ref:  {reference}\n" \
                  f"BLEU: {np.round(bleu_score, 2)}\n" \
                  f"ROUGE-1: {np.round(rouge_score[0]['rouge-1']['f'], 2)}\n" \
                  f"ROUGE-2: {np.round(rouge_score[0]['rouge-2']['f'], 2)}\n" \
                  f"ROUGE-L: {np.round(rouge_score[0]['rouge-l']['f'], 2)}")
        
        print('***** Consistency Evaluation *****')
        print(f'Total {len(bleu_results)} TP pairs')
        print(f'BLEU: {np.mean(bleu_results)}')
        print(f'ROUGE-1: {np.mean(rouge1_results)}')
        print(f'ROUGE-2: {np.mean(rouge2_results)}')
        print(f'ROUGE-L: {np.mean(rougeL_results)}')
        
        success_ratio = (1 - error_count / len(predictions)) * 100
        
        return {
            'success-ratio': success_ratio,
            "total-tp": len(tp) / len(predictions) * 100,
            "total-fp": len(fp) / len(predictions) * 100,
            "total-fn": len(fn) / len(predictions) * 100,
            "total-tn": len(tn) / len(predictions) * 100,
            "total-precision": precision * 100,
            "total-recall": recall * 100,
            "total-f1": f1 * 100,
            "bleu": np.mean(bleu_results) * 100 if bleu_results else 0,
            "rouge-1": np.mean(rouge1_results) * 100 if rouge1_results else 0,
            "rouge-2": np.mean(rouge2_results) * 100 if rouge2_results else 0,
            "rouge-L": np.mean(rougeL_results) * 100 if rougeL_results else 0,
            
            "success-f1": success_ratio * f1,
            "success-bleu": success_ratio * np.mean(bleu_results) if bleu_results else 0,
            "success-rouge-1": success_ratio * np.mean(rouge1_results) if rouge1_results else 0,
            "success-rouge-2": success_ratio * np.mean(rouge2_results) if rouge2_results else 0,
            "success-rouge-L": success_ratio * np.mean(rougeL_results) if rougeL_results else 0,
        }