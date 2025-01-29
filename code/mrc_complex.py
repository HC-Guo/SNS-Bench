import os
import re
import json
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


OPTION_F1_THRESHOLD = 0.8
REASON_F1_THRESHOLD = 0.5


@LOAD_DATASET.register_module()
class MRCComplexDataset(BaseDataset):
    """Custom dataset for MRC complex tasks."""

    @staticmethod
    def load(path: str, name: str, version: str = None) -> Dataset:
        """Load the MRC complex dataset from a JSON file.

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
                "content": item["content"],
                "answer": item["answer"],
            })

        return Dataset.from_list(dataset)


class MRCComplexEvaluator(BaseEvaluator):
    """Custom evaluator for MRC complex tasks."""

    def score(self, predictions, references, **kwargs):
        """Evaluate the predictions against the ground truth references.

        Args:
            predictions: List of predicted answers.
            references: List of ground truth answers.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary of evaluation metrics.
        """
        print("\n\n[Note-MRC-Complex Evaluator]\n\n")
        
        assert len(predictions) == len(references), (
            "Predictions and references should have the same length."
        )

        num_samples = 1000
        infos = []
        tp = tn = fp = fn = 0
        postprocess_error_count = 0

        # N-gram level
        option_f1_list = []
        option_em_list = []
        reason_f1_list = []
        reason_em_list = []

        # Answer level
        option_al_precision_list = []
        option_al_recall_list = []
        reason_al_precision_list = []
        reason_al_recall_list = []

        prompts = kwargs["origin_prompt"]

        for pred, gt, prompt in zip(predictions, references, prompts):
            info = {}
            print(f"\n[Test]\nGT: {gt}\nPred: {pred}\n")
            pred, pred_info = v4_result_postprocess(pred)
            gt, gt_info = v4_result_postprocess(gt)
            print(f"[Postprocess]\nGT_Info: {gt_info}\nGT: {gt}\nPred_Info: {pred_info}\nPred: {pred}")

            if gt_info == 'postprocess error' or pred_info == 'postprocess error':
                postprocess_error_count += 1
                continue

            if not gt and not pred:
                info['status'] = 'tn'
                tn += 1
                print(f"[Status] -> TN")
            elif not gt and pred:
                info['status'] = 'fp'
                fp += 1
                print(f"[Status] -> FP")
            elif gt and not pred:
                info['status'] = 'fn'
                fn += 1
                print(f"[Status] -> FN")
            else:
                info['status'] = 'tp'
                tp += 1
                print(f"[Status] -> TP")

            option_f1, option_em, option_al_p, option_al_r = eval_option(gt, pred)
            print(f"[Option Eval]")
            print(f"Option f1: {option_f1}, Option em: {option_em}, Option answer-level precision: {option_al_p}, Option answer-level recall: {option_al_r}")

            reason_f1, reason_em, reason_al_p, reason_al_r = eval_reason(gt, pred)
            print(f"[Reason Eval]")
            print(f"Reason f1: {reason_f1}, Reason em: {reason_em}, Reason answer-level precision: {reason_al_p}, Reason answer-level recall: {reason_al_r}")

            option_f1_list.append(option_f1)
            option_em_list.append(option_em)
            reason_f1_list.append(reason_f1)
            reason_em_list.append(reason_em)
            option_al_precision_list.append(option_al_p)
            option_al_recall_list.append(option_al_r)
            reason_al_precision_list.append(reason_al_p)
            reason_al_recall_list.append(reason_al_r)

            info['option_f1'] = option_f1
            info['option_em'] = option_em
            info['reason_f1'] = reason_f1
            info['reason_em'] = reason_em
            infos.append(info)

        total_precision = tp / (tp + fp) if (tp + fp) != 0 else -1
        total_recall = tp / (tp + fn) if (tp + fn) != 0 else -1
        total_f1 = (2 * total_precision * total_recall / (total_precision + total_recall)) if (total_precision + total_recall) > 0 else 0
        
        print(f"Postprocess error count: {postprocess_error_count}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f'Total - Precision: {total_precision}, Recall: {total_recall}, F1: {total_f1}')
        print(f"Option f1: {np.mean(option_f1_list)}\nOption em: {np.mean(option_em_list)}\nReason f1: {np.mean(reason_f1_list)}\nReason em: {np.mean(reason_em_list)}")
        print(f"Answer Level Option: Precision: {np.mean(option_al_precision_list)}, Recall: {np.mean(option_al_recall_list)}")
        print(f"Answer Level Reason: Precision: {np.mean(reason_al_precision_list)}, Recall: {np.mean(reason_al_recall_list)}")
        print("End")

        success_ratio = (1 - postprocess_error_count / num_samples) * 100
        
        return {
            "success-ratio": success_ratio,
            "total-precision": total_precision * 100 if total_precision > 0 else 0,
            "total-recall": total_recall * 100 if total_recall > 0 else 0,
            "total-f1": total_f1 * 100 if total_f1 > 0 else 0,
            "option-f1": np.mean(option_f1_list) * 100 if option_f1_list else 0,
            "option-em": np.mean(option_em_list) * 100 if option_em_list else 0,
            "reason-f1": np.mean(reason_f1_list) * 100 if reason_f1_list else 0,
            "reason-em": np.mean(reason_em_list) * 100 if reason_em_list else 0,
            "answer-level-option-f1": np.mean(option_al_precision_list) * 100 if option_al_precision_list else 0,
            "answer-level-option-em": np.mean(option_al_recall_list) * 100 if option_al_recall_list else 0,
            "answer-level-reason-f1": np.mean(reason_al_precision_list) * 100 if reason_al_precision_list else 0,
            "answer-level-reason-em": np.mean(reason_al_recall_list) * 100 if reason_al_recall_list else 0,
            "success-total-precision": success_ratio * total_precision if total_precision > 0 else 0,
            "success-total-recall": success_ratio * total_recall if total_recall > 0 else 0,
            "success-total-f1": success_ratio * total_f1 if total_f1 > 0 else 0,
            "success-option-f1": success_ratio * np.mean(option_f1_list) if option_f1_list else 0,
            "success-option-em": success_ratio * np.mean(option_em_list) if option_em_list else 0,
            "success-reason-f1": success_ratio * np.mean(reason_f1_list) if reason_f1_list else 0,
            "success-reason-em": success_ratio * np.mean(reason_em_list) if reason_em_list else 0,
            "success-answer-level-option-f1": success_ratio * np.mean(option_al_precision_list) if option_al_precision_list else 0,
            "success-answer-level-option-em": success_ratio * np.mean(option_al_recall_list) if option_al_recall_list else 0,
            "success-answer-level-reason-f1": success_ratio * np.mean(reason_al_precision_list) if reason_al_precision_list else 0,
            "success-answer-level-reason-em": success_ratio * np.mean(reason_al_recall_list) if reason_al_recall_list else 0,
        }


def v4_result_postprocess(s):
    """Postprocess the results obtained from the MRC model.

    Args:
        s: The result string obtained from the model.

    Returns:
        A tuple containing the processed result and a status message.
    """
    try:
        def _get_content_result(gpt_res):
            content_result_pattern = r'(?<=<Result>).+?(?=</Result>)'
            content_result_matches = list(re.finditer(content_result_pattern, gpt_res, re.DOTALL))
            return content_result_matches[-1].group(0).strip()

        std_result = _get_content_result(s)
        std_result = std_result.lstrip('```json').rstrip('```')  # added by qwen
        answer_list = []
        dedup_map = {}
        d = json.loads(std_result)
        assert isinstance(d, list)

        for item in d:
            assert "Option" in item and "Reason" in item
            assert isinstance(item['Option'], str) and isinstance(item['Reason'], str)
            dedup_map[item['Option']] = item['Reason']

        for k, v in dedup_map.items():
            answer_list.append({'Option': k, 'Reason': v})

        return answer_list, 'postprocess success' if answer_list else 'postprocess success'
    except Exception:
        pass
    return [], 'postprocess error'


def eval_option(gt, pred, debug=False):
    """Evaluate the option predictions against ground truth.

    Args:
        gt: Ground truth answers.
        pred: Predicted answers.
        debug: Whether to output debug information.

    Returns:
        A tuple containing F1 score, EM score, precision, and recall.
    """
    if not gt and not pred:
        return 1, 1, 1, 1
    elif gt and not pred:
        return 0, 0, 0, 0
    elif not gt and pred:
        return 0, 0, 0, 0
    else:
        references = [item['Option'] for item in gt]
        predictions = [item['Option'] for item in pred]

        if debug:
            print(references)

        pred_entity_f1_list = []
        pred_entity_em_list = []

        for prediction in predictions:
            pred_entity_f1_list.append(calc_f1_score(references, prediction))
            pred_entity_em_list.append(calc_em_score(references, prediction))

        record_f1 = np.mean(pred_entity_f1_list)
        record_em = np.mean(pred_entity_em_list)

        if debug:
            print(record_f1, record_em)

        # Answer-level evaluation
        precision, recall = calculate_metrics(references, predictions, threshold=OPTION_F1_THRESHOLD)
    
    return record_f1, record_em, precision, recall


def eval_reason(gt, pred, debug=False):
    """Evaluate the reason predictions against ground truth.

    Args:
        gt: Ground truth answers.
        pred: Predicted answers.
        debug: Whether to output debug information.

    Returns:
        A tuple containing F1 score, EM score, precision, and recall.
    """
    if not gt and not pred:
        return 1, 1, 1, 1
    elif gt and not pred or not gt and pred:
        return 0, 0, 0, 0
    else:
        references = [item['Reason'] for item in gt]
        predictions = [item['Reason'] for item in pred]

        if debug:
            print(references)

        pred_entity_f1_list = []
        pred_entity_em_list = []

        for prediction in predictions:
            pred_entity_f1_list.append(calc_f1_score(references, prediction))
            pred_entity_em_list.append(calc_em_score(references, prediction))

        record_f1 = np.mean(pred_entity_f1_list)
        record_em = np.mean(pred_entity_em_list)

        if debug:
            print(record_f1, record_em)

        # Answer-level evaluation
        precision, recall = calculate_metrics(references, predictions, threshold=REASON_F1_THRESHOLD)

    return record_f1, record_em, precision, recall


def calculate_similarity(entity1, entity2):
    """Calculate the similarity between two entities.

    Args:
        entity1: The first entity (string).
        entity2: The second entity (string).

    Returns:
        The F1 score representing similarity.
    """
    e1_segs = _tokenize_chars(_normalize(entity1))
    e2_segs = _tokenize_chars(_normalize(entity2))
    lcs, lcs_len = find_lcs(e1_segs, e2_segs)

    if lcs_len == 0:
        return 0

    prec = 1.0 * lcs_len / len(e1_segs)
    rec = 1.0 * lcs_len / len(e2_segs)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


def calculate_metrics(refs, preds, threshold=OPTION_F1_THRESHOLD):
    """Calculate precision and recall between predictions and references.

    Args:
        refs: Ground truth answers.
        preds: Predicted answers.
        threshold: Threshold for similarity.

    Returns:
        A tuple containing precision and recall.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred in preds:
        if any(calculate_similarity(pred, ref) > threshold for ref in refs):
            true_positives += 1
        else:
            false_positives += 1

    for ref in refs:
        if not any(calculate_similarity(ref, pred) > threshold for pred in preds):
            false_negatives += 1

    if true_positives == 0:
        return 0, 0
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall


def calc_f1_score(answers, prediction, debug=False):
    """Calculate F1 score for the given prediction against answers.

    Args:
        answers: List of ground truth answers.
        prediction: The predicted answer.
        debug: Whether to output debug information.

    Returns:
        The maximum F1 score.
    """
    f1_scores = []

    for ans in answers:
        ans_segs = _tokenize_chars(_normalize(ans))
        prediction_segs = _tokenize_chars(_normalize(prediction))

        if debug:
            print(json.dumps(ans_segs, ensure_ascii=False))
            print(json.dumps(prediction_segs, ensure_ascii=False))

        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)

        if lcs_len == 0:
            f1_scores.append(0)
            continue

        prec = 1.0 * lcs_len / len(prediction_segs)
        rec = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)

    return max(f1_scores)


def calc_em_score(answers, prediction):
    """Calculate exact match score for the given prediction against answers.

    Args:
        answers: List of ground truth answers.
        prediction: The predicted answer.

    Returns:
        The exact match score.
    """
    for ans in answers:
        if _normalize(ans) == _normalize(prediction):
            return 1
    return 0


def _tokenize_chars(text):
    """Tokenize characters in the given text.

    Args:
        text: Input text as a unicode string.

    Returns:
        A list of tokenized segments.
    """
    def _is_chinese_char(cp):
        """Check whether CP is the codepoint of a CJK character."""
        return (
            (0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x2F800 <= cp <= 0x2FA1F)
        )

    output = []
    buff = ""

    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff:
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff:
        output.append(buff)

    return output


def _normalize(in_str):
    """Normalize the input unicode string.

    Args:
        in_str: Input string.

    Returns:
        The normalized string.
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    
    out_segs = [char for char in in_str if char not in sp_char]
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """Find the longest common subsequence between s1 and s2.

    Args:
        s1: First sequence (list).
        s2: Second sequence (list).

    Returns:
        A tuple containing the longest common subsequence and its length.
    """
    m = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    max_len = 0
    p = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > max_len:
                    max_len = m[i + 1][j + 1]
                    p = i + 1

    return s1[p - max_len:p], max_len