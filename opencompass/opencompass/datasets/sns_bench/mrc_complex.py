import os
import re
import json
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class NoteMRCComplexDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        from modelscope.msdatasets import MsDataset
        
        dataset = []
        ds =  MsDataset.load(path, subset_name=name, split='test')
        data = [d  for d in ds]
        for item in data:
            dataset.append({
                "content"   : item["content"],
                "answer"    : item["answer"],
            })
        dataset = Dataset.from_list(dataset)
        return dataset



OPTION_F1_TRHES=0.8
REASON_F1_THRES=0.5

class NoteMRCComplexEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        assert len(predictions) == len(references), (
            "predictions and references should have the same length")        

        num_samples = len(references)
        infos = []
        tp = tn = fp = fn = 0
        postprocess_error_cnt = 0
        
        # ngram level
        option_f1_list = []
        option_em_list = []
        reason_f1_list = []
        reason_em_list = []
        # answer level
        option_al_p_list = []
        option_al_r_list = []
        reason_al_p_list = []
        reason_al_r_list = []


        for pred, gt in zip(predictions, references):
            info = {}
            pred, pred_info = v4_result_postprocess(pred)
            gt, gt_info = v4_result_postprocess(gt)
            print(f"[Postprocess]\nGT_Info: {gt_info}\nGT: {gt}\nPred_Info: {pred_info}\nPred: {pred}")
            if gt_info == 'postprocess error' or pred_info == 'postprocess error':
                postprocess_error_cnt += 1
                continue
            if gt == [] and pred == []:
                info['status'] = 'tn'
                tn += 1
            elif gt == [] and pred != []:
                info['status'] = 'fp'
                fp += 1
            elif gt != [] and pred == []:
                info['status'] = 'fn'
                fn += 1
            else:
                info['status'] = 'tp'
                # all has answers
                tp += 1
            option_f1, option_em, option_al_p, option_al_r = eval_option(gt, pred)
            print(f"[Option Eval]")
            print(f"Option f1: {option_f1}, Option em: {option_em}, Option answer level precision: {option_al_p}, Option answer level recall: {option_al_r}")
            reason_f1, reason_em, reason_al_p, reason_al_r = eval_reason(gt, pred)
            print(f"[Reason Eval]")
            print(f"Reason f1: {reason_f1}, Reason em: {reason_em}, Reason answer level precision: {reason_al_p}, Reason answer level recall: {reason_al_r}")
            option_f1_list.append(option_f1)
            option_em_list.append(option_em)
            reason_f1_list.append(reason_f1)
            reason_em_list.append(reason_em)
            option_al_p_list.append(option_al_p)
            option_al_r_list.append(option_al_r)
            reason_al_p_list.append(reason_al_p)
            reason_al_r_list.append(reason_al_r)
            info['option_f1'] = option_f1
            info['option_em'] = option_em
            info['reason_f1'] = reason_f1
            info['reason_em'] = reason_em
            infos.append(info)
        
        total_precision = tp / (tp + fp) if (tp + fp) != 0 else -1
        total_recall = tp / (tp + fn) if (tp + fn) != 0 else -1
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
        print(f"Postprocess error count: {postprocess_error_cnt}")
        print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
        print(f'Totally, precision {total_precision}, recall {total_recall}, f1 {total_f1}')
        print(f"Option f1: {np.mean(option_f1_list)}\nOption em: {np.mean(option_em_list)}\nReason f1: {np.mean(reason_f1_list)}\nReason em: {np.mean(reason_em_list)}")
        print(f"AnswerLevel_option: precision: {np.mean(option_al_p_list)}, recall: {np.mean(option_al_r_list)}")
        print(f"AnswerLevel_reason: precision: {np.mean(reason_al_p_list)}, recall: {np.mean(reason_al_r_list)}")
        print("end")
        
        success_ratio = (1 - postprocess_error_cnt / num_samples) * 100
        
        return {
                "success-ratio"         : success_ratio,
                "total-precision"       : total_precision * 100 if total_precision > 0 else 0,
                "total-recall"          : total_recall * 100 if total_precision > 0 else 0,
                "total-f1"              : total_f1 * 100 if total_f1 > 0 else 0,
                "option-f1"             : np.mean(option_f1_list) * 100   if option_f1_list else 0,
                "option-em"             : np.mean(option_em_list) * 100   if option_em_list else 0,
                "reason-f1"             : np.mean(reason_f1_list) * 100   if reason_f1_list else 0,
                "reason-em"             : np.mean(reason_em_list) * 100   if reason_em_list else 0,
                "answer-level-option-f1": np.mean(option_al_p_list) * 100 if option_al_p_list else 0,
                "answer-level-option-em": np.mean(option_al_r_list) * 100 if option_al_p_list else 0,
                "answer-level-reason-f1": np.mean(reason_al_p_list) * 100 if reason_al_p_list else 0,
                "answer-level-reason-em": np.mean(reason_al_r_list) * 100 if reason_al_r_list else 0,

                "success-total-precision"       : success_ratio * total_precision           if total_precision > 0 else 0,
                "success-total-recall"          : success_ratio * total_recall              if total_precision > 0 else 0,
                "success-total-f1"              : success_ratio * total_f1                  if total_f1 > 0 else 0,
                "success-option-f1"             : success_ratio * np.mean(option_f1_list)   if option_f1_list else 0,
                "success-option-em"             : success_ratio * np.mean(option_em_list)   if option_em_list else 0,
                "success-reason-f1"             : success_ratio * np.mean(reason_f1_list)   if reason_f1_list else 0,
                "success-reason-em"             : success_ratio * np.mean(reason_em_list)   if reason_em_list else 0,
                "success-answer-level-option-f1": success_ratio * np.mean(option_al_p_list) if option_al_p_list else 0,
                "success-answer-level-option-em": success_ratio * np.mean(option_al_r_list) if option_al_p_list else 0,
                "success-answer-level-reason-f1": success_ratio * np.mean(reason_al_p_list) if reason_al_p_list else 0,
                "success-answer-level-reason-em": success_ratio * np.mean(reason_al_r_list) if reason_al_r_list else 0,
            }


def v4_result_postprocess(s):
    try:
        def _get_content_result(gpt_res):
            content_result_pattern = r'(?<=<Result>).+?(?=</Result>)'
            content_result_matches = list(re.finditer(content_result_pattern, gpt_res, re.DOTALL))
            return content_result_matches[-1].group(0).strip()
        std_result = _get_content_result(s)
        std_result = std_result.lstrip('```json').rstrip('```') # added by qwen
        answer_list = []
        dedup_map = {}
        d=json.loads(std_result)
        assert isinstance(d, list)
        for item in d:
            assert "Option" in item and "Reason" in item
            assert isinstance(item['Option'], str) and isinstance(item['Reason'], str)
            dedup_map[item['Option']] = item['Reason']
        for k, v in dedup_map.items():
            answer_list.append({'Option': k, 'Reason': v})
        if len(answer_list) > 0:
            return answer_list, 'postprocess success'
        else:
            return answer_list, 'postprocess success'
    except:
        pass
    return [], 'postprocess error'



def eval_option(gt, pred, debug=False):
    if gt == [] and pred == []:
        return 1, 1, 1, 1
    elif (gt != [] and pred == []):
        return 0, 0, 0, 0
    elif (gt == [] and pred != []):
        return 0, 0, 0, 0
    else:
        references = [item['Option'] for item in gt]
        predictions = [item['Option'] for item in pred]
        if debug: print(references)
        pred_entity_f1_list = []
        pred_entity_em_list = []
        for predition in predictions:
            pred_entity_f1_list.append(calc_f1_score(references, predition))
            pred_entity_em_list.append(calc_em_score(references, predition))
        
        record_f1 = np.mean(pred_entity_f1_list)
        record_em = np.mean(pred_entity_em_list)
        if debug: print(record_f1, record_em)
        # answer-level evaluation
        precision, recall = calculate_metrics(references, predictions,threshold=OPTION_F1_TRHES)
    # print(precision, recall)
    return record_f1, record_em, precision, recall


def eval_reason(gt, pred, debug=False):
    if gt == [] and pred == []:
        return 1, 1, 1, 1
    elif (gt != [] and pred == []) or (gt == [] and pred != []):
        return 0, 0, 0, 0
    else:
        references = [item['Reason'] for item in gt]
        predictions = [item['Reason'] for item in pred]
        if debug: print(references)
        pred_entity_f1_list = []
        pred_entity_em_list = []
        for predition in predictions:
            pred_entity_f1_list.append(calc_f1_score(references, predition))
            pred_entity_em_list.append(calc_em_score(references, predition))
        
        record_f1 = np.mean(pred_entity_f1_list)
        record_em = np.mean(pred_entity_em_list)
        if debug: print(record_f1, record_em)
        # answer-level evaluation
        precision, recall = calculate_metrics(references, predictions, threshold=REASON_F1_THRES)
    return record_f1, record_em, precision, recall


def calculate_similarity(entity1, entity2):
    """ Two entities'(texts') similarity for answer-level"""
    e1_segs = _tokenize_chars(_normalize(entity1))
    e2_segs = _tokenize_chars(_normalize(entity2))
    lcs, lcs_len = find_lcs(e1_segs, e2_segs)
    if lcs_len == 0:
        return 0
    prec = 1.0 * lcs_len / len(e1_segs)
    rec = 1.0 * lcs_len / len(e2_segs)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

def calculate_metrics(refs, preds, threshold=OPTION_F1_TRHES):
    """ Calculate precision and recall between predictions and references for answer-level 
    compare (pred-ref-level) f1 score with threshold to calculate (answer-level) f1 score"""
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
        prec = 1.0*lcs_len/len(prediction_segs)
        rec = 1.0*lcs_len/len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def _tokenize_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > max_len:
                    max_len = m[i+1][j+1]
                    p = i+1
    return s1[p-max_len:p], max_len
