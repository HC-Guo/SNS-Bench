import os
import json
import re
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class QueryCorrTopicDataset(BaseDataset):
    """Custom dataset for query correction based on topic."""

    @staticmethod
    def load(path: str, name: str, version: str = None) -> Dataset:
        """Load the query correction topic dataset from a JSON file.

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


LEVELS = ['-1', '0', '1', '2', '3']


class QueryCorrTopicEvaluator(BaseEvaluator):
    """Custom evaluator for query correction based on topic."""

    def score(self, predictions, references, **kwargs) -> dict:
        """Evaluate predictions against references.

        Args:
            predictions: List of predicted answers.
            references: List of ground truth answers.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary of evaluation scores.
        """
        assert len(predictions) == len(references), (
            "The number of predictions must equal the number of references."
        )

        counter = {level: {"tp": 0, "fp": 0, "fn": 0} for level in LEVELS}
        pattern = r"(-1|0|1|2|3)"
        error_count = 0
        
        prompts = kwargs["origin_prompt"]
        
        for pred, ref, prompt in zip(predictions, references, prompts):
            matches = re.findall(pattern, pred)
            print(f"Matches:\n{matches}\n")
            if len(matches) == 0:
                error_count += 1
                continue
            
            pred = matches[0]
            if pred == ref:
                counter[ref]["tp"] += 1
            else:
                counter[ref]["fn"] += 1
                counter[pred]["fp"] += 1
        
        scores = {"success-ratio": (1 - error_count / len(predictions)) * 100}
        
        for level in LEVELS:
            scores[f"level({level})-precision"] = (
                counter[level]["tp"] / (counter[level]["tp"] + counter[level]["fp"] + 1e-5) * 100
            )
            scores[f"level({level})-recall"] = (
                counter[level]["tp"] / (counter[level]["tp"] + counter[level]["fn"] + 1e-5) * 100
            )
            scores[f"level({level})-f1"] = (
                2 * scores[f"level({level})-precision"] * scores[f"level({level})-recall"] /
                (scores[f"level({level})-precision"] + scores[f"level({level})-recall"] + 1e-5)
            )
        
        scores["success-macro-f1"] = np.mean([scores[f"level({level})-f1"] for level in LEVELS]) * (1 - error_count / len(predictions))
        
        return scores