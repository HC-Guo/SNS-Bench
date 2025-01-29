import os
import re
import json
from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class ChlwDataset(BaseDataset):
    """Custom dataset for CHLW."""

    @staticmethod
    def load(path: str, name: str, version: str = None) -> Dataset:
        """Load dataset from a JSON file.

        Args:
            path (str): The directory path containing the dataset file.
            name (str): The base name of the dataset file.
            version (str, optional): The version of the dataset.

        Returns:
            Dataset: A Dataset object containing the loaded data.
        """
        dataset = []
        if version:
            name = f"{name}_{version}"
        file_path = os.path.join(path, f"{name}.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            dataset.append({
                "content": item["content"],
                "answer": item["answer"],
            })

        return Dataset.from_list(dataset)


class ChlwEvaluator(BaseEvaluator):
    """Custom evaluator for CHLW."""

    def score(self, predictions: list, references: list, **kwargs) -> dict:
        """Calculate evaluation metrics.

        Args:
            predictions (list): A list of predicted outputs.
            references (list): A list of ground truth references.
            kwargs: Additional arguments containing origin prompts.

        Returns:
            dict: A dictionary with evaluation metrics.
        """
        print("\n\n[Note - CHLW Evaluator]\n\n")
        assert len(predictions) == len(references), (
            "Predictions and references should have the same length.")
        
        prompts = kwargs["origin_prompt"]
        f1_score = 0
        precision = 0
        recall = 0
        error_count = 0

        for prediction, reference, prompt in zip(predictions, references, prompts):
            print("[Test]")
            try:
                reference = eval(reference) or [""]
                prediction = self._extract_prediction(prediction)
                assert isinstance(prediction, list), (
                    "Prediction and reference must be lists.")
                prediction = prediction or [""]
                assert all(isinstance(pred, str) for pred in prediction), (
                    "Prediction and reference must be lists of strings.")
            except Exception as e:
                print(f"Literal-Eval Error ({e}):\nPrediction: {prediction}\nReference: {reference}\n")
                error_count += 1
                continue

            print(f"Literal-Eval Success:\nprediction: {prediction}\nreference: {reference}\n")

            if prediction == [""] and reference == [""]:
                print("Prediction and reference are both empty.\n\n")

            results = set(prediction)
            labels = set(reference)
            f1_score += len(labels & results)
            precision += len(results)
            recall += len(labels)


        success_ratio = (1 - error_count / len(predictions)) * 100

        return {
            'success_ratio': success_ratio,
            'f1': round(f1_score * 2 / (precision + recall + 0.0001), 4) * 100,
            'precision': round(f1_score / (precision + 0.0001), 4) * 100,
            'recall': round(f1_score / (recall + 0.0001), 4) * 100,
            'success_f1': success_ratio * (f1_score * 2 / (precision + recall + 0.0001)),
            'success_precision': success_ratio * (f1_score / (precision + 0.0001)),
            'success_recall': success_ratio * (f1_score / (recall + 0.0001)),
        }

    def _extract_prediction(self, prediction: str) -> list:
        """Extracts and evaluates the prediction.

        Args:
            prediction (str): The raw prediction string.

        Returns:
            list: The evaluated prediction as a list.
        """
        match = re.compile(r'\[[^\[\]]*\]', re.DOTALL).search(prediction)
        if match:
            prediction = match.group(0)
        return eval(prediction)
