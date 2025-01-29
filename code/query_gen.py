import os
import re
import json
import torch
import numpy as np

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_evaluator.icl_anls_evaluator import calculate_anls_score
from sentence_transformers import SentenceTransformers, util


@LOAD_DATASET.register_module()
class QueryGenDataset(BaseDataset):
    """Custom dataset for query generation tasks."""

    @staticmethod
    def load(path: str, name: str, version: str = None) -> Dataset:
        """Load the query generation dataset from a JSON file.

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


class QueryGenEvaluator(BaseEvaluator):
    """Custom evaluator for query generation tasks."""

    def score(self, predictions, references, **kwargs) -> dict:
        """Evaluate the predictions against references.

        Args:
            predictions: List of predicted answers.
            references: List of ground truth answers.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary of evaluation scores.
        """
        print("\n\n[Note: QueryGen Evaluator]\n\n")
        assert len(predictions) == len(references), (
            "Predictions and references should have the same length."
        )
        print(f"kwargs: {kwargs}")
        
        similarity_scores = []
        scores = []
        
        # calculate embedding similarity
        embed_model = SentenceTransformers(
            self.embed_model_path,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        for pred, ref in zip(predictions, references):
            truncated_pred = re.split(r'[\n]|<|endofresponse|>', pred.strip(), 1)[0]
            embed_pred = embed_model.encode(truncated_pred, convert_to_tensor=True)
            embed_ref = embed_model.encode(ref, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embed_pred, embed_ref).item() * 100
            anls = calculate_anls_score(pred, ref)
            scores.append((similarity + anls) / 2)
            
        return {
            "final_scores": np.mean(scores),
        }

