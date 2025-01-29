import os
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset


@LOAD_DATASET.register_module()
class QueryCorrExplainDataset(BaseDataset):
    """Custom dataset for query correction and explanation tasks."""

    @staticmethod
    def load(path: str, name: str, version: str = None) -> Dataset:
        """Load the query correction explanation dataset from a JSON file.

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
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "answer": item["answer"],
            })

        return Dataset.from_list(dataset)