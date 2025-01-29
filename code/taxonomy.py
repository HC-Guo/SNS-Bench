import os
import re
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset


@LOAD_DATASET.register_module()
class TaxonomyDataset(BaseDataset):
    """Custom dataset for taxonomy classification tasks."""

    @staticmethod
    def load(path: str, name: str, subset: str, version: str = None) -> Dataset:
        """Load the taxonomy dataset from a JSON file.

        Args:
            path: The directory path containing the dataset file.
            name: The name of the dataset file (without extension).
            subset: The subset of the dataset to load ('single' or 'multi').
            version: Optional version string to append to the filename.

        Returns:
            A Dataset object containing the loaded data.

        Raises:
            ValueError: If an unknown subset is specified.
        """
        dataset = []
        if version:
            name = f"{name}_{version}"
        
        file_path = os.path.join(path, f"{name}.json")
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        for item in data:
            if subset == "single":
                dataset.append({
                    "content": item["content"],
                    "candidates": item["candidates"],
                    "answer": item["answer"],
                })
            elif subset == "multi":
                dataset.append({
                    "content": item["content"],
                    "candidates_primary": item["candidates_primary"],
                    "candidates_secondary": item["candidates_secondary"],
                    "candidates_tertiary": item["candidates_tertiary"],
                    "answer": [item["answer"]],
                })
            else:
                raise ValueError(f"Unknown subset name: {subset}")

        return Dataset.from_list(dataset)


@TEXT_POSTPROCESSORS.register_module()
def taxonomy_postprocess(text: str, **kwargs) -> str:
    """Custom postprocess function for taxonomy outputs.

    Args:
        text: The input text to process.
        **kwargs: Additional keyword arguments.

    Returns:
        The processed text as a string.
    """
    text = text.replace("答案：", "").replace("Answer:", "").strip()
    truncated_text = re.split(r'[\n.,]|<|endofresponse|>|<|end|>')