import os
import re
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset


@LOAD_DATASET.register_module()
class HashtagDataset(BaseDataset):
    """Custom dataset for hashtag tasks."""

    @staticmethod
    def load(path: str, name: str, subset: str, version: str = None) -> Dataset:
        """Load the hashtag dataset from a JSON file.

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
            if subset in ["single", "multi"]:
                dataset.append({
                    "content": item["content"],
                    "candidates": item["candidates"],
                    "answer": item["answer"],
                })
            else:
                raise ValueError(f"Unknown subset: {subset}")

        return Dataset.from_list(dataset)


@TEXT_POSTPROCESSORS.register_module()
def hashtag_postprocess(text: str, **kwargs) -> str | list:
    """Custom postprocess function for hashtag task.

    Args:
        text: The input text to process.
        **kwargs: Additional keyword arguments.

    Returns:
        Processed text as a string or list of strings.
    """
    text = text.replace("答案：", "").replace("Answer:", "").strip()
    subset = kwargs.get("subset")

    if subset == "single":
        truncated_text = re.split(r'[\n.,]|<|endofresponse|>|<|end|>|<|im_end|>', text, 1)
        return truncated_text[0].strip() if truncated_text else ""
    
    elif subset == "multi":
        delimiter = kwargs.get("delimiter", ",")
        truncated_text = re.split(r'[\n]|<|endofresponse|>', text, 1)[0]
        return [t.strip() for t in truncated_text.split(delimiter) if t]
    
    return text