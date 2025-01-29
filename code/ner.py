import os
import re
import json

from datasets import Dataset
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset


@LOAD_DATASET.register_module()
class NERDataset(BaseDataset):
    """Custom dataset for Named Entity Recognition (NER) tasks."""

    @staticmethod
    def load(path: str, name: str, version: str = None) -> Dataset:
        """Load the NER dataset from a JSON file.

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
                "question": item["question"],
                "content": item["content"],
                "answer": item["answer"],
            })

        return Dataset.from_list(dataset)


@TEXT_POSTPROCESSORS.register_module()
def ner_postprocess(text: str, **kwargs) -> list:
    """Custom postprocess function for NER outputs.

    Args:
        text: The input text to process.
        **kwargs: Additional keyword arguments.

    Returns:
        A list of processed text segments.
    """
    text = text.replace("Answer:", "").strip()
    delimiter = kwargs.get("delimiter", ",")
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n]|<|endofresponse|>', text, 1)[0]
    truncated_text_segments = truncated_text.split(delimiter)

    final_text = [t.strip() for t in truncated_text_segments if t]
    
    return final_text