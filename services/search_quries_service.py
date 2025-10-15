from abc import ABC, abstractmethod
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from schemas.text_schemas import SearchQueryExtractor


class TransformersSearchQueryExtractor(SearchQueryExtractor):
    
    """Transformer-based implementation of the SearchQueryExtractor interface."""

    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the lightweight transformer model for search query generation.

        Args:
            model_name: Hugging Face model name (default: 'google/flan-t5-small').
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def extract(self, text: str, num_queries: int = 5) -> List[str]:
        """
        Generate search-like queries using the transformer model.

        Args:
            text: The input paragraph.
            num_queries: Number of queries to generate.

        Returns:
            List[str]: A list of extracted search queries.
        """
        prompt = (
            f"Generate {num_queries} useful and distinct search queries "
            f"from the following paragraph:\n{text.strip()}"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=96, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean and split queries
        queries = [
            q.strip("-• \n").rstrip(".")
            for q in generated_text.split("\n")
            if q.strip()
        ]
        return queries
