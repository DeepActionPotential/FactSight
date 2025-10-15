from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


class AITextDetector(ABC):
    """
    Abstract base class for all AI text detectors.
    Defines the interface that all concrete detectors must implement.
    """

    @abstractmethod
    def detect(self, text: str) -> bool:
        """
        Detect whether the given text is AI-generated.

        Args:
            text (str): The input text to analyze.

        Returns:
            bool: True if the text is AI-generated, False if it is human-written.
        """
        pass


class EmotionDetector(ABC):
    """Abstract base class for emotion detection."""

    @abstractmethod
    def analyze(self, text: str):
        """Analyze the emotion in the given text and return a structured result."""
        pass


@dataclass
class EmotionResult:
    """Structured output for detected emotions."""
    dominant_emotion: str
    confidence: float
    all_scores: Dict[str, float]


class SearchQueryExtractor(ABC):
    """Abstract base class for extracting search queries from text."""

    @abstractmethod
    def extract(self, text: str, num_queries: int = 5) -> List[str]:
        """
        Extract search-like queries from a given paragraph.

        Args:
            text: The input text to extract queries from.
            num_queries: Number of queries to generate.

        Returns:
            List[str]: A list of extracted search queries.
        """
        pass




from abc import ABC, abstractmethod


# ===== Abstract base class =====
class FakeNewsDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> int:
        """Return 1 for real news, 0 for fake news."""
        pass
