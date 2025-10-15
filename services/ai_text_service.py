import joblib
import re
import string
from typing import Any
from nltk.corpus import stopwords


from schemas.text_schemas import AITextDetector


class NBAITextDetector(AITextDetector):
    """
    Naive Bayes AI Text Detector that classifies whether a text
    is AI-generated or human-written using a pre-trained joblib model.

    Attributes:
        model_path (str): Path to the saved joblib model.
        model (Any): Loaded ML model for prediction.
        min_words (int): Minimum number of words required for valid detection.
    """

    def __init__(self, model_path: str = "./models/ai_text_detector.joblib", min_words: int = 0):
        """
        Initialize the NBAITextDetector.

        Args:
            model_path (str, optional): Path to the trained joblib model file.
            min_words (int, optional): Minimum number of words for detection.
        """
        self.model_path = model_path
        self.min_words = min_words
        self.model = self._load_model()
        self.stop_words = set(stopwords.words('english'))

    def _load_model(self) -> Any:
        """Load the trained joblib model."""
        return joblib.load(self.model_path)

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'\d+', '', text)
        text = ''.join(ch for ch in text if ch.isprintable())
        return text

    def detect(self, text: str) -> bool:
        """
        Detect whether a given text is AI-generated.

        Args:
            text (str): Input text to classify.

        Returns:
            bool: True if AI-generated, False if human-written.
        """
        if len(text.split()) < self.min_words:
            raise ValueError(f"Text must be at least {self.min_words} words long.")

        processed = self._preprocess_text(text)
        prediction = self.model.predict([processed])
        return bool(int(prediction[0]) == 1) 
