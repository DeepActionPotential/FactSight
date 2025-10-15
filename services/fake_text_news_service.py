from abc import ABC, abstractmethod
import re
import torch
import contractions
import nltk
from nltk.corpus import stopwords
import torch.nn as nn

from schemas.text_schemas import FakeNewsDetector
from models.models import LSTMClassifier

# Download stopwords (first run only)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


class FakeTextNewsDetector(FakeNewsDetector):
    """
    A class for detecting fake news in text articles using a pre-trained LSTM model.

    This class loads a trained PyTorch LSTM classifier and corresponding vocabulary
    to evaluate whether a given text is real or fake. It performs preprocessing,
    tokenization, and encoding before feeding the text to the neural network.

    Attributes:
        model_path (str): Path to the saved PyTorch model file (.pt).
        vocab_path (str): Path to the saved vocabulary mapping file (.pt).
        device (torch.device): The computing device used for inference ('cuda' or 'cpu').
        model (nn.Module): The loaded LSTM model for fake news detection.
        word2idx (dict): Mapping of words to integer indices for token encoding.
    """

    def __init__(self, model_path: str = "models/model.pt", vocab_path: str = "models/word2idx.pt", device=None):
        """
        Initializes the FakeTextNewsDetector.

        Loads the trained model and vocabulary, sets up the computing device, and
        prepares the model for inference.

        Args:
            model_path (str): Path to the pre-trained LSTM model (.pt). Defaults to "models/model.pt".
            vocab_path (str): Path to the vocabulary mapping file (.pt). Defaults to "models/word2idx.pt".
            device (torch.device, optional): Torch device ('cuda' or 'cpu').
                If not provided, automatically detects CUDA availability.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.serialization.add_safe_globals([LSTMClassifier])

        # Load vocabulary dictionary
        self.word2idx = torch.load(vocab_path, map_location=self.device)
        vocab_size = len(self.word2idx)

        # Load the LSTM model
        self.model = LSTMClassifier(vocab_size=vocab_size)
        loaded_model = torch.load(model_path, map_location=self.device, weights_only=False)

        # Support both raw state_dict and full model serialization
        if isinstance(loaded_model, dict):
            self.model.load_state_dict(loaded_model)
        else:
            self.model.load_state_dict(loaded_model.state_dict())

        self.model.to(self.device)
        self.model.eval()

    def _clean_text(self, text: str):
        """
        Cleans and tokenizes the input text for model inference.

        Steps include:
          - Expanding contractions (e.g., "don't" → "do not")
          - Removing non-ASCII characters, HTML tags, URLs, and special symbols
          - Lowercasing all text
          - Removing single-letter words and stopwords

        Args:
            text (str): The input raw text.

        Returns:
            list[str]: A list of cleaned and filtered tokens.
        """
        text = contractions.fix(text)
        text = text.encode("ascii", "ignore").decode()
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\b\w{1}\b", "", text)
        tokens = [w for w in text.split() if w not in stop_words]
        return tokens

    def _encode(self, tokens, max_len: int = 300):
        """
        Encodes tokens into a fixed-length tensor for model input.

        Each token is mapped to its index in the loaded vocabulary.
        If the token is unknown, a special <UNK> index is used.
        The sequence is padded or truncated to the specified length.

        Args:
            tokens (list[str]): List of word tokens.
            max_len (int): Maximum sequence length. Defaults to 300.

        Returns:
            torch.Tensor: Encoded token tensor of shape (1, max_len).
        """
        ids = [self.word2idx.get(w, self.word2idx.get("<UNK>", 1)) for w in tokens]
        if len(ids) < max_len:
            ids += [self.word2idx.get("<PAD>", 0)] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def detect(self, text: str) -> bool:
        """
        Predicts whether the given text is fake or real.

        The method preprocesses the text, encodes it into numerical form,
        and runs inference using the LSTM model. The output score is a probability
        between 0 and 1, where values above 0.5 indicate fake news.

        Args:
            text (str): The input text (article or sentence).

        Returns:
            bool: True if the text is predicted as fake, False if real.
        """
        tokens = self._clean_text(text)
        seq = self._encode(tokens).to(self.device)
        with torch.no_grad():
            score = self.model(seq).item()
        return score >= 0.5
