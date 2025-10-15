from transformers import pipeline, AutoTokenizer
from schemas.text_schemas import EmotionResult, EmotionDetector
from typing import List, Dict

class TransformersEmotionDetector(EmotionDetector):
    """Emotion detector using a lightweight DistilRoBERTa model from Hugging Face."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion detection model.
        Args:
            model_name: Pretrained Hugging Face model for emotion detection.
        """
        self.model_name = model_name
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )
        # Load tokenizer for proper text truncation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def analyze(self, text: str) -> EmotionResult:
        """
        Analyze emotion in the given text using chunked processing for long texts.
        Args:
            text: The text to analyze.
        Returns:
            EmotionResult: Structured result containing dominant emotion and confidence.
        """
        # Split text into chunks if it's too long
        max_tokens_per_chunk = 400  # Conservative limit to leave room for special tokens
        chunks = self._split_text_into_chunks(text, max_tokens_per_chunk)

        if len(chunks) == 1:
            # Single chunk - use original logic
            results: List[List[Dict[str, float]]] = self.pipeline(chunks[0])
        else:
            # Multiple chunks - analyze each and aggregate results
            results: List[List[Dict[str, float]]] = self.pipeline(chunks)

        # Aggregate emotion scores across all chunks
        aggregated_scores = self._aggregate_emotion_scores(results)

        # Get the most likely emotion
        dominant_emotion = max(aggregated_scores, key=aggregated_scores.get)
        confidence = aggregated_scores[dominant_emotion]

        return EmotionResult(
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            all_scores=aggregated_scores
        )

    def _split_text_into_chunks(self, text: str, max_tokens_per_chunk: int) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        Args:
            text: The text to split
            max_tokens_per_chunk: Maximum tokens per chunk
        Returns:
            List of text chunks
        """
        # Split text into sentences first for better chunk boundaries
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add period back if it was removed
            if not sentence.endswith('.'):
                sentence += '.'

            sentence_tokens = len(self.tokenizer.tokenize(sentence))

            # If adding this sentence would exceed limit, start new chunk
            if current_tokens + sentence_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _aggregate_emotion_scores(self, results: List[List[Dict[str, float]]]) -> Dict[str, float]:
        """
        Aggregate emotion scores from multiple chunks.
        Args:
            results: List of emotion classification results from each chunk
        Returns:
            Dictionary of aggregated emotion scores
        """
        if not results:
            return {}

        # Collect all emotion scores with weights
        emotion_totals = {}
        emotion_weights = {}

        for chunk_results in results:
            # Get confidence scores for this chunk
            chunk_scores = {entry["label"]: entry["score"] for entry in chunk_results}

            # Weight by confidence (more confident predictions get higher weight)
            total_confidence = sum(chunk_scores.values())

            for emotion, score in chunk_scores.items():
                weight = score / total_confidence if total_confidence > 0 else 0
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
                emotion_weights[emotion] = emotion_weights.get(emotion, 0) + 1  # Simple count for now

        # Average the scores across chunks
        aggregated_scores = {}
        for emotion in emotion_totals:
            # Use weighted average based on number of chunks that detected this emotion
            aggregated_scores[emotion] = emotion_totals[emotion] / emotion_weights[emotion]

        return aggregated_scores
