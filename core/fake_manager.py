from schemas.text_schemas import AITextDetector
from schemas.vision_schemas import FakeFaceDetector
from schemas.vision_schemas import FaceDetector
from schemas.text_schemas import FakeNewsDetector
from schemas.vision_schemas import AIImageDetector
from schemas.text_schemas import EmotionDetector
from schemas.text_schemas import SearchQueryExtractor
from schemas.fake_manager_schemas import News, AggregatedNewsAnalysis, ImageAnalysis
from schemas.vision_schemas import FaceMainPoints
from services.fact_search_service import FactCheckService
from utils.utils import open_image
from typing import List, Optional, Union
from PIL import Image as PILImage
from pathlib import Path
import uuid
from datetime import datetime


class FakeNewsManager:
    """Manager that aggregates multiple detectors and services to analyze a single
    news item (text + images) and produce an aggregated analysis.

    Responsibilities:
    - Run text-based detectors (AI text detector, fake-news/text classifier).
    - Extract search queries and run fact-checking.
    - Run emotion analysis on the text.
    - Run image-level detectors (face detection, AI-image detection, deepfake
      face detection) and crop faces for per-face analysis.

    Attributes:
        ai_text_detector: Optional AI text detector; must provide `.detect(text) -> bool|float|None`.
        fake_face_detector: Optional face-level deepfake detector; must provide `.detect(pil_image) -> bool|float|None`.
        face_detector: Optional face detector; must provide `.detect(pil_image) -> list[FaceMainPoints]`.
        news_detector: Optional fake-news/text detector; must provide `.detect(text) -> bool|float|None`.
        ai_image_detector: Optional AI-image detector; must provide `.detect(pil_image) -> bool|float|None`.
        query_extractor: Optional extractor that returns list[str] from text.
        emotion_detector: Optional emotion detector; must provide `.analyze(text)`.
        fact_checker: Optional fact-check service; must provide `.verify_claim(query)`.
    """

    ai_text_detector: Optional[AITextDetector]
    fake_face_detector: Optional[FakeFaceDetector]
    face_detector: Optional[FaceDetector]
    news_detector: Optional[FakeNewsDetector]
    ai_image_detector: Optional[AIImageDetector]
    query_extractor: Optional[SearchQueryExtractor]
    emotion_detector: Optional[EmotionDetector]
    fact_checker: Optional[FactCheckService]

    def __init__(
        self,
        *,
        ai_text_detector: Optional[AITextDetector] = None,
        fake_face_detector: Optional[FakeFaceDetector] = None,
        face_detector: Optional[FaceDetector] = None,
        news_detector: Optional[FakeNewsDetector] = None,
        ai_image_detector: Optional[AIImageDetector] = None,
        query_extractor: Optional[SearchQueryExtractor] = None,
        emotion_detector: Optional[EmotionDetector] = None,
        fact_checker: Optional[FactCheckService] = None,
    ) -> None:
        """Create a FakeNewsManager.

        All parameters are optional; missing detectors/services are simply skipped
        during analysis. Types are intentionally permissive to accommodate a
        variety of detector implementations used in this project.
        """
        self.ai_text_detector = ai_text_detector
        self.fake_face_detector = fake_face_detector
        self.face_detector = face_detector
        self.news_detector = news_detector
        self.ai_image_detector = ai_image_detector
        self.query_extractor = query_extractor
        self.emotion_detector = emotion_detector
        self.fact_checker = fact_checker

    def test(self) -> None:
        """Lightweight method used for quick smoke tests.

        Intended for interactive debugging only; it prints a short marker.
        """
        print("test")

    def _crop_face(self, img: PILImage, face_mp: FaceMainPoints) -> PILImage:
        """Crop a face region from a PIL image using coordinates from
        a `FaceMainPoints` object.

        Args:
            img: PIL.Image instance to crop from.
            face_mp: FaceMainPoints providing `box_start_point` and
                `box_end_point` coordinates as (x, y) tuples.

        Returns:
            A new PIL.Image containing only the cropped face region.
        """
        x1, y1 = face_mp.box_start_point
        x2, y2 = face_mp.box_end_point
        return img.crop((x1, y1, x2, y2))

    def analyze(
        self,
        news: News,
        fakeness_score_threshold: float = 0.6,
    ) -> AggregatedNewsAnalysis:
        """Analyze a `News` item and return an `AggregatedNewsAnalysis`.

        The method coordinates text and image analyzers, runs optional
        fact-checking on extracted queries, and constructs an
        `AggregatedNewsAnalysis` object that summarizes all results.

        Args:
            news: `News` object containing `text` (str) and `images` (list of
                paths or file-like objects) to analyze.
            fakeness_score_threshold: Float threshold in [0, 1] used by the
                aggregated analysis to decide the final `is_fake_final_decision`.

        Returns:
            AggregatedNewsAnalysis populated with detector outputs and a
            computed final decision.
        """
        # Text detectors
        is_ai_text = self.ai_text_detector.detect(news.text) if self.ai_text_detector else None
        is_fake_text = self.news_detector.detect(news.text) if self.news_detector else None

        # Query extraction & emotion
        queries: List[str] = self.query_extractor.extract(news.text) if self.query_extractor else []
        emotion = self.emotion_detector.analyze(news.text) if self.emotion_detector else None

        # Run fact-checking for each extracted query; if no queries, fall back to full text
        fact_check: Optional[List[object]] = None
        if self.fact_checker:
            fact_check = []
            targets = queries if queries else [news.text]
            for q in targets:
                res = self.fact_checker.verify_claim(q)
                if res is not None:
                    fact_check.append(res)

        # Image-level analysis
        images_analysis: List[ImageAnalysis] = []
        for img_in in news.images:
            img = open_image(img_in)

            faces = self.face_detector.detect(img) if self.face_detector else []
            is_ai_image = self.ai_image_detector.detect(img) if self.ai_image_detector else False

            deepfake_faces: List[bool] = []
            if self.fake_face_detector and faces:
                for f in faces:
                    face_img = self._crop_face(img, f)
                    deepfake_faces.append(bool(self.fake_face_detector.detect(face_img)))

            # Ensure image_path is a string as required by schema
            if isinstance(img_in, (str, Path)):
                image_path = str(img_in)
            else:
                image_path = ""

            images_analysis.append(
                ImageAnalysis(
                    image_path=image_path,
                    is_ai_image=is_ai_image,
                    faces=faces,
                    deepfake_faces=deepfake_faces,
                )
            )

        aggregated_news_analysis = AggregatedNewsAnalysis(
            is_fake_final_decision=None,
            analysis_timestamp=datetime.now().isoformat(),
            analysis_id=str(uuid.uuid4()),
            text=news.text,
            is_ai_text=is_ai_text,
            is_fake_text=is_fake_text,
            queries=queries,
            emotion=emotion,
            fact_check=fact_check,
            images=images_analysis,
        )

        # Compute final decision using the AggregatedNewsAnalysis helper
        aggregated_news_analysis.is_fake_final_decision = (
            aggregated_news_analysis.compute_final_decision(fakeness_score_threshold)
        )

        return aggregated_news_analysis




