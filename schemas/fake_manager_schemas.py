from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Optional, Any, Tuple, Union
from pathlib import Path
import numpy as np
from PIL.Image import Image
from schemas.vision_schemas import FaceMainPoints
from schemas.text_schemas import EmotionResult
from schemas.fact_search_schemas import FactCheckResult



@dataclass
class News:
    text: str
    images: List[Union[str, Path, Image, np.ndarray]]



@dataclass
class IsFakeNewsResult:

    is_fake_text: bool
    is_ai_text: bool
    is_deepfake_faces: list[bool]




@dataclass
class ImageAnalysis:
    image_path: str
    is_ai_image: bool
    faces: List[FaceMainPoints]
    deepfake_faces: List[bool]



@dataclass
class AggregatedNewsAnalysis:
    is_fake_final_decision: bool
    text: str
    is_ai_text: Optional[bool]
    is_fake_text: Optional[bool]
    queries: List[str]
    emotion: Optional["EmotionResult"]
    fact_check: Optional[List["FactCheckResult"]]
    images: List["ImageAnalysis"]
    analysis_timestamp: str
    analysis_id: str

    # ---------- existing ----------
    def to_json(self) -> dict:
        def _convert(obj: Any) -> Any:
            if is_dataclass(obj):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, list):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        return _convert(self)

    # ---------- NEW: scoring helpers ----------
    @staticmethod
    def _score_fact_verdict(verdict: str) -> float:
        """Map summary_verdict to [0,1] fake-likelihood."""
        if not verdict:
            return 0.0
        v = verdict.strip().lower()
        table = {
            "false": 1.0, "pants on fire": 1.0, "fake": 1.0,
            "likely false": 0.9, "incorrect": 0.9, "misleading": 0.85, "deceptive": 0.85,
            "unsubstantiated": 0.6, "missing context": 0.5,
            "mixed": 0.5, "partly false": 0.7, "partly true": 0.3,
            "mostly true": 0.1, "true": 0.0, "correct": 0.0
        }
        # fuzzy match by containment if exact key not present
        for k, s in table.items():
            if k in v:
                return s
        return 0.0

    @staticmethod
    def _score_fact_checks(fact_checks: Optional[List["FactCheckResult"]]) -> float:
        if not fact_checks:
            return 0.0
        # take the strongest (max) verdict score across checks
        return max((AggregatedNewsAnalysis._score_fact_verdict(fc.summary_verdict) for fc in fact_checks if fc and fc.summary_verdict), default=0.0)

    @staticmethod
    def _score_text(is_fake_text: Optional[bool], is_ai_text: Optional[bool]) -> float:
        score = 0.0
        if is_fake_text is True:
            score += 0.7
        elif is_fake_text is False:
            score += 0.0  # explicit non-fake does not add risk

        # AI text alone is NOT proof of fake; small nudge only
        if is_ai_text is True:
            score += 0.1
        elif is_ai_text is False:
            score += 0.0
        return min(score, 1.0)

    @staticmethod
    def _score_images(images: List["ImageAnalysis"]) -> float:
        if not images:
            return 0.0
        # per-image risk
        per_image = []
        for img in images:
            s = 0.0
            if getattr(img, "is_ai_image", False):
                s = max(s, 0.5)
            # deepfake face => strong signal
            if any(bool(x) for x in getattr(img, "deepfake_faces", []) or []):
                s = max(s, 0.8)
            per_image.append(s)

        if not per_image:
            return 0.0

        # Combine by "noisy OR": 1 - Π(1 - s_i)
        from math import prod
        return 1.0 - prod(1.0 - s for s in per_image)

    @staticmethod
    def _score_emotion(emotion: Optional["EmotionResult"]) -> float:
        # Small bump only if highly sensational and confident
        if not emotion:
            return 0.0
        dom = (emotion.dominant_emotion or "").strip().lower()
        conf = float(getattr(emotion, "confidence", 0.0) or 0.0)
        sensational = {"anger", "fear", "surprise", "disgust"}
        if dom in sensational and conf >= 0.75:
            return 0.1
        return 0.0

    def _active_weights(self, w_fact=0.5, w_text=0.3, w_img=0.15, w_emote=0.05) -> Tuple[float, float, float, float]:
        """Re-normalize weights if any component is missing."""
        have_fact = bool(self.fact_check)
        have_text = (self.is_fake_text is not None) or (self.is_ai_text is not None)
        have_img  = bool(self.images)
        have_emo  = bool(self.emotion)

        parts = [
            (w_fact if have_fact else 0.0),
            (w_text if have_text else 0.0),
            (w_img  if have_img  else 0.0),
            (w_emote if have_emo else 0.0),
        ]
        total = sum(parts) or 1.0
        return tuple(p / total for p in parts)  # normalized

    def compute_final_score(self) -> tuple[float, dict]:
        """Return (score, breakdown) in [0,1]."""
        wf, wt, wi, we = self._active_weights()

        s_fact = self._score_fact_checks(self.fact_check)
        s_text = self._score_text(self.is_fake_text, self.is_ai_text)
        s_img  = self._score_images(self.images)
        s_emo  = self._score_emotion(self.emotion)

        score = wf * s_fact + wt * s_text + wi * s_img + we * s_emo
        breakdown = {
            "weights": {"fact": wf, "text": wt, "images": wi, "emotion": we},
            "components": {"fact": s_fact, "text": s_text, "images": s_img, "emotion": s_emo},
            "total": score
        }
        return score, breakdown

    def compute_final_decision(self, threshold: float = 0.60) -> bool:
        """Set and return final decision based on weighted score."""
        score, _ = self.compute_final_score()
        self.is_fake_final_decision = bool(score >= threshold)
        return self.is_fake_final_decision
