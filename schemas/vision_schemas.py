from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
from scrfd import Face
import numpy as np
import os
from pathlib import Path


class AIImageDetector(ABC):
    """
    Abstract base class for AI image detectors.
    """

    @abstractmethod
    def detect(self, image: Image.Image) -> bool:
        """
        Detect whether the given image is AI-generated.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            bool: True if AI-generated, False if real.
        """
        pass


@dataclass
class FaceMainPoints:
    """
    The main points of the face.

    Attributes:
    box_start_point (Tuple[int, int]): The start point of the bounding box.
    box_end_point (Tuple[int, int]): The end point of the bounding box.
    box_probabilty_score (float): The probability score of the bounding box.
    left_eye (Tuple[int, int], optional): The left eye coordinates. Defaults to (0, 0).
    right_eye (Tuple[int, int], optional): The right eye coordinates. Defaults to (0, 0).
    nose (Tuple[int, int], optional): The nose coordinates. Defaults to (0, 0).
    left_mouth (Tuple[int, int], optional): The left mouth coordinates. Defaults to (0, 0).
    right_mouth (Tuple[int, int], optional): The right mouth coordinates. Defaults to (0, 0).
    """
    box_start_point: Tuple[int, int]
    box_end_point: Tuple[int, int]
    box_probabilty_score: float
    left_eye: Tuple[int, int] = (0, 0)
    right_eye: Tuple[int, int] = (0, 0)
    nose: Tuple[int, int] = (0, 0)
    left_mouth: Tuple[int, int] = (0, 0)
    right_mouth: Tuple[int, int] = (0, 0)


class FaceDetector(ABC):
    """
    The face detector interface.
    """

    @abstractmethod
    def detect(self, image_path: str) -> List[FaceMainPoints]:
        """
        Detect the faces in an image.

        Args:
        image_path (str): The path to the image.

        Returns:
        A list of FaceMainPoints objects, one for each face in the image.
        """
        pass

    @abstractmethod
    def convert_face_to_face_main_points(self, face: Face) -> FaceMainPoints:
        """
        Convert a Face object to a FaceMainPoints object.

        Args:
        face (Face): The face to convert.

        Returns:
        A FaceMainPoints object.
        """
        pass


class FakeFaceDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image, threshold: float = 0.5) -> bool:
        """
        Base detector method.
        Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses should implement this method.")