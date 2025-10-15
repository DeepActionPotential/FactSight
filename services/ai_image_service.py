import torch
import timm
from torchvision import transforms
from PIL import Image
from typing import Any
from schemas.vision_schemas import AIImageDetector


class ENetAIImageDetector(AIImageDetector):
    """
    EfficientNet-B3 AI Image Detector that classifies whether an image
    is AI-generated or real using a pre-trained PyTorch model.

    Attributes:
        model_path (str): Path to the trained model file (.pt).
        model (Any): Loaded PyTorch model.
        device (str): Device to run inference on ('cuda' or 'cpu').
    """

    def __init__(self, model_path: str = "./models/efficientnet_b3_full_ai_image_classifier.pt"):
        """
        Initialize the ENetAIImageDetector.

        Args:
            model_path (str, optional): Path to the trained EfficientNet model.
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.transform = self._build_transform()

    def _load_model(self) -> Any:
        """Load the trained EfficientNet-B3 model."""
        if self.model_path.endswith(".pt"):
            model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        else:
            model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=1)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _build_transform(self) -> Any:
        """Return preprocessing pipeline for input images."""
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert a PIL Image to a normalized tensor."""
        return self.transform(image).unsqueeze(0).to(self.device)

    def detect(self, image: Image.Image) -> bool:
        """
        Detect whether a given PIL image is AI-generated.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            bool: True if AI-generated, False if real.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image.Image object.")

        img_tensor = self._preprocess_image(image)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            prob = torch.sigmoid(outputs).item()

        is_ai_generated = prob >= 0.001
        return is_ai_generated
