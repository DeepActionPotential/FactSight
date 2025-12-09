from dataclasses import dataclass, field
import os


@dataclass
class FlaskConfig:
    """Flask application configuration."""
    SECRET_KEY: str = field(default_factory=lambda: os.environ.get('SECRET_KEY', 'dev-secret-key'))
    UPLOAD_FOLDER: str = 'static/uploads'
    MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50MB


@dataclass
class ModelPaths:
    """Model file paths configuration."""
    # Text detection models
    AI_TEXT_DETECTOR: str = "./models/ai_text_detector.joblib"
    FAKE_NEWS_DETECTOR: str = "models/fake_news_detector.pt"
    VOCAB_PATH: str = "models/word2idx.pt"

    # Image detection models
    EFFICIENTNET_AI_IMAGE: str = "./models/efficientnet_b3_full_ai_image_classifier.pt"
    FACE_DETECTION: str = "models/face_det_10g.onnx"
    MESO4_DF: str = "models/Meso4_DF.h5"
    MESO4_F2F: str = "models/Meso4_F2F.h5"


@dataclass
class ServiceConfig:
    """Service-specific configuration parameters."""
    # Face detection thresholds
    FACE_DETECTION_THRESHOLD: float = 0.5
    FACE_DETECTION_NMS: float = 0.5

    # Analysis thresholds
    FAKENESS_SCORE_THRESHOLD: float = 0.6

    # API Keys
    FACT_API_KEY: str = "your-google-fact-check"

    # Model paths
    models: ModelPaths = field(default_factory=ModelPaths)


@dataclass
class Config:
    """Main application configuration."""
    flask: FlaskConfig = field(default_factory=FlaskConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    def __post_init__(self):
        """Ensure upload folder exists after initialization."""
        if hasattr(self.flask, 'UPLOAD_FOLDER'):
            os.makedirs(self.flask.UPLOAD_FOLDER, exist_ok=True)


# Global configuration instance
general_config = Config()
