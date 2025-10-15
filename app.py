from flask import Flask, render_template, request, jsonify, redirect, url_for
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Optional, Tuple, Any
import os
import uuid
import base64
import io
from datetime import datetime
from PIL import Image

# Import configuration
from config import general_config

# -----------------------------
# Configuration and Setup
# -----------------------------

# Initialize Flask app with configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = general_config.flask.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = general_config.flask.MAX_CONTENT_LENGTH

# -----------------------------
# Data models (matching your schema)
# -----------------------------

from schemas.fact_search_schemas import FactCheckEntry, FactCheckResult
from schemas.text_schemas import EmotionResult
from schemas.vision_schemas import FaceMainPoints
from schemas.fake_manager_schemas import ImageAnalysis, AggregatedNewsAnalysis
from core.fake_manager import FakeNewsManager
from schemas.fake_manager_schemas import News

from services.ai_text_service import NBAITextDetector
from services.fake_text_news_service import FakeTextNewsDetector
from services.search_quries_service import TransformersSearchQueryExtractor
from services.text_emotion_service import TransformersEmotionDetector
from services.fact_search_service import FactCheckService
from services.ai_image_service import ENetAIImageDetector
from services.face_detection_service import SCRFDFaceDetector
from services.deepfake_service import Meso4FakeFaceDetector
from models.models import LSTMClassifier


# In-memory store: analysis_id -> JSON dict
STORE: dict[str, dict] = {}

# -----------------------------
# Helpers
# -----------------------------
def save_image(image_data: str, filename: str | None = None) -> Optional[str]:
    try:
        if image_data.startswith('data:image'):
            header, b64 = image_data.split(',', 1)
            mime = header.split(';')[0].split(':')[1]  # e.g. image/png
            ext = {'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp'}.get(mime, 'jpg')
        else:
            b64 = image_data
            ext = 'jpg'

        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        img.verify()  # validate

        img = Image.open(io.BytesIO(raw))
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        if not filename:
            filename = f"{uuid.uuid4().hex}.{ext}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(path, quality=92)
        return path
    except Exception as e:
        print(f"Error saving image: {e}")
        import traceback
        traceback.print_exc()
        return None

def initialize_services():
    """Initialize and configure all ML services."""
    return FakeNewsManager(
        ai_text_detector=NBAITextDetector(general_config.service.models.AI_TEXT_DETECTOR),
        news_detector=FakeTextNewsDetector(
            model_path=general_config.service.models.FAKE_NEWS_DETECTOR,
            vocab_path=general_config.service.models.VOCAB_PATH,
        ),
        query_extractor=TransformersSearchQueryExtractor(),
        emotion_detector=TransformersEmotionDetector(),
        fact_checker=FactCheckService(api_key=general_config.service.FACT_API_KEY),
        ai_image_detector=ENetAIImageDetector(general_config.service.models.EFFICIENTNET_AI_IMAGE),
        face_detector=SCRFDFaceDetector(
            model_path=general_config.service.models.FACE_DETECTION,
            threshold_probability=general_config.service.FACE_DETECTION_THRESHOLD,
            nms=general_config.service.FACE_DETECTION_NMS,
        ),
        fake_face_detector=Meso4FakeFaceDetector(
            df_model_path=general_config.service.models.MESO4_DF,
            f2f_model_path=general_config.service.models.MESO4_F2F,
        ),
    )


# Initialize services
fake_news_manager = initialize_services()

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'images' not in data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400

        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'success': False, 'error': 'News text is required'}), 400

        images_in = data.get('images') or []
        if not images_in:
            return jsonify({'success': False, 'error': 'At least one image is required'}), 400

        saved_fs_paths_disk = []
        saved_fs_paths_web = []
        for img in images_in:
            path = save_image(img.get('data', ''))
            if path:
                saved_fs_paths_disk.append(path)
                saved_fs_paths_web.append('/' + path.replace('\\', '/'))

        news = News(text=text, images=saved_fs_paths_disk)
        analysis = fake_news_manager.analyze(news, fakeness_score_threshold=general_config.service.FAKENESS_SCORE_THRESHOLD)
        analysis_json = analysis.to_json()

        # Overwrite image paths in the JSON to web paths for frontend rendering
        for i, img_entry in enumerate(analysis_json.get("images", [])):
            if i < len(saved_fs_paths_web):
                img_entry["image_path"] = saved_fs_paths_web[i]

        STORE[analysis_json["analysis_id"]] = analysis_json

        return jsonify({'success': True, 'analysis_id': analysis_json["analysis_id"]})
    except Exception as e:
        print("Analysis error:", e)
        import traceback
        traceback.print_exc()

        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analysis/<analysis_id>')
def analysis_page(analysis_id):
    analysis = STORE.get(analysis_id)
    if not analysis:
        return redirect(url_for('index'))
    return render_template('analysis.html', analysis=analysis)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("Starting News Analyzer Server...")
    print("Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
