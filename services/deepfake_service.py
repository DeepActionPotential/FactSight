from PIL import Image

from schemas.vision_schemas import FakeFaceDetector

import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


class Meso4FakeFaceDetector(FakeFaceDetector):
    """
    Deepfake face detector using Meso4 models trained on DF and F2F datasets.
    Combines both models' predictions for a more robust detection.
    """

    def __init__(self,
                 df_model_path: str = "models/Meso4_DF.h5",
                 f2f_model_path: str = "models/Meso4_F2F.h5",
                 input_shape=(256, 256, 3)):
        """
        Initialize and load both pretrained models.

        Args:
            df_model_path (str): Path to Meso4 model trained on DeepFake dataset.
            f2f_model_path (str): Path to Meso4 model trained on Face2Face dataset.
            input_shape (tuple): Expected input shape for the models.
        """
        self.df_model_path = df_model_path
        self.f2f_model_path = f2f_model_path
        self.input_shape = input_shape

        # Build both models
        self.model_df = self._build_meso4()
        self.model_f2f = self._build_meso4()

        # Load pretrained weights
        self.model_df.load_weights(self.df_model_path)
        self.model_f2f.load_weights(self.f2f_model_path)

    # ------------------ Internal Model Builder ------------------

    def _build_meso4(self):
        """Build the Meso4 CNN model architecture."""
        model = Sequential()

        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(8, (5, 5), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(4, 4), strides=4))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # ------------------ Image Preprocessing ------------------

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for prediction.

        Args:
            image (PIL.Image): Input face image.

        Returns:
            np.ndarray: Preprocessed image ready for model input.
        """
        img = image.convert("RGB")
        img = img.resize((256, 256))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # ------------------ Detection ------------------

    def detect(self, image: Image.Image, threshold: float = 0.5, verbose=True) -> bool:
        """
        Detect whether an input face image is fake or real.

        Args:
            image (PIL.Image): Input image of a face.
            threshold (float): Decision threshold (default 0.5).

        Returns:
            bool: True if fake, False if real.
        """
        img = self._preprocess_image(image)

        # Get individual predictions
        pred_df = float(self.model_df.predict(img, verbose=0)[0][0])
        pred_f2f = float(self.model_f2f.predict(img, verbose=0)[0][0])

        # Combine predictions (average)
        combined_pred = (pred_df + pred_f2f) / 2.0
        
        if verbose:
            # Print detailed info
            print(f"🔍 Meso4_DF Prediction: {pred_df:.4f}")
            print(f"🔍 Meso4_F2F Prediction: {pred_f2f:.4f}")
            print(f"⚖️ Combined Score: {combined_pred:.4f}")

        # Determine if fake
        is_fake = combined_pred >= threshold

        if is_fake:
            print("🔴 Deepfake detected.")
        else:
            print("🟢 Real face detected.")

        return is_fake
