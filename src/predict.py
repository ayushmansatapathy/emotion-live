import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from typing import List, Tuple, Union, Optional
from src.dataset import EMOTION_CLASSES

class EmotionPredictor:
    """
    High-performance, domain-aligned Emotion Predictor.
    Supports both FER+ ONNX (64x64 Grayscale) and custom MobileNetV3 (112x112).
    Ensures input is strictly preprocessed in the grayscale domain matching FER datasets,
    with CLAHE contrast enhancement for real-world webcam lighting variations.
    """
    def __init__(self,
                 model_path: str = 'models/emotion-ferplus-8.onnx',
                 use_onnx: bool = True,
                 use_clahe: bool = True,
                 num_threads: int = 4):
        # Fallback to emotion_model.onnx if ferplus not present
        if not os.path.exists(model_path):
            if os.path.exists('models/emotion_model.onnx'):
                model_path = 'models/emotion_model.onnx'
            elif os.path.exists('models/best_emotion_model.pth'):
                model_path = 'models/best_emotion_model.pth'
                use_onnx = False

        self.model_path = model_path
        self.use_onnx = use_onnx
        self.use_clahe = use_clahe
        self.classes = EMOTION_CLASSES  # ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if use_onnx:
            assert os.path.exists(model_path), f"ONNX model not found at {model_path}"
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape

            # Detect whether model expects 64x64 Grayscale (FER+) or 112x112 RGB (MobileNet)
            self.is_ferplus = (len(self.input_shape) == 4 and self.input_shape[1] == 1 and self.input_shape[2] == 64)
            # FER+ 8-class indices: 0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt
            # Mapped to target: ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.ferplus_map = [4, 5, 6, 1, 3, 2, 0]
        else:
            self.is_ferplus = False
            from src.model import build_model
            self.model = build_model('mobilenet_v3_small', 7, pretrained=False)
            ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                self.model.load_state_dict(ckpt['model_state'])
            else:
                self.model.load_state_dict(ckpt)
            self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop into model-ready tensor format.
        Preserves square geometry, converts to Grayscale with CLAHE.
        """
        if face_bgr is None or face_bgr.size == 0:
            target_dim = 64 if self.is_ferplus else 112
            channels = 1 if self.is_ferplus else 3
            return np.zeros((channels, target_dim, target_dim), dtype=np.float32)

        # 1. Convert to Grayscale
        if len(face_bgr.shape) == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr

        # 2. Apply CLAHE contrast enhancement
        if self.use_clahe:
            gray = self.clahe.apply(gray)

        # 3. Square crop guarantee
        h, w = gray.shape
        side = min(h, w)
        cx, cy = w // 2, h // 2
        square = gray[max(0, cy - side//2):min(h, cy + side//2), max(0, cx - side//2):min(w, cx + side//2)]

        if self.is_ferplus:
            # FER+ expects (1, 64, 64) float32
            resized = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            return resized.reshape(1, 64, 64)
        else:
            # MobileNet expects (3, 112, 112) normalized float32
            resized_gray = cv2.resize(square, (112, 112), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)
            normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
            return np.transpose(normalized, (2, 0, 1))

    def predict_crops(self, crops_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch inference on face crops.
        Returns: List of 7-class probability vectors [angry, disgust, fear, happy, sad, surprise, neutral].
        """
        if not crops_bgr:
            return []

        batch_tensors = np.stack([self.preprocess_crop(c) for c in crops_bgr], axis=0)

        if self.use_onnx:
            raw_out = self.session.run([self.output_name], {self.input_name: batch_tensors})[0]
            if self.is_ferplus:
                results = []
                for logits in raw_out:
                    exp = np.exp(logits - np.max(logits))
                    probs_8 = exp / np.sum(exp)
                    # Map to 7 classes
                    probs_7 = np.array([probs_8[idx] for idx in self.ferplus_map], dtype=np.float32)
                    probs_7 /= np.sum(probs_7)
                    results.append(probs_7)
                return results
            else:
                logits = raw_out
        else:
            with torch.no_grad():
                logits = self.model(torch.from_numpy(batch_tensors)).numpy()

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return [probs[i] for i in range(len(crops_bgr))]

    def predict_single(self, crop_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        probs = self.predict_crops([crop_bgr])[0]
        top_idx = int(np.argmax(probs))
        return self.classes[top_idx], float(probs[top_idx]), probs
