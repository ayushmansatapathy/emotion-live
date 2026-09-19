import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from typing import List, Tuple, Union, Optional
from src.dataset import EMOTION_CLASSES

class EmotionPredictor:
    """
    High-speed emotion inference engine using ONNX Runtime or PyTorch.
    Handles image preprocessing (CLAHE contrast equalization, resizing, normalization)
    and outputs probabilities across all 7 emotion classes.
    """
    def __init__(self,
                 model_path: str = 'models/emotion_model.onnx',
                 img_size: int = 112,
                 use_onnx: bool = True,
                 use_clahe: bool = True,
                 num_threads: int = 4):
        self.model_path = model_path
        self.img_size = img_size
        self.use_onnx = use_onnx
        self.use_clahe = use_clahe
        self.classes = EMOTION_CLASSES

        # Preprocessing normalization constants (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # CLAHE for low-light enhancement
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
        else:
            from src.model import build_model
            self.model = build_model('mobilenet_v3_small', 7, pretrained=False)
            ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                self.model.load_state_dict(ckpt['model_state'])
            else:
                self.model.load_state_dict(ckpt)
            self.model.eval()

    def preprocess_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a face crop:
        1. Optional CLAHE on luminance (Y channel in YCrCb)
        2. Convert BGR to RGB
        3. Resize to model input size
        4. Normalize
        Returns: (3, H, W) float32 array
        """
        if face_bgr is None or face_bgr.size == 0:
            return np.zeros((3, self.img_size, self.img_size), dtype=np.float32)

        # 1. CLAHE if enabled
        if self.use_clahe:
            ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
            face_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        # 2. Convert to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # 3. Resize
        resized = cv2.resize(face_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # 4. Normalize to float [0, 1] then standardize
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std

        # Transpose to (3, H, W)
        tensor_img = np.transpose(normalized, (2, 0, 1))
        return tensor_img

    def predict_crops(self, crops_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch inference on a list of cropped BGR face images.
        Returns: List of probability vectors (shape [7] for each face).
        """
        if not crops_bgr:
            return []

        batch_tensors = np.stack([self.preprocess_crop(c) for c in crops_bgr], axis=0)

        if self.use_onnx:
            logits = self.session.run([self.output_name], {self.input_name: batch_tensors})[0]
        else:
            with torch.no_grad():
                logits = self.model(torch.from_numpy(batch_tensors)).numpy()

        # Softmax over class logits
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return [probs[i] for i in range(len(crops_bgr))]

    def predict_single(self, crop_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Convenience method for single face crop."""
        probs = self.predict_crops([crop_bgr])[0]
        top_idx = int(np.argmax(probs))
        return self.classes[top_idx], float(probs[top_idx]), probs
