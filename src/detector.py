import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    """
    High-performance Face Detector wrapper supporting OpenCV YuNet.
    Ensures square facial bounding boxes centered on facial features
    to eliminate aspect-ratio squashing and neck/chest contamination.
    """
    def __init__(self,
                 model_path: str = 'models/face_detection_yunet_2023mar.onnx',
                 conf_threshold: float = 0.55,
                 nms_threshold: float = 0.3,
                 top_k: int = 5000,
                 scale_factor: float = 1.15,
                 margin_pct: Optional[float] = None):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.scale_factor = (1.0 + margin_pct) if margin_pct is not None else scale_factor

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found at {model_path}. Please download it.")

        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            conf_threshold,
            nms_threshold,
            top_k
        )
        self.current_size = (320, 320)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.
        Returns: List of square bounding boxes as (x, y, w, h) centered on face.
        """
        h, w = frame.shape[:2]
        if (w, h) != self.current_size:
            self.detector.setInputSize((w, h))
            self.current_size = (w, h)

        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        square_boxes = []
        for face in faces:
            x, y, box_w, box_h = face[:4]
            score = face[14]
            if score < self.conf_threshold:
                continue

            # Compute facial center: slightly higher than center of YuNet box to focus on eyes/brows/mouth
            cx = int(x + box_w * 0.5)
            cy = int(y + box_h * 0.48)
            
            # Use maximum dimension to form a true square
            side = int(max(box_w, box_h) * self.scale_factor)
            half = side // 2

            # Square coordinates clamped to frame boundaries
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            final_w = max(1, x2 - x1)
            final_h = max(1, y2 - y1)
            square_boxes.append((x1, y1, final_w, final_h))

        return square_boxes
