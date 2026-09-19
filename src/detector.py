import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    """
    High-performance Face Detector wrapper supporting OpenCV YuNet.
    Includes bounding box margin expansion and image boundary clamping.
    """
    def __init__(self,
                 model_path: str = 'models/face_detection_yunet_2023mar.onnx',
                 conf_threshold: float = 0.6,
                 nms_threshold: float = 0.3,
                 top_k: int = 5000,
                 margin_pct: float = 0.18):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.margin_pct = margin_pct

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found at {model_path}. Please download it.")

        # Default initialization size
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
        Returns: List of bounding boxes as (x, y, w, h) with margin applied.
        """
        h, w = frame.shape[:2]
        if (w, h) != self.current_size:
            self.detector.setInputSize((w, h))
            self.current_size = (w, h)

        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        expanded_boxes = []
        for face in faces:
            # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            x, y, box_w, box_h = face[:4]
            score = face[14]
            if score < self.conf_threshold:
                continue

            # Expand bounding box by margin_pct
            margin_x = box_w * self.margin_pct
            margin_y = box_h * self.margin_pct

            x1 = max(0, int(x - margin_x))
            y1 = max(0, int(y - margin_y))
            x2 = min(w, int(x + box_w + margin_x))
            y2 = min(h, int(y + box_h + margin_y))

            final_w = max(1, x2 - x1)
            final_h = max(1, y2 - y1)
            expanded_boxes.append((x1, y1, final_w, final_h))

        return expanded_boxes
