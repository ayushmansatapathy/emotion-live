import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import FaceDetector

def test_detector_empty_dark_frame():
    detector = FaceDetector(model_path='models/face_detection_yunet_2023mar.onnx')
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector.detect(dark_frame)
    assert isinstance(boxes, list)
    assert len(boxes) == 0

def test_detector_margin_bounds():
    detector = FaceDetector(model_path='models/face_detection_yunet_2023mar.onnx', margin_pct=0.20)
    # Test on synthetic frame
    h, w = 480, 640
    frame = np.full((h, w, 3), 50, dtype=np.uint8)
    # Face skin oval
    cv2.ellipse(frame, (w//2, h//2), (70, 90), 0, 0, 360, (180, 205, 235), -1)
    boxes = detector.detect(frame)
    for bx, by, bw, bh in boxes:
        assert bx >= 0 and by >= 0
        assert bx + bw <= w
        assert by + bh <= h
