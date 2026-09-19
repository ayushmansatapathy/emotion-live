import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import FaceDetector
from src.predict import EmotionPredictor
from src.tracker import FaceTracker
from src.dataset import EMOTION_CLASSES
from realtime import RealtimeEmotionApp

def test_pipeline_on_synthetic_stream():
    # Verify that RealtimeEmotionApp initializes and runs on synthetic feed without crashing
    app = RealtimeEmotionApp(
        source='synthetic',
        model_path='models/emotion_model.onnx',
        use_onnx=True,
        conf_threshold=0.40
    )
    # Run headless for 3 seconds
    app.run(max_seconds=3.0, display=False)
    assert app.frame_idx > 0

def test_pipeline_no_face_case():
    detector = FaceDetector(model_path='models/face_detection_yunet_2023mar.onnx')
    predictor = EmotionPredictor(model_path='models/emotion_model.onnx', use_onnx=True)
    tracker = FaceTracker(classes=EMOTION_CLASSES)

    # Empty frame
    blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    boxes = detector.detect(blank_frame)
    assert len(boxes) == 0

    tracked_faces = tracker.update(boxes, None)
    assert len(tracked_faces) == 0
