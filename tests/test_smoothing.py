import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.smoothing import EmotionSmoother
from src.dataset import EMOTION_CLASSES

def test_smoothing_uncertain_threshold():
    smoother = EmotionSmoother(classes=EMOTION_CLASSES, confidence_threshold=0.40)
    # Uniform distribution (1/7 ~= 0.143 < 0.40) -> Must be Uncertain
    uniform_probs = np.full(7, 1.0 / 7.0, dtype=np.float32)
    label, conf, smoothed = smoother.update(uniform_probs)
    assert label == "Uncertain"
    assert conf < 0.40

def test_smoothing_confident_label():
    smoother = EmotionSmoother(classes=EMOTION_CLASSES, confidence_threshold=0.40)
    # Strong happy prediction (index 3)
    happy_probs = np.zeros(7, dtype=np.float32)
    happy_probs[3] = 0.90
    happy_probs[6] = 0.10
    
    label, conf, smoothed = smoother.update(happy_probs)
    assert label == "happy"
    assert conf > 0.40
    assert smoothed[3] > 0.50

def test_smoothing_anti_flicker():
    smoother = EmotionSmoother(classes=EMOTION_CLASSES, window_size=5, ema_alpha=0.5)
    
    # Establish stable happy state for 4 frames
    happy_probs = np.zeros(7, dtype=np.float32)
    happy_probs[3] = 0.85
    for _ in range(4):
        smoother.update(happy_probs)
        
    # A single noisy frame with a flash of sad
    sad_probs = np.zeros(7, dtype=np.float32)
    sad_probs[4] = 0.70
    
    label, conf, smoothed = smoother.update(sad_probs)
    # Because of EMA and 5-frame window, it should remain 'happy' or suppress the single-frame glitch!
    assert label == "happy"
