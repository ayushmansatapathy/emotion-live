import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

class EmotionSmoother:
    """
    Temporal smoother for emotion probabilities and labels.
    Prevents jitter and flickering between adjacent frames.
    
    Features:
    - Exponential Moving Average (EMA) of class probability vectors
    - Rolling window majority voting & probability averaging
    - Confidence thresholding: returns 'Uncertain' if top probability < threshold
    """
    def __init__(self,
                 classes: List[str],
                 window_size: int = 5,
                 ema_alpha: float = 0.65,
                 confidence_threshold: float = 0.40):
        self.classes = classes
        self.num_classes = len(classes)
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold

        self.history = deque(maxlen=window_size)
        self.ema_probs = None
        self.current_label = "Uncertain"
        self.current_confidence = 0.0

    def reset(self):
        """Reset internal history when a face is lost or new face appears."""
        self.history.clear()
        self.ema_probs = None
        self.current_label = "Uncertain"
        self.current_confidence = 0.0

    def update(self, raw_probs: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Update smoother with new raw probabilities (shape: [num_classes]).
        Returns:
            label: str ('Uncertain' or emotion name)
            confidence: float (0.0 to 1.0)
            smoothed_probs: np.ndarray (smoothed probability distribution)
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float32)
        assert raw_probs.shape[0] == self.num_classes, f"Expected shape ({self.num_classes},), got {raw_probs.shape}"

        # 1. Update Exponential Moving Average (EMA)
        if self.ema_probs is None:
            self.ema_probs = raw_probs.copy()
        else:
            self.ema_probs = self.ema_alpha * raw_probs + (1.0 - self.ema_alpha) * self.ema_probs

        # 2. Add to rolling history window
        self.history.append(self.ema_probs)

        # 3. Compute window-averaged probabilities
        smoothed_probs = np.mean(self.history, axis=0)
        # Re-normalize to sum to 1.0
        smoothed_probs = smoothed_probs / (np.sum(smoothed_probs) + 1e-7)

        top_idx = int(np.argmax(smoothed_probs))
        top_conf = float(smoothed_probs[top_idx])

        # 4. Confidence thresholding
        if top_conf < self.confidence_threshold:
            self.current_label = "Uncertain"
        else:
            self.current_label = self.classes[top_idx]

        self.current_confidence = top_conf
        return self.current_label, self.current_confidence, smoothed_probs

    def get_probabilities_dict(self, probs: np.ndarray) -> Dict[str, float]:
        """Convert probability vector to a dictionary of emotion: probability."""
        return {cls_name: float(p) for cls_name, p in zip(self.classes, probs)}
