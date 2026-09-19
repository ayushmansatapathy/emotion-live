import numpy as np
from typing import List, Dict, Tuple, Optional
from src.smoothing import EmotionSmoother

def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculate IoU between boxA and boxB [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

class TrackedFace:
    """Represents an active face track with temporal smoothing."""
    def __init__(self, track_id: int, bbox: np.ndarray, classes: List[str], window_size: int = 5, ema_alpha: float = 0.65, confidence_threshold: float = 0.40):
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=np.int32)  # [x, y, w, h]
        self.smoother = EmotionSmoother(classes, window_size=window_size, ema_alpha=ema_alpha, confidence_threshold=confidence_threshold)
        self.lost_frames = 0
        self.current_label = "Uncertain"
        self.current_confidence = 0.0
        self.current_probs = np.zeros(len(classes), dtype=np.float32)

    def update_detection(self, bbox: np.ndarray, raw_probs: Optional[np.ndarray] = None):
        """Update track with new bounding box and optional classification probabilities."""
        self.bbox = np.array(bbox, dtype=np.int32)
        self.lost_frames = 0
        if raw_probs is not None:
            self.current_label, self.current_confidence, self.current_probs = self.smoother.update(raw_probs)

    def mark_missed(self):
        """Increment lost frames count when face is not detected in current frame."""
        self.lost_frames += 1

class FaceTracker:
    """
    Lightweight Centroid & IoU Face Tracker.
    Maintains face identity across frames, enables temporal smoothing per person,
    and supports frame-skipping by propagating prior bounding boxes.
    """
    def __init__(self, classes: List[str], iou_threshold: float = 0.35, max_lost: int = 8,
                 window_size: int = 5, ema_alpha: float = 0.65, confidence_threshold: float = 0.40):
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold

        self.next_id = 0
        self.tracks: Dict[int, TrackedFace] = {}

    def update(self, detected_boxes: List[np.ndarray], raw_probs_list: Optional[List[np.ndarray]] = None) -> List[TrackedFace]:
        """
        Associate detections with existing tracks using IoU matching.
        Returns list of active TrackedFace objects.
        """
        num_dets = len(detected_boxes)
        track_ids = list(self.tracks.keys())
        matched_dets = set()
        matched_tracks = set()

        if track_ids and num_dets > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(track_ids), num_dets), dtype=np.float32)
            for t_idx, tid in enumerate(track_ids):
                for d_idx, dbox in enumerate(detected_boxes):
                    iou_matrix[t_idx, d_idx] = calculate_iou(self.tracks[tid].bbox, dbox)

            # Greedy matching by highest IoU
            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[t_idx]

                # Match found
                probs = raw_probs_list[d_idx] if raw_probs_list is not None else None
                self.tracks[tid].update_detection(detected_boxes[d_idx], probs)
                matched_tracks.add(tid)
                matched_dets.add(d_idx)

                # Invalidate row and column
                iou_matrix[t_idx, :] = -1
                iou_matrix[:, d_idx] = -1

        # Create new tracks for unmatched detections
        for d_idx in range(num_dets):
            if d_idx not in matched_dets:
                new_track = TrackedFace(
                    track_id=self.next_id,
                    bbox=detected_boxes[d_idx],
                    classes=self.classes,
                    window_size=self.window_size,
                    ema_alpha=self.ema_alpha,
                    confidence_threshold=self.confidence_threshold
                )
                probs = raw_probs_list[d_idx] if raw_probs_list is not None else None
                if probs is not None:
                    new_track.update_detection(detected_boxes[d_idx], probs)
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Handle unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].mark_missed()

        # Remove dead tracks
        dead_ids = [tid for tid, track in self.tracks.items() if track.lost_frames > self.max_lost]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return only currently active (not dead) tracks
        return [t for t in self.tracks.values() if t.lost_frames <= self.max_lost]
