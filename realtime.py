import os
import sys
import time
import threading
import queue
import argparse
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.detector import FaceDetector
from src.predict import EmotionPredictor
from src.tracker import FaceTracker
from src.dataset import EMOTION_CLASSES

# Color palette for 7 emotions (BGR format)
EMOTION_COLORS = {
    'happy': (50, 220, 100),       # Vibrant Green
    'surprise': (255, 210, 0),     # Vibrant Cyan/Yellow
    'neutral': (220, 220, 220),    # Clean White/Silver
    'sad': (220, 140, 60),         # Slate Blue
    'angry': (60, 60, 230),        # Crimson Red
    'fear': (180, 80, 200),        # Purple
    'disgust': (50, 160, 180),     # Olive/Gold
    'Uncertain': (130, 130, 130)   # Neutral Gray
}

class ThreadedCamera:
    """
    Dedicated thread for video capture to decouple capture I/O from inference.
    Always provides the newest frame and drops stale ones to guarantee zero latency lag.
    """
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.is_synthetic = (source == 'synthetic' or source == -1)

        self.cap = None
        if not self.is_synthetic:
            try:
                src_val = int(source) if str(source).isdigit() else source
                self.cap = cv2.VideoCapture(src_val)
                if not self.cap.isOpened():
                    print(f"[Warning] Could not open video source {source}. Falling back to synthetic source.")
                    self.is_synthetic = True
                else:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception as e:
                print(f"[Warning] Camera error: {e}. Using synthetic source.")
                self.is_synthetic = True

        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.frame_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _generate_synthetic_frame(self):
        """Generates dynamic synthetic face frames with moving eyes and mouth for testing."""
        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        t = self.frame_count * 0.05
        # Moving face
        cx = int(self.width // 2 + 40 * np.sin(t * 0.8))
        cy = int(self.height // 2 + 25 * np.cos(t * 0.6))
        
        # Face skin oval
        cv2.ellipse(frame, (cx, cy), (75, 100), 0, 0, 360, (180, 205, 235), -1)
        # Eyes
        cv2.circle(frame, (cx - 28, cy - 25), 9, (60, 50, 40), -1)
        cv2.circle(frame, (cx + 28, cy - 25), 9, (60, 50, 40), -1)
        cv2.circle(frame, (cx - 28, cy - 25), 4, (240, 240, 240), -1)
        cv2.circle(frame, (cx + 28, cy - 25), 4, (240, 240, 240), -1)
        # Nose
        cv2.line(frame, (cx, cy - 5), (cx, cy + 20), (140, 160, 190), 2)
        # Dynamic mouth
        mouth_curve = int(15 + 10 * np.sin(t))
        cv2.ellipse(frame, (cx, cy + 45), (30, mouth_curve), 0, 0, 180, (50, 50, 180), 2)
        return frame

    def _capture_loop(self):
        while self.running:
            if self.is_synthetic:
                frame = self._generate_synthetic_frame()
                self.frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                ret = True
            else:
                ret, frame = self.cap.read()
                if not ret:
                    # If video file ended, loop it
                    if isinstance(self.source, str) and not self.source.isdigit():
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.01)
                    continue

            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

class RealtimeEmotionApp:
    """
    Production-grade Real-Time Facial Emotion Detection Application.
    Features:
    - Thread-safe non-blocking camera capture
    - OpenCV YuNet face detection with margin expansion
    - ONNX Runtime emotion inference with CLAHE enhancement
    - Multi-face tracking with frame-skipping and temporal smoothing
    - Sleek probability bar HUD and latency counter
    - Screenshot (s), Video Recording (r), and Quit (q) shortcuts
    """
    def __init__(self,
                 source=0,
                 model_path='models/emotion_model.onnx',
                 yunet_path='models/face_detection_yunet_2023mar.onnx',
                 use_onnx=True,
                 conf_threshold=0.40,
                 classify_every_n=1):
        self.source = source
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.conf_threshold = conf_threshold
        self.classify_every_n = classify_every_n

        print("[App] Initializing face detector...")
        self.detector = FaceDetector(model_path=yunet_path, conf_threshold=0.55)

        print("[App] Initializing emotion classifier...")
        self.predictor = EmotionPredictor(
            model_path=model_path,
            use_onnx=use_onnx,
            use_clahe=True,
            num_threads=4
        )

        print("[App] Initializing tracker and temporal smoother...")
        self.tracker = FaceTracker(
            classes=EMOTION_CLASSES,
            confidence_threshold=conf_threshold,
            window_size=6,
            ema_alpha=0.65
        )

        self.camera = ThreadedCamera(source=source)
        self.recording = False
        self.video_writer = None
        self.frame_idx = 0

        os.makedirs('screenshots', exist_ok=True)
        os.makedirs('videos', exist_ok=True)

    def draw_ui(self, frame: np.ndarray, tracked_faces, fps: float, latencies: dict) -> np.ndarray:
        """Render high-contrast, polished HUD overlay on video frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Performance HUD (top-left)
        hud_w, hud_h = 240, 78
        cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (25, 25, 25), -1)
        cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (60, 60, 60), 1)

        cv2.putText(out, f"FPS: {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 120), 2)
        cv2.putText(out, f"Latency: {latencies.get('total', 0):.1f}ms", (125, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
        cv2.putText(out, f"Det: {latencies.get('detect', 0):.1f}ms | Cls: {latencies.get('classify', 0):.1f}ms",
                    (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        rec_text = "REC [ON]" if self.recording else "r: Record | s: Snap | q: Exit"
        rec_color = (0, 0, 255) if self.recording else (150, 150, 150)
        cv2.putText(out, rec_text, (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, rec_color, 1)

        # 2. Draw Bounding Boxes and Face HUD
        primary_face = None
        for face in tracked_faces:
            bx, by, bw, bh = face.bbox
            label = face.current_label
            conf = face.current_confidence
            color = EMOTION_COLORS.get(label, (200, 200, 200))

            # Bounding box with corner accents
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, 2)
            
            # Label badge
            badge_text = f"ID #{face.track_id}: {label.upper()} {conf*100:.0f}%" if label != "Uncertain" else f"ID #{face.track_id}: UNCERTAIN"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            badge_y1 = max(0, by - th - 10)
            cv2.rectangle(out, (bx, badge_y1), (bx + tw + 12, by), color, -1)
            cv2.putText(out, badge_text, (bx + 6, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10, 10, 10), 2)

            if primary_face is None or (bw * bh) > (primary_face.bbox[2] * primary_face.bbox[3]):
                primary_face = face

        # 3. Probability Bar Panel (top-right)
        panel_w, panel_h = 220, 190
        px1, py1 = w - panel_w - 15, 10
        cv2.rectangle(out, (px1, py1), (px1 + panel_w, py1 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(out, (px1, py1), (px1 + panel_w, py1 + panel_h), (60, 60, 60), 1)

        title = "PROBABILITIES" if primary_face else "SEARCHING..."
        cv2.putText(out, title, (px1 + 12, py1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

        probs = primary_face.current_probs if primary_face else np.zeros(len(EMOTION_CLASSES))
        bar_y = py1 + 42
        for i, emotion in enumerate(EMOTION_CLASSES):
            p = float(probs[i])
            bar_color = EMOTION_COLORS.get(emotion, (180, 180, 180))

            # Emotion label
            cv2.putText(out, f"{emotion[:4].upper():<4}", (px1 + 12, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)

            # Background bar
            max_bar_w = 110
            cv2.rectangle(out, (px1 + 55, bar_y + 1), (px1 + 55 + max_bar_w, bar_y + 12), (45, 45, 45), -1)
            
            # Active probability bar
            curr_bar_w = int(p * max_bar_w)
            if curr_bar_w > 0:
                cv2.rectangle(out, (px1 + 55, bar_y + 1), (px1 + 55 + curr_bar_w, bar_y + 12), bar_color, -1)

            # Percentage text
            cv2.putText(out, f"{p*100:3.0f}%", (px1 + 175, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            bar_y += 20

        return out

    def run(self, max_seconds: Optional[float] = None, display: bool = True):
        """Main real-time pipeline loop."""
        print("\n" + "="*70)
        print("REAL-TIME FACIAL EMOTION DETECTION RUNNING")
        print("Controls:")
        print("  'q' : Quit application")
        print("  's' : Save screenshot")
        print("  'r' : Toggle video recording")
        print("="*70 + "\n")

        self.camera.start()
        time.sleep(0.5)  # Let camera warm up

        start_app_time = time.time()
        fps_tracker = []
        latencies = {'detect': 0.0, 'classify': 0.0, 'total': 0.0}

        try:
            while True:
                t0 = time.perf_counter()
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.005)
                    continue

                self.frame_idx += 1
                should_classify = (self.frame_idx % self.classify_every_n == 0)

                # 1. Face Detection
                t_det_start = time.perf_counter()
                boxes = self.detector.detect(frame)
                t_det = (time.perf_counter() - t_det_start) * 1000.0

                # 2. Emotion Classification on Crops
                t_cls_start = time.perf_counter()
                probs_list = None
                if should_classify and boxes:
                    crops = []
                    h, w = frame.shape[:2]
                    for bx, by, bw, bh in boxes:
                        c_x1 = max(0, bx)
                        c_y1 = max(0, by)
                        c_x2 = min(w, bx + bw)
                        c_y2 = min(h, by + bh)
                        crop = frame[c_y1:c_y2, c_x1:c_x2]
                        if crop.size > 0:
                            crops.append(crop)
                    if crops:
                        probs_list = self.predictor.predict_crops(crops)

                tracked_faces = self.tracker.update(boxes, probs_list)
                t_cls = (time.perf_counter() - t_cls_start) * 1000.0

                # Calculate FPS & Latencies
                t_total = (time.perf_counter() - t0) * 1000.0
                fps_instant = 1000.0 / max(1.0, t_total)
                fps_tracker.append(fps_instant)
                if len(fps_tracker) > 30:
                    fps_tracker.pop(0)
                avg_fps = float(np.mean(fps_tracker))

                latencies['detect'] = t_det
                latencies['classify'] = t_cls
                latencies['total'] = t_total

                # 3. Draw HUD
                rendered_frame = self.draw_ui(frame, tracked_faces, avg_fps, latencies)

                # Recording handler
                if self.recording:
                    if self.video_writer is None:
                        h, w = rendered_frame.shape[:2]
                        filename = f"videos/recording_{int(time.time())}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(filename, fourcc, 25.0, (w, h))
                    self.video_writer.write(rendered_frame)

                # Display or headless check
                if display:
                    cv2.imshow("Real-Time Facial Emotion Detection", rendered_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[App] Exit key 'q' pressed.")
                        break
                    elif key == ord('s'):
                        shot_path = f"screenshots/screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(shot_path, rendered_frame)
                        print(f"[App] Screenshot saved to: {shot_path}")
                    elif key == ord('r'):
                        self.recording = not self.recording
                        if not self.recording and self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                            print("[App] Recording stopped and saved.")
                        elif self.recording:
                            print("[App] Recording started...")

                if max_seconds and (time.time() - start_app_time) >= max_seconds:
                    print(f"[App] Finished execution duration of {max_seconds} seconds.")
                    break

        finally:
            self.camera.stop()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            print("[App] Application closed cleanly.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='Camera index (0), video file path, or "synthetic"')
    parser.add_argument('--model', default='models/emotion_model.onnx', help='Path to ONNX or PyTorch model')
    parser.add_argument('--pytorch', action='store_true', help='Use PyTorch instead of ONNX Runtime')
    parser.add_argument('--conf-thresh', type=float, default=0.40, help='Confidence threshold for Uncertain state')
    parser.add_argument('--duration', type=float, default=None, help='Auto-stop after N seconds (for testing)')
    parser.add_argument('--no-display', action='store_true', help='Run headless without cv2.imshow')
    args = parser.parse_args()

    app = RealtimeEmotionApp(
        source=args.source,
        model_path=args.model,
        use_onnx=not args.pytorch,
        conf_threshold=args.conf_thresh
    )
    app.run(max_seconds=args.duration, display=not args.no_display)
