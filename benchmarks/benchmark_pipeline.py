import os
import sys
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import FaceDetector
from src.predict import EmotionPredictor
from src.tracker import FaceTracker
from src.dataset import EMOTION_CLASSES

def generate_synthetic_faces_frame(w=640, h=480, num_faces=1):
    """Generate a realistic test frame with synthetic oval face structures."""
    frame = np.full((h, w, 3), 40, dtype=np.uint8)  # dark background
    
    positions = [(w // 2, h // 2)] if num_faces == 1 else [(w // 3, h // 2), (2 * w // 3, h // 2)]
    
    for cx, cy in positions:
        # Face skin tone oval
        cv2.ellipse(frame, (cx, cy), (65, 85), 0, 0, 360, (180, 205, 235), -1)
        # Eyes
        cv2.circle(frame, (cx - 25, cy - 20), 8, (60, 50, 40), -1)
        cv2.circle(frame, (cx + 25, cy - 20), 8, (60, 50, 40), -1)
        # Nose
        cv2.line(frame, (cx, cy - 5), (cx, cy + 15), (140, 160, 190), 2)
        # Smile / mouth
        cv2.ellipse(frame, (cx, cy + 35), (25, 12), 0, 0, 180, (50, 50, 180), 2)
        
    return frame

def draw_hud(frame: np.ndarray, faces, fps: float, latencies: dict):
    """Render bounding boxes, probability bars, and FPS metrics overlay."""
    out = frame.copy()
    h, w = out.shape[:2]
    
    # 1. Performance HUD box
    hud_h, hud_w = 90, 240
    cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
    cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (0, 255, 180), 1)
    
    cv2.putText(out, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(out, f"Total: {latencies.get('total', 0):.1f}ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(out, f"Det: {latencies.get('detect', 0):.1f}ms | Cls: {latencies.get('classify', 0):.1f}ms", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # 2. Draw faces
    for face in faces:
        x, y, fw, fh = face.bbox
        label = face.current_label
        conf = face.current_confidence
        
        color = (0, 255, 0) if label == 'happy' else (0, 200, 255) if label == 'surprise' else (200, 200, 200)
        cv2.rectangle(out, (x, y), (x + fw, y + fh), color, 2)
        text = f"{label.upper()} {conf*100:.0f}%" if label != "Uncertain" else "UNCERTAIN"
        cv2.putText(out, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return out

def run_pipeline_benchmark(num_frames: int = 150, use_onnx: bool = True):
    print("\n" + "="*70)
    backend_name = "ONNX Runtime" if use_onnx else "PyTorch CPU"
    print(f"RUNNING SPEED & LATENCY BENCHMARK ({backend_name})")
    print(f"Benchmarking {num_frames} frames on CPU...")
    print("="*70)
    
    detector = FaceDetector(conf_threshold=0.5)
    
    model_path = 'models/emotion_model.onnx' if use_onnx else 'models/best_emotion_model.pth'
    predictor = EmotionPredictor(model_path=model_path, use_onnx=use_onnx, num_threads=4)
    tracker = FaceTracker(classes=EMOTION_CLASSES)
    
    # Generate stand-in frames
    test_frame = generate_synthetic_faces_frame(640, 480, num_faces=1)
    
    timings = {
        'capture': [],
        'detect': [],
        'preprocess': [],
        'classify': [],
        'draw': [],
        'total': []
    }
    
    # Warmup
    for _ in range(10):
        _ = detector.detect(test_frame)
        _ = predictor.predict_crops([test_frame[100:200, 100:200]])
        
    for frame_idx in range(num_frames):
        t_start = time.perf_counter()
        
        # 1. Capture stand-in
        t0 = time.perf_counter()
        frame = test_frame.copy()
        t_cap = (time.perf_counter() - t0) * 1000.0
        
        # 2. Detect
        t0 = time.perf_counter()
        boxes = detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000.0
        
        # 3. Preprocess & Crop
        t0 = time.perf_counter()
        crops = []
        h, w = frame.shape[:2]
        for bx, by, bw, bh in boxes:
            cx1 = max(0, bx)
            cy1 = max(0, by)
            cx2 = min(w, bx + bw)
            cy2 = min(h, by + bh)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crops.append(crop)
        t_prep = (time.perf_counter() - t0) * 1000.0
        
        # 4. Classify & Track
        t0 = time.perf_counter()
        if crops:
            probs_list = predictor.predict_crops(crops)
        else:
            # Fallback if synthetic face wasn't matched
            probs_list = predictor.predict_crops([frame[100:250, 200:350]])
            boxes = [np.array([200, 100, 150, 150])]
            
        tracked_faces = tracker.update(boxes, probs_list)
        t_cls = (time.perf_counter() - t0) * 1000.0
        
        # 5. Draw
        t0 = time.perf_counter()
        fps_est = 1000.0 / max(1.0, (time.perf_counter() - t_start) * 1000.0)
        out_frame = draw_hud(frame, tracked_faces, fps_est, {'total': 0, 'detect': t_det, 'classify': t_cls})
        t_draw = (time.perf_counter() - t0) * 1000.0
        
        t_total = (time.perf_counter() - t_start) * 1000.0
        
        timings['capture'].append(t_cap)
        timings['detect'].append(t_det)
        timings['preprocess'].append(t_prep)
        timings['classify'].append(t_cls)
        timings['draw'].append(t_draw)
        timings['total'].append(t_total)
        
    avg_timings = {k: np.mean(v) for k, v in timings.items()}
    p95_timings = {k: np.percentile(v, 95) for k, v in timings.items()}
    fps = 1000.0 / avg_timings['total']
    
    print("\nPER-STAGE TIMING BREAKDOWN:")
    print(f"{'Stage':<18} | {'Avg Latency (ms)':<18} | {'P95 Latency (ms)':<18}")
    print("-" * 60)
    for stage in ['capture', 'detect', 'preprocess', 'classify', 'draw', 'total']:
        print(f"{stage.capitalize():<18} | {avg_timings[stage]:<18.2f} | {p95_timings[stage]:<18.2f}")
    print("-" * 60)
    print(f"OVERALL THROUGHPUT: {fps:.1f} FPS (Target: >= 20 FPS)")
    print(f"END-TO-END LATENCY: {avg_timings['total']:.2f} ms (Target: < 100 ms)")
    
    passed_fps = fps >= 20.0
    passed_latency = avg_timings['total'] < 100.0
    print(f"Target FPS Met:     {passed_fps}")
    print(f"Target Latency Met: {passed_latency}")
    print("=" * 70 + "\n")
    
    return {
        'backend': backend_name,
        'fps': fps,
        'avg_latency': avg_timings['total'],
        'breakdown': avg_timings,
        'p95': p95_timings,
        'passed': passed_fps and passed_latency
    }

if __name__ == '__main__':
    run_pipeline_benchmark(num_frames=100, use_onnx=True)
