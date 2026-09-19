# Real-Time Facial Emotion Detection (emotion-live)

A production-grade, low-latency Facial Emotion Recognition (FER) system in Python designed for real-time webcam operation. The system continuously detects faces, classifies each face across 7 distinct emotions (*angry, disgust, fear, happy, neutral, sad, surprise*), tracks individuals across frames, eliminates flicker via temporal smoothing, and displays an on-screen HUD with real-time probability distribution bars and latency profiling.

---

## Performance Highlights

| Metric | Target | Measured (ONNX Runtime CPU) | Status |
| :--- | :--- | :--- | :--- |
| **Pipeline Throughput** | $\ge 20$ FPS | **56.4 FPS** | **Passed** (2.8x safety margin) |
| **End-to-End Latency** | $< 100$ ms | **17.74 ms** (P95: 19.63 ms) | **Passed** |
| **Classification Latency** | - | **2.02 ms** | 6.6x faster than PyTorch |
| **Test Accuracy (PrivateTest)** | $\ge 65\%$ | **59.85%** | Shortfall documented below |
| **Test Macro-F1** | $\ge 0.58$ | **0.5533** | Shortfall documented below |
| **Label Flicker Transitions** | Minimal | **0 transitions** across 15 jittered frames | **Passed** |
| **PyTest Test Suite** | 100% pass | **11 / 11 passed** | **Passed** |

---

## Architecture Overview

```
+-------------------------------------------------------------------------------+
|                       REAL-TIME APPLICATION PIPELINE                          |
|                                                                               |
|  [ Threaded Video Capture ]                                                   |
|       │                                                                       |
|       ▼                                                                       |
|  [ OpenCV YuNet Face Detection ] (15.37 ms)                                   |
|       │  - Bounding box margin expansion (+18% for chin/forehead context)     |
|       ▼                                                                       |
|  [ CLAHE Preprocessing & Normalization ] (0.01 ms)                            |
|       │  - Y-channel contrast enhancement for invariant low-light handling    |
|       ▼                                                                       |
|  [ ONNX Runtime Classifier ] (2.02 ms)                                        |
|       │  - Fine-tuned MobileNetV3-Small (1.07M parameters)                    |
|       │  - Softmax probability vector (7 emotion classes)                     |
|       ▼                                                                       |
|  [ Centroid & IoU Face Tracker + Temporal Smoother ]                          |
|       │  - Exponential Moving Average (EMA, alpha=0.65) + 6-frame window      |
|       │  - Confidence thresholding: displays 'Uncertain' if confidence < 40%  |
|       ▼                                                                       |
|  [ High-Contrast HUD Renderer ] (0.24 ms)                                     |
|          - Color-coded face boxes and track IDs                               |
|          - 7-bar live probability panel                                       |
|          - Real-time FPS and latency breakdown counter                        |
+-------------------------------------------------------------------------------+
```

---

## Benchmarks & Profiling (CPU Only)

Benchmarked on 100 consecutive frames with 640x480 resolution on an Intel 12-thread laptop CPU without GPU acceleration:

### Per-Stage Timing Breakdown

| Stage | PyTorch CPU (ms) | ONNX Runtime CPU (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **Capture (Threaded)** | 0.10 ms | 0.10 ms | 1.0x |
| **Face Detect (YuNet)** | 19.12 ms | 15.37 ms | 1.2x |
| **Preprocess (CLAHE + Norm)** | 0.01 ms | 0.01 ms | 1.0x |
| **Emotion Classify** | **13.33 ms** | **2.02 ms** | **6.6x faster** |
| **Draw HUD** | 0.25 ms | 0.24 ms | 1.0x |
| **Total End-to-End Latency** | **32.82 ms** | **17.74 ms** | **1.85x faster** |
| **Effective Throughput** | **30.5 FPS** | **56.4 FPS** | **+85% FPS** |

---

## Model Evaluation (FER2013 Test Set)

The emotion classifier is a lightweight MobileNetV3-Small model fine-tuned on the 28,709 images of the canonical FER2013 dataset using cosine annealing and inverse-frequency class weights.

### Per-Class Test Results (PrivateTest, 3,589 samples)

| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Happy** | 0.8196 | 0.8271 | **0.8233** | 879 |
| **Surprise** | 0.7246 | 0.6514 | **0.6861** | 416 |
| **Neutral** | 0.5596 | 0.6150 | **0.5860** | 626 |
| **Angry** | 0.4841 | 0.5275 | **0.5049** | 491 |
| **Sad** | 0.4559 | 0.5051 | **0.4792** | 594 |
| **Fear** | 0.4718 | 0.3485 | **0.4009** | 528 |
| **Disgust** | 0.3860 | 0.4000 | **0.3929** | 55 |
| **Overall / Macro** | **0.5574** | **0.5535** | **0.5533** | **3,589** |

### Confusion Matrix
The normalized confusion matrix is stored in `reports/confusion_matrix.png`:
- High true positive recognition for **Happy (83%)**, **Surprise (65%)**, **Neutral (62%)**, and **Angry (53%)**.
- Moderate confusion occurs between adjacent negative expressions (e.g. *Sad* vs. *Neutral*, *Fear* vs. *Angry*).

---

## Quick Start

### 1. Requirements
Ensure Python 3.10+ is installed.
```bash
pip install -r requirements.txt
```

### 2. Run Live Detection (Webcam)
To start continuous real-time analysis using your physical webcam:
```bash
python realtime.py --source 0
```

### 3. Run on a Video File or Synthetic Test Stream
```bash
# Synthetic test stream (dynamic facial animation):
python realtime.py --source synthetic

# Video file:
python realtime.py --source path/to/video.mp4
```

### 4. Interactive Keyboard Shortcuts
- `q`: Quit the application cleanly
- `s`: Save a timestamped snapshot to `screenshots/`
- `r`: Toggle video recording to `videos/`

---

## Project Structure

```
emotion-live/
├── data/                      # Cached dataset splits (fer2013_train.npz, etc.)
├── models/                    # Exported ONNX model & PyTorch checkpoints
│   ├── emotion_model.onnx
│   ├── best_emotion_model.pth
│   └── face_detection_yunet_2023mar.onnx
├── src/                       # Modular pipeline source code
│   ├── dataset.py             # Data loader, augmentations & class balancing
│   ├── model.py               # MobileNetV3-Small & MobileNetV2 definitions
│   ├── train.py               # Training engine with CosineAnnealing & label smoothing
│   ├── evaluate.py            # Evaluation metrics & confusion matrix generator
│   ├── detector.py            # OpenCV YuNet face detector with margin expansion
│   ├── predict.py             # ONNX Runtime inference & CLAHE preprocessor
│   ├── tracker.py             # Centroid / IoU multi-face tracker
│   └── smoothing.py           # Temporal EMA & rolling window majority vote
├── benchmarks/                # Profiling scripts
│   └── benchmark_pipeline.py  # Automated per-stage FPS/latency benchmark
├── tests/                     # Unit test suite (PyTest)
│   ├── test_model.py          # Model architecture & output shape tests
│   ├── test_detector.py       # Face detector boundary tests
│   ├── test_smoothing.py      # Anti-flicker & confidence threshold tests
│   ├── test_pipeline.py       # Full live pipeline integration tests
│   ├── test_benchmark.py      # FPS threshold assertions
│   ├── verify_phase4.py       # 65s continuous feed & edge case verification
│   └── verify_phase5_stability.py # Jitter and anti-bias verification
├── reports/                   # Visual logs (confusion matrix, sample grids)
├── realtime.py                # Main live application entry point
├── PROGRESS.md                # Phased development and iteration log
├── requirements.txt           # Project dependencies
└── README.md                  # System documentation
```

---

## Running Automated Tests

Run the full PyTest suite:
```bash
python -m pytest -v tests/
```

Run the pipeline benchmark:
```bash
python benchmarks/benchmark_pipeline.py
```

Run edge-case and memory stability verification:
```bash
python tests/verify_phase4.py
```

---

## Limitations & Real-World Considerations

1. **FER2013 Label Noise & Class Imbalance**:
   - The canonical FER2013 dataset exhibits significant label noise (~10–15% contested or ambiguous labels). Human agreement on FER2013 is estimated at 65% ± 5%.
   - Class imbalance is acute: *Disgust* contains only 55 test samples compared to 879 for *Happy*. While smoothed class weighting partially compensates, *Disgust* and *Fear* remain the most challenging categories.
2. **Facial Expression $\neq$ True Emotional State**:
   - The model predicts visible facial expressions (action unit activations), not internal emotional experiences. A person smiling under stress may be classified as "Happy".
3. **Lighting & Pose Sensitivity**:
   - Extreme head pitch/yaw ($>45^\circ$) reduces face detection confidence.
   - Low-light scenarios are actively stabilized through CLAHE (Contrast Limited Adaptive Histogram Equalization), but severe darkness requires supplementary illumination.

---

## Privacy & Ethical Disclosures

- **100% On-Device Processing**: All video capture, face detection, and emotion classification execute purely locally on your CPU. No frames, audio, or metadata are transmitted over the network.
- **Explicit Consent & Recording**: No video feeds or images are persisted to disk unless the user explicitly presses `s` (screenshot) or `r` (recording). All recorded files are saved locally in `screenshots/` and `videos/`.
