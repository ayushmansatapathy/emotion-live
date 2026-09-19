# Real-Time Facial Emotion Detection (emotion-live)

An end-to-end, low-latency Facial Emotion Recognition (FER) system in Python designed for real-time webcam operation on consumer CPU hardware. The pipeline detects faces, tracks identities across frames, extracts aspect-ratio-invariant facial crops, and classifies expressions across seven emotions (*angry, disgust, fear, happy, neutral, sad, surprise*) with real-time probability visualization and temporal smoothing.

---

## Technical Specifications & Performance

All metrics reflect real execution on an Intel 12-thread laptop CPU without GPU acceleration.

| Metric | Target | Measured (ONNX Runtime CPU) | Verification Status |
| :--- | :--- | :--- | :--- |
| **Pipeline Throughput** | $\ge 20$ FPS | **51.6 - 56.4 FPS** | Passed (2.5x - 2.8x target) |
| **End-to-End Latency** | $< 100$ ms | **17.74 - 19.40 ms** (P95: 22.64 ms) | Passed |
| **Classification Latency** | - | **1.94 - 4.98 ms** | 6.6x faster than PyTorch CPU |
| **FER2013 Test Accuracy** | $\ge 65\%$ | **59.85%** (Happy: 82.3% F1, Surprise: 68.6% F1) | Documented below |
| **FER2013 Macro-F1** | $\ge 0.58$ | **0.5533** | Documented below |
| **Label Stability** | No flicker | **0 flicker transitions** across 15 jittered frames | Passed |
| **Automated Test Suite** | 100% pass | **11 / 11 tests passed** (`pytest`) | Passed |

---

## Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
|                            LIVE APPLICATION PIPELINE                              |
|                                                                                   |
|  [ Threaded Video Capture ]                                                       |
|       │  - Dedicated daemon thread with lock-free buffer                           |
|       │  - Drops stale frames; always provides newest frame to eliminate lag      |
|       ▼                                                                           |
|  [ Face Detection: OpenCV YuNet ] (15.37 ms)                                      |
|       │  - ONNX-based deep face detector (cv2.FaceDetectorYN)                     |
|       │  - Centered square crop expansion to eliminate aspect-ratio squashing     |
|       │  - Facial feature centering avoids neck, throat, and clothing noise       |
|       ▼                                                                           |
|  [ Image Preprocessing & Domain Alignment ] (0.01 ms)                             |
|       │  - Grayscale conversion (eliminates color chrominance domain mismatch)    |
|       │  - CLAHE (Contrast Limited Adaptive Histogram Equalization)               |
|       ▼                                                                           |
|  [ Emotion Classification: ONNX Runtime ] (1.94 - 4.98 ms)                        |
|       │  - Optimized INT8/FP32 execution with intra-op CPU thread pools           |
|       │  - Softmax probability distributions across 7 emotion classes             |
|       ▼                                                                           |
|  [ Face Tracker & Temporal Smoother ]                                             |
|       │  - Multi-face tracking via Centroid & Intersection-over-Union (IoU)       |
|       │  - Exponential Moving Average (EMA, alpha=0.75) + 4-frame rolling window  |
|       │  - Confidence thresholding (tau=0.35); shows 'Uncertain' on low entropy   |
|       ▼                                                                           |
|  [ Heads-Up Display (HUD) Rendering ] (0.24 ms)                                   |
|          - Color-coded bounding boxes and persistent track IDs                    |
|          - Live 7-bar horizontal probability distribution panel                   |
|          - Real-time FPS, total latency, and per-stage profiling metrics          |
+-----------------------------------------------------------------------------------+
```

---

## Benchmarks and Latency Profiling

Benchmarked across 100 consecutive frames at 640x480 resolution on CPU:

### Per-Stage Latency Breakdown

| Pipeline Stage | PyTorch CPU (ms) | ONNX Runtime CPU (ms) | Relative Speedup |
| :--- | :--- | :--- | :--- |
| **Capture (Threaded I/O)** | 0.10 ms | 0.09 ms | 1.1x |
| **Face Detection (YuNet)** | 19.12 ms | 17.13 ms | 1.1x |
| **Preprocessing (Square + CLAHE)** | 0.01 ms | 0.01 ms | 1.0x |
| **Emotion Classification** | **13.33 ms** | **1.94 ms** | **6.8x faster** |
| **HUD Drawing & Rendering** | 0.25 ms | 0.23 ms | 1.1x |
| **Total End-to-End Latency** | **32.82 ms** | **19.40 ms** | **1.69x faster** |
| **Pipeline Throughput** | **30.5 FPS** | **51.6 FPS** | **+69.2% Throughput** |

---

## Model Training and Evaluation

### Dataset Preparation
- **Source**: Canonical FER2013 dataset (Train: 28,709, PublicTest: 3,589, PrivateTest: 3,589).
- **Class Distribution & Balancing**: Severe imbalance (e.g., Disgust has only 436 training samples vs. 7,215 for Happy) was addressed using inverse-frequency class weights:
  $$w_c = \left(\frac{N}{K \cdot N_c}\right)^{0.5}$$
- **Data Augmentations**: Random horizontal flip ($p=0.5$), random rotation ($\pm 15^\circ$), color jitter (brightness, contrast, saturation), and random affine translations.

### Evaluation Metrics (PrivateTest Split, 3,589 samples)

| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Happy** | 0.8196 | 0.8271 | **0.8233** | 879 |
| **Surprise** | 0.7246 | 0.6514 | **0.6861** | 416 |
| **Neutral** | 0.5596 | 0.6150 | **0.5860** | 626 |
| **Angry** | 0.4841 | 0.5275 | **0.5049** | 491 |
| **Sad** | 0.4559 | 0.5051 | **0.4792** | 594 |
| **Fear** | 0.4718 | 0.3485 | **0.4009** | 528 |
| **Disgust** | 0.3860 | 0.4000 | **0.3929** | 55 |
| **Macro Average** | **0.5574** | **0.5535** | **0.5533** | **3,589** |
| **Overall Accuracy** | - | - | **59.85%** | **3,589** |

### Confusion Matrix
The normalized confusion matrix is exported to `reports/confusion_matrix.png`:
- High accuracy is achieved on **Happy (83%)**, **Surprise (65%)**, and **Neutral (62%)**.
- Moderate confusion occurs between adjacent negative valence expressions (*Sad* vs. *Neutral*, *Fear* vs. *Angry*), consistent with established FER benchmark findings.

---

## Installation and Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Working webcam (optional; synthetic mode available for headless environments)

### 2. Install Dependencies
```bash
git clone https://github.com/ayushmansatapathy/emotion.git
cd emotion
pip install -r requirements.txt
```

---

## Usage

### Run Live Webcam Application
To start continuous real-time emotion recognition:
```bash
python realtime.py --source 0
```

### Run on Video File or Synthetic Feed
```bash
# Synthetic test stream (dynamic animated face for automated testing):
python realtime.py --source synthetic

# Pre-recorded video file:
python realtime.py --source path/to/video.mp4
```

### Keyboard Shortcuts
During live operation:
- `q`: Exit the application cleanly.
- `s`: Save a timestamped high-resolution snapshot to `screenshots/`.
- `r`: Start or stop recording the live video output to `videos/`.

---

## Project Structure

```
emotion-live/
├── data/                                 # Cached dataset archives (train, val, test npz)
├── models/                               # Neural network models & checkpoints
│   ├── emotion-ferplus-8.onnx            # High-accuracy ONNX model
│   ├── emotion_model.onnx                # Fine-tuned MobileNetV3 ONNX model
│   ├── best_emotion_model.pth            # PyTorch training checkpoint
│   └── face_detection_yunet_2023mar.onnx # OpenCV YuNet face detection weights
├── src/                                  # Core pipeline modules
│   ├── dataset.py                        # Dataset loader, augmentations & class balancing
│   ├── model.py                          # Neural network architecture definitions
│   ├── train.py                          # Training loop with CosineAnnealing & label smoothing
│   ├── evaluate.py                       # Evaluation metrics & confusion matrix generation
│   ├── detector.py                       # Face detector with square crop & landmark centering
│   ├── predict.py                        # ONNX Runtime inference & CLAHE preprocessor
│   ├── tracker.py                        # Centroid & IoU multi-face tracker
│   └── smoothing.py                      # Temporal EMA & rolling window majority voting
├── benchmarks/
│   └── benchmark_pipeline.py             # Latency profiling and FPS benchmarking script
├── tests/                                # Automated PyTest unit and integration tests
│   ├── test_model.py                     # Model architecture & output tensor shape tests
│   ├── test_detector.py                  # Face detector boundary clamping tests
│   ├── test_smoothing.py                 # Anti-flicker & confidence threshold tests
│   ├── test_pipeline.py                  # Full live pipeline integration tests
│   ├── test_benchmark.py                 # FPS and latency assertion tests
│   ├── verify_phase4.py                  # Continuous 65s feed and edge-case verification
│   └── verify_phase5_stability.py        # Noise jitter stability and anti-bias verification
├── reports/                              # Output visual artifacts (confusion matrix, samples)
├── realtime.py                           # Main real-time live application
├── PROGRESS.md                           # Phased iteration log and experimental history
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## Verification and Automated Testing

### Execute Unit and Integration Tests
```bash
python -m pytest -v tests/
```

### Execute Speed & Latency Benchmark
```bash
python benchmarks/benchmark_pipeline.py
```

### Execute Edge-Case & Stability Suite
Verifies continuous 65s execution without memory leaks, handling of pitch-black frames, multi-face tracking, and boundary clipping:
```bash
python tests/verify_phase4.py
```

---

## Domain Considerations and Limitations

1. **FER Benchmark Label Noise**:
   - The original FER2013 dataset contains approximately 10–15% noisy or contested human annotations. Human agreement on FER2013 images is documented at $65\% \pm 5\%$ due to low resolution ($48\times 48$) and subject ambiguity.
2. **Expression vs. Internal Emotional State**:
   - The model detects outward facial muscle contractions and morphological configurations (Facial Action Units). Outward facial expressions do not always correspond directly to internal emotional experiences.
3. **Lighting and Head Pose Variations**:
   - Severe head rotations ($> 45^\circ$ pitch or yaw) reduce face detection confidence. While CLAHE provides robustness against low light, extreme under-exposure requires ambient illumination.

---

## Privacy and Data Security

- **Strictly Local Computation**: All video capture, face detection, tracking, and classification execute entirely on the local device. No video frames, audio, or biometric features are transmitted to remote servers.
- **Explicit Recording Action**: Video frames are processed strictly in volatile memory. No media is written to storage unless the user explicitly initiates a screenshot (`s`) or video recording (`r`).
