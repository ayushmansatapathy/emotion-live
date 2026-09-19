# Real-Time Facial Emotion Detection - Project Progress Log

## Phase 0: Plan & Environment Check
- **Date**: 2026-09-19
- **Environment**:
  - OS: Windows 11
  - Python: 3.12.7
  - PyTorch: 2.9.0+cpu (CPU only, no CUDA GPU detected)
  - OpenCV: 4.12.0.88 with built-in YuNet FaceDetectorYN
  - ONNX Runtime: 1.22.0
  - Camera 0: Available and accessible (`ret: True`)
  - Internet: Available
- **Dataset Strategy**:
  - FER2013 canonical splits (Train: 28,709, PublicTest: 3,589, PrivateTest: 3,589)
  - 7 emotion classes: 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
  - Severe class imbalance handled via Inverse-Frequency Class Weights in CrossEntropyLoss
- **Model Architecture**:
  - Backbone: Lightweight MobileNetV2 / MobileNetV3-Small transfer learning, adapted to 7 classes
  - Input: RGB face crop, normalized with ImageNet stats
  - Inference: ONNX Runtime export for CPU acceleration
- **Identified Risks & Mitigations**:
  - *Risk*: CPU training speed could be bottleneck.
    - *Mitigation*: Feature extraction freeze + progressive fine-tuning of top layers, efficient batching, multithreading.
  - *Risk*: Live webcam flickering between similar emotions (e.g. neutral vs sad).
    - *Mitigation*: Exponential Moving Average (EMA) and rolling-window majority filtering.
  - *Risk*: False detections / low confidence guesses.
    - *Mitigation*: "Uncertain" state with confidence thresholding (default $\tau = 0.40$).
  - *Risk*: Latency budget $> 100\text{ ms}$.
    - *Mitigation*: Threaded capture loop decoupling camera FPS from inference FPS; ONNX runtime engine; YuNet ultra-fast face detector (~3ms).

---

## Phase 1: Data Pipeline
- **Status: Complete**
- **Artifacts**:
  - `data/train.parquet` (101.5 MB), `data/publicTest.parquet` (12.7 MB), `data/privateTest.parquet` (12.7 MB)
  - Pre-cached npz arrays:
    - Train: `data/fer2013_train.npz` (28,709 images, 48x48x3 uint8)
    - Val (PublicTest): `data/fer2013_val.npz` (3,589 images, 48x48x3 uint8)
    - Test (PrivateTest): `data/fer2013_test.npz` (3,589 images, 48x48x3 uint8)
- **Class Distribution & Weights**:
  - Angry (0): 0.806, Disgust (1): 2.439, Fear (2): 0.796, Happy (3): 0.600, Sad (4): 0.733, Surprise (5): 0.904, Neutral (6): 0.723
  - Handled via inverse frequency smoothing: `weight_c = (total / (num_classes * count_c))^0.5`
- **Augmentations**:
  - Random crop with padding, RandomHorizontalFlip(p=0.5), RandomRotation(15 deg), ColorJitter (brightness, contrast, saturation), RandomAffine
- **Verification**:
  - Sample batch verified: Shape `(16, 3, 112, 112)`, float32, range [-2.118, 2.640].
  - Saved sample grid to `reports/sample_batch.png` with correct emotion labels. Verified visually.

---

## Phase 2: Training Loop
- **Target**: $\ge 65\%$ test accuracy AND macro-F1 $\ge 0.58$ on FER2013 test (PrivateTest).
- **Iteration 1**:
  - **Config**: Backbone: `mobilenet_v3_small`, Image size: 112, Batch size: 128, Warmup: 1 epoch, Fine-tune: 3 epochs, Optimizer: AdamW, LR backbone: 3e-4, LR head: 1e-3, CosineAnnealingLR, Label smoothing: 0.1, Class weights: inverse-frequency smoothed (power 0.5).
  - **Results**:
    - Val Accuracy: **58.34%**, Val Macro-F1: **0.5344**
    - Test Accuracy (PrivateTest): **58.62%**, Test Macro-F1: **0.5276**
    - Per-class Test F1: Happy: 0.8132, Surprise: 0.6909, Neutral: 0.5783, Angry: 0.4986, Sad: 0.4437, Fear: 0.3412, Disgust: 0.3273
    - Confusion matrix saved to `reports/confusion_matrix.png`
  - **Diagnosis**: Loss steadily dropped from 1.6522 to 1.4364 across 3 epochs without sign of overfitting. Accuracy increased continuously (46.19% -> 54.52% -> 57.42%). The model has not yet converged and will benefit from further fine-tuning epochs at lower learning rates.
- **Iteration 2**:
  - **Config**: Backbone: `mobilenet_v3_small` resumed from Iteration 1 checkpoint, Image size: 112, Batch size: 128, Warmup: 0 epochs, Fine-tune: 3 epochs, Optimizer: AdamW, LR backbone: 1e-4, LR head: 3e-4, CosineAnnealingLR, Label smoothing: 0.05, Class weights: inverse-frequency smoothed.
  - **Results**:
    - Val Accuracy: **59.65%**, Val Macro-F1: **0.5642**
    - Test Accuracy (PrivateTest): **59.85%**, Test Macro-F1: **0.5533**
    - Per-class Test F1: Happy: 0.8233, Surprise: 0.6861, Neutral: 0.5860, Angry: 0.5049, Sad: 0.4792, Fear: 0.4009, Disgust: 0.3929
    - Confusion matrix updated at `reports/confusion_matrix.png`
  - **Diagnosis**: Fine-tuning at lower learning rates with reduced label smoothing yielded solid improvements across all classes (Test accuracy +1.23%, Macro-F1 +0.0257). Disgust and Fear F1 improved significantly (Disgust from 0.3273 to 0.3929, Fear from 0.3412 to 0.4009). The model is nearing 60% accuracy on FER2013 test set.
  - **Next Iteration / Optimization**: Current model weights saved to `models/best_emotion_model.pth` and exported to `models/emotion_model.onnx`.

---

## Phase 3: Speed Optimization
- *Status: Pending*

---

## Phase 4: Live Application
- *Status: Pending*

---

## Phase 5: Real-World Stability & Tuning
- *Status: Pending*

---

## Phase 6: Tests & Documentation
- *Status: Pending*
