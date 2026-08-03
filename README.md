# 🏏 Cricket Batting Shot Classification

A computer vision pipeline that detects a batsman in video, estimates their 3D body pose, and classifies the batting shot they played — built to compare two batsmen side-by-side using biomechanical pose data instead of raw pixels.

## Overview

Instead of classifying shots directly from video frames, this pipeline strips away everything but the batsman's *movement*: it detects the striker, extracts and enhances the relevant frames, lifts them into 3D skeletal poses, and feeds a temporal sequence of that skeleton into the classifier. The result is a shot label — **drive**, **defence**, **flick**, or **pull** — for each batsman, enabling apples-to-apples technique comparison regardless of camera angle, jersey color, or background clutter.

## Pipeline

```
Video (batsman A) ─┐
                    ├─► YOLO Detection ─► Crop & Preprocess ─► METRAbs 3D Pose ─► Frame Selection ─► TCN Classifier ─► Shot Label
Video (batsman B) ─┘
```

1. **Batsman Detection** — A custom-trained YOLO model locates the striker in each frame and crops to the bounding box, discarding background.
2. **Image Preprocessing** — Frames are resized to a fixed height and sharpened with a high-boost filter (`original + amplification × (original − blurred)`) to enhance edge detail before pose estimation.
3. **3D Pose Estimation** — [METRAbs](https://github.com/isarandi/metrabs) lifts each cropped frame into a 30-joint 3D skeleton (`smpl+head_30`), filtered by detection confidence.
4. **Frame Selection** — An interval-based RMSE sampling method picks the 20 most motion-representative frames from the full sequence (interpolating if fewer than 20 frames are available), so the classifier sees the shape of the movement, not redundant near-duplicates.
5. **Shot Classification** — The 20-frame pose sequence is normalized, flattened to `(20, 90)`, and passed through a trained **Temporal Convolutional Network (TCN)** to predict the shot type with a confidence score.

## Models Used

| Component | Model | Purpose |
|---|---|---|
| Detection | YOLO (custom-trained) | Locate the striker batsman in-frame |
| Pose Estimation | METRAbs | Monocular 3D human pose estimation (30 joints) |
| Classification | TCN (Keras) | Temporal shot classification from pose sequences |

All models are cached via a singleton `ModelManager` so repeated pipeline runs (e.g. comparing multiple video pairs) don't reload weights from disk each time.

## Shot Classes

- Drive
- Defence
- Flick
- Pull

## Usage

```python
results = run_complete_pipeline(
    video1_path="path/to/batsman_1.mp4",
    video2_path="path/to/batsman_2.mp4",
    yolo_model_path="path/to/batsman_detect.pt",
    metrabs_model_path="path/to/metrabs-model",
    shot_model_path="path/to/tcn_best_model_20frames.keras",
    output_dir="output"
)
```

Each video is processed independently through the full pipeline, and both predicted shots (with confidence scores) are returned for direct comparison.

## Tech Stack

- **TensorFlow / Keras** — TCN shot classifier
- **TensorFlow Hub** — METRAbs 3D pose model
- **Ultralytics YOLO** — batsman detection
- **OpenCV** — video I/O and frame preprocessing
- **NumPy / Pandas** — pose data handling

## Notes

- This project builds on 3D human pose estimation work using a MetRAbs-based pipeline, adapted here specifically for cricket batting technique analysis.
- The frame selection strategy (interval + RMSE) is designed to make the classifier robust to variable video length and frame rate, always feeding it a fixed 20-frame motion signature.

---

📓 Original notebook: [Kaggle — shot_Classification](https://www.kaggle.com/code/sourav32/shot-classification)
