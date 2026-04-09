# Synthetic IMU Generation for Fall Detection

Adapting the IMUTube framework to generate synthetic accelerometer (IMU) data from video
for fall detection, without requiring physical sensors.

**Dataset:** UR Fall Detection Dataset (URFD) — Kwolek & Kepski, 2014  
**Target Publication:** Frontiers Journal

---

## Overview

This research generates synthetic IMU data by analyzing hip movement in CCTV video frames.
The core idea is that when a person falls, their hip position changes drastically in a short time —
this rapid change in position can be measured as acceleration, acting as a "virtual sensor."

The pipeline is evaluated against real accelerometer data from the URFD dataset,
using two 2D pose estimation approaches: OpenPose (baseline) and MediaPipe (primary).

---

## Pipeline

```
Video PNG frames
        |
MediaPipe 2D Pose Estimation
(17 keypoints per frame, Detectron COCO format)
        |
Mid-hip pixel position extraction
(average of LHip index 11 and RHip index 12)
        |
Savitzky-Golay Smoothing
(window=15, polyorder=3)
        |
Finite Difference -> Pixel Acceleration
a[t] = (p[t+1] - 2*p[t] + p[t-1]) / dt^2
        |
Synthetic IMU Signal
(magnitude = sqrt(ax^2 + ay^2))
        |
Extract 10 Statistical Features
        |
SVM / Random Forest Classifier
        |
Fall or ADL
```

Note: 3D pose reconstruction (VideoPose3D) and camera calibration (GeoCalib) are used
for visualization and analysis only. The synthetic IMU is computed directly from 2D pixel
positions because URFD uses a fixed camera.

---

## Repository Structure

```
synthetic-imu-fall-detection/
|
|-- awal/                           # OpenPose pipeline (baseline)
|   |-- extract_poses.py
|   |-- openpose_to_npz.py
|   |-- pnp_global_tracking.py
|   |-- generate_3d_poses.py
|   |-- generate_imu.py
|   `-- compare_imu.py
|
|-- mediapipe_pipeline/             # MediaPipe pipeline (primary)
|   `-- extract_poses_mediapipe.py
|
|-- scripts/                        # Utility and conversion scripts
|   |-- prepare_urfd_dataset.py
|   |-- prepare_urfd_dataset_HumanEVA.py
|   |-- converter.py
|   |-- converter2.py
|   |-- check_npz.py
|   |-- process_openpose.py
|   `-- analizedz1.py
|
|-- imutube/                        # IMUTube framework adaptation
|   |-- cli.py
|   |-- config.py
|   |-- orchestrator.py
|   |-- stages/
|   `-- utils/
|
|-- .gitignore
`-- README.md
```

---

## Dataset

URFD (UR Fall Detection Dataset) — Kwolek & Kepski, 2014

| Property        | Value                          |
|-----------------|--------------------------------|
| Total videos    | 70 (cam0) + 70 (cam1)          |
| Fall sequences  | 30 per camera                  |
| ADL sequences   | 40 per camera                  |
| Resolution      | 640x480 px (RGB PNG frames)    |
| Frame rate      | 30 FPS                         |
| Ground truth    | Accelerometer SVM per frame    |
| Sensor position | Waist / hip                    |

Important: The URFD .mp4 files are side-by-side composites (depth map left + RGB right)
at 640x240px. Always use the pre-extracted PNG frames at 640x480px as input.

---

## Environment Setup

Three separate virtual environments are required to avoid library conflicts.

**1. MediaPipe environment**
```bash
python -m venv mediapipe_env
source mediapipe_env/bin/activate
pip install mediapipe==0.10.9 opencv-python
```

**2. GeoCalib environment**
```bash
python -m venv geocalib_env
source geocalib_env/bin/activate
# Follow installation at: https://github.com/cvg/GeoCalib
```

**3. Main environment**
```bash
pip install numpy==1.24.0 opencv-python==4.7.0.72 scipy torch==2.1
```

---

## Running the Pipeline

**Step 1 — 2D Pose Extraction (MediaPipe)**
```bash
source mediapipe_pipeline/mediapipe_env/bin/activate
python mediapipe_pipeline/extract_poses_mediapipe.py
```

**Step 2 — Convert JSON to NPZ**
```bash
python scripts/converter2.py
```

**Step 3 — 3D Pose Reconstruction (visualization only)**
```bash
cd VideoPose3D
python run.py -d custom -k urfd_mediapipe \
  --evaluate pretrained_h36m_detectron_coco.bin \
  --render --viz-export output.json -arc 3,3,3,3,3
```

**Step 4 — Camera Calibration**
```bash
source awal/geocalib_env/bin/activate
python scripts/run_geocalib_batch.py
```

**Step 5 — Generate Synthetic IMU**
```bash
python scripts/generate_imu_pixel.py
# Output shape: (N_frames, 3) -> [accel_x, accel_y, magnitude]
```

**Step 6 — Fall Detection Classification**
```bash
python scripts/evaluate_fall_detection.py
# SVM and Random Forest with Leave-One-Out Cross Validation
```

---

## Keypoint Mapping

**MediaPipe (33 keypoints) to Detectron COCO (17 keypoints)**
```python
MEDIAPIPE_TO_DETECTRON = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# Index 23 = LHip, Index 24 = RHip  <-- most critical for fall detection
```

**OpenPose COCO (18 keypoints) to Detectron COCO (17 keypoints)**
```python
OPENPOSE_TO_DETECTRON = [0, 16, 15, -1, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
# -1 = LEar not available in OpenPose, filled with zeros
```

---

## Results (Current)

### 2D Pose Detection Rate

| Approach  | cam0  | cam1  | Videos processed       |
|-----------|-------|-------|------------------------|
| OpenPose  | 84.2% | ~50%  | 70 videos (cam0 only)  |
| MediaPipe | 76.0% | 88.6% | 101 videos (cam0+cam1) |

### Synthetic IMU Signal Quality (FALL/ADL ratio)

| Approach  | Mean ratio | Peak ratio | Real IMU target |
|-----------|------------|------------|-----------------|
| OpenPose  | 1.27x      | 1.27x      | 3.19x           |
| MediaPipe | 3.34x      | 3.18x      | 3.19x           |

MediaPipe peak ratio (3.18x) is nearly identical to the real IMU sensor (3.19x).

### Fall Detection Accuracy (Leave-One-Out Cross Validation)

| IMU Source             | SVM Accuracy | Random Forest Accuracy |
|------------------------|:------------:|:----------------------:|
| Real IMU (target)      | 95.7%        | 97.1%                  |
| OpenPose synthetic     | 80.0%        | 82.9%                  |
| MediaPipe synthetic    | 95.7%        | 90.0%                  |

MediaPipe-based synthetic IMU achieves the same SVM accuracy as the real physical sensor.

---

## OpenPose vs MediaPipe Comparison

| Aspect                  | OpenPose COCO         | MediaPipe             |
|-------------------------|-----------------------|-----------------------|
| Developer               | Carnegie Mellon Univ. | Google                |
| Model size              | 200 MB                | Built-in              |
| Detection rate (cam0)   | 84.2%                 | 76.0%                 |
| Detection rate (cam1)   | ~50%                  | 88.6%                 |
| Hip confidence (fall)   | 0.076 - 0.130         | 0.999 - 1.000         |
| Python compatibility    | 2 / 3                 | 3.10                  |
| Setup complexity        | High (build source)   | Low (pip install)     |
| FALL/ADL ratio          | 1.27x                 | 3.18x                 |
| SVM accuracy            | 80.0%                 | 95.7%                 |

---

## Challenges and Solutions

| Challenge                                       | Solution                                        |
|-------------------------------------------------|-------------------------------------------------|
| URFD .mp4 files are side-by-side composites     | Use pre-extracted PNG frames from /rgb/ folders |
| OpenPose detects 2 people (dual-image layout)   | Use --number_people_max 1 flag                  |
| DeepCalib requires Python 2.7 + TensorFlow 1.4  | Replaced with GeoCalib (ECCV 2024)              |
| Library conflicts between tools                 | Three separate virtual environments             |
| VideoPose3D outputs root-relative coordinates   | Use 2D pixel positions directly for IMU         |
| PnP global tracking unstable                    | Pixel-based approach used instead               |

---

## References

1. Kwon et al. — IMUTube: Virtual IMU from Video
2. Pavllo et al. — VideoPose3D, CVPR 2019
3. Lugaresi et al. — MediaPipe Pose, Google 2020
4. Cao et al. — OpenPose, IEEE TPAMI 2021
5. Veicht et al. — GeoCalib, ECCV 2024 (https://github.com/cvg/GeoCalib)
6. Kwolek & Kepski — URFD Dataset, 2014
