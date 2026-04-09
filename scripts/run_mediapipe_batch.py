"""
Script batch MediaPipe untuk semua video URFD
Input : PNG frames dari /workspace/URFD/rgb/{nama}/{nama}/
Output: JSON keypoints ke /workspace/mediapipe_pipeline/json/{nama}/

Format JSON sama dengan OpenPose agar pipeline selanjutnya konsisten.
MediaPipe menghasilkan 33 keypoint, kita mapping ke 17 keypoint Detectron COCO
untuk kompatibilitas dengan VideoPose3D.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================
RGB_DIR    = Path("/workspace/URFD/rgb")
OUTPUT_DIR = Path("/workspace/mediapipe_pipeline/json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Confidence threshold (sama dengan OpenPose pipeline)
CONF_THRESHOLD = 0.3

# Mapping MediaPipe index -> Detectron COCO 17 keypoints
# Detectron: Nose(0), LEye(1), REye(2), LEar(3), REar(4),
#            LShoulder(5), RShoulder(6), LElbow(7), RElbow(8),
#            LWrist(9), RWrist(10), LHip(11), RHip(12),
#            LKnee(13), RKnee(14), LAnkle(15), RAnkle(16)
MEDIAPIPE_TO_DETECTRON = [
    0,   # Detectron 0  (Nose)       <- MediaPipe 0  (nose)
    2,   # Detectron 1  (LEye)       <- MediaPipe 2  (left_eye)
    5,   # Detectron 2  (REye)       <- MediaPipe 5  (right_eye)
    7,   # Detectron 3  (LEar)       <- MediaPipe 7  (left_ear)
    8,   # Detectron 4  (REar)       <- MediaPipe 8  (right_ear)
    11,  # Detectron 5  (LShoulder)  <- MediaPipe 11 (left_shoulder)
    12,  # Detectron 6  (RShoulder)  <- MediaPipe 12 (right_shoulder)
    13,  # Detectron 7  (LElbow)     <- MediaPipe 13 (left_elbow)
    14,  # Detectron 8  (RElbow)     <- MediaPipe 14 (right_elbow)
    15,  # Detectron 9  (LWrist)     <- MediaPipe 15 (left_wrist)
    16,  # Detectron 10 (RWrist)     <- MediaPipe 16 (right_wrist)
    23,  # Detectron 11 (LHip)       <- MediaPipe 23 (left_hip)
    24,  # Detectron 12 (RHip)       <- MediaPipe 24 (right_hip)
    25,  # Detectron 13 (LKnee)      <- MediaPipe 25 (left_knee)
    26,  # Detectron 14 (RKnee)      <- MediaPipe 26 (right_knee)
    27,  # Detectron 15 (LAnkle)     <- MediaPipe 27 (left_ankle)
    28,  # Detectron 16 (RAnkle)     <- MediaPipe 28 (right_ankle)
]

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose

# ============================================================
# FUNGSI PROSES SATU VIDEO
# ============================================================
def process_video(frame_dir, output_dir):
    """
    Proses semua frame PNG dari satu video.
    Berbeda dari OpenPose yang memproses video/batch,
    MediaPipe dijalankan frame per frame dari PNG.
    """
    frame_files = sorted(frame_dir.glob("*.png"))
    if len(frame_files) == 0:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    detected   = 0
    total      = len(frame_files)

    with mp_pose.Pose(
        static_image_mode=True,       # mode gambar statis (per frame PNG)
        model_complexity=2,           # 0=cepat, 1=medium, 2=akurat
        min_detection_confidence=CONF_THRESHOLD,
        min_tracking_confidence=CONF_THRESHOLD
    ) as pose:

        for frame_idx, frame_file in enumerate(frame_files):
            # Load gambar
            img_bgr = cv2.imread(str(frame_file))
            if img_bgr is None:
                # Simpan frame kosong jika gambar tidak bisa dibaca
                _save_empty(output_dir, frame_file.stem)
                continue

            h, w = img_bgr.shape[:2]

            # MediaPipe butuh RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                detected += 1
                keypoints = []

                for mp_idx in MEDIAPIPE_TO_DETECTRON:
                    lm   = results.pose_landmarks.landmark[mp_idx]
                    # Konversi dari normalized (0-1) ke pixel
                    x    = lm.x * w
                    y    = lm.y * h
                    conf = lm.visibility  # MediaPipe visibility sebagai confidence
                    keypoints.extend([x, y, conf])

                json_data = {
                    "version": 1.3,
                    "people": [{
                        "pose_keypoints_2d": keypoints
                    }]
                }
            else:
                # Tidak ada deteksi
                json_data = {
                    "version": 1.3,
                    "people": []
                }

            # Simpan JSON dengan nama file yang konsisten
            output_file = output_dir / f"{frame_file.stem}_keypoints.json"
            with open(output_file, 'w') as f:
                json.dump(json_data, f)

    return detected / total * 100 if total > 0 else 0

def _save_empty(output_dir, stem):
    output_file = output_dir / f"{stem}_keypoints.json"
    with open(output_file, 'w') as f:
        json.dump({"version": 1.3, "people": []}, f)

# ============================================================
# PROSES SEMUA VIDEO
# ============================================================
video_folders = sorted(RGB_DIR.iterdir())
total         = len(video_folders)

print(f"Total video: {total}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print("=" * 50)

for count, video_folder in enumerate(video_folders, start=1):
    video_name = video_folder.name
    frame_dir  = video_folder / video_name  # nested folder
    output_dir = OUTPUT_DIR / video_name

    # Skip jika sudah diproses
    existing = list(output_dir.glob("*_keypoints.json"))
    if output_dir.exists() and len(existing) > 0:
        print(f"[{count}/{total}] Skip {video_name} (sudah ada {len(existing)} file)")
        continue

    if not frame_dir.exists():
        print(f"[{count}/{total}] SKIP {video_name} - folder tidak ditemukan")
        continue

    print(f"[{count}/{total}] Processing {video_name}...")
    rate = process_video(frame_dir, output_dir)
    print(f"  Detection rate: {rate:.1f}%")

print("")
print("=" * 50)
print("SELESAI!")
