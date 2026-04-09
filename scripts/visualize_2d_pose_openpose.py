"""
Visualisasi 2D Pose OpenPose - overlay keypoint di atas frame asli
Sama seperti MediaPipe tapi menggunakan JSON OpenPose (18 keypoint COCO)
"""

import cv2
import json
import numpy as np
from pathlib import Path

RGB_DIR  = Path("/workspace/URFD/rgb")
JSON_DIR = Path("/workspace/awal/json")
OUT_DIR  = Path("/workspace/mediapipe_pipeline/visualisasi")
OUT_DIR.mkdir(exist_ok=True)

# Koneksi skeleton OpenPose COCO 18 keypoint
SKELETON = [
    (0, 1),                   # Nose - Neck
    (1, 2), (2, 3), (3, 4),   # Neck - RShoulder - RElbow - RWrist
    (1, 5), (5, 6), (6, 7),   # Neck - LShoulder - LElbow - LWrist
    (1, 8), (8, 9), (9, 10),  # Neck - RHip - RKnee - RAnkle
    (1, 11),(11,12),(12,13),  # Neck - LHip - LKnee - LAnkle
    (0, 14),(14,16),          # Nose - REye - REar
    (0, 15),(15,17),          # Nose - LEye - LEar
]

COLORS = {
    'head':  (255, 200, 0),
    'upper': (0, 200, 255),
    'lower': (0, 255, 100),
}

def get_color(i, j):
    head_kp  = {0, 14, 15, 16, 17}
    lower_kp = {8, 9, 10, 11, 12, 13}
    if i in head_kp or j in head_kp:   return COLORS['head']
    if i in lower_kp or j in lower_kp: return COLORS['lower']
    return COLORS['upper']

def draw_pose_openpose(frame, keypoints, conf_threshold=0.3):
    """Gambar skeleton OpenPose 18 keypoint di atas frame"""
    kp = np.array(keypoints).reshape(18, 3)

    for i, j in SKELETON:
        if kp[i, 2] > conf_threshold and kp[j, 2] > conf_threshold:
            pt1 = (int(kp[i, 0]), int(kp[i, 1]))
            pt2 = (int(kp[j, 0]), int(kp[j, 1]))
            cv2.line(frame, pt1, pt2, get_color(i, j), 2, cv2.LINE_AA)

    for idx in range(18):
        if kp[idx, 2] > conf_threshold:
            pt = (int(kp[idx, 0]), int(kp[idx, 1]))
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, get_color(idx, idx), 1, cv2.LINE_AA)

    return frame

def create_pose_video(video_name, label):
    frame_dir  = RGB_DIR / video_name / video_name
    json_dir   = JSON_DIR / video_name
    output_mp4 = OUT_DIR / f"2d_pose_openpose_{video_name}.mp4"

    frame_files = sorted(frame_dir.glob("*.png"))
    json_files  = sorted(json_dir.glob("*_keypoints.json"))

    if len(frame_files) == 0:
        print(f"ERROR: tidak ada frame di {frame_dir}")
        return

    sample = cv2.imread(str(frame_files[0]))
    h, w   = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(str(output_mp4), fourcc, 30, (w, h))

    n = min(len(frame_files), len(json_files))
    detected = 0

    for i in range(n):
        frame = cv2.imread(str(frame_files[i]))
        data  = json.loads(json_files[i].read_text())

        if len(data['people']) > 0:
            kp = data['people'][0]['pose_keypoints_2d']
            frame = draw_pose_openpose(frame, kp)
            detected += 1

        color_label = (0, 200, 100) if 'adl' in video_name else (0, 0, 255)
        cv2.putText(frame, f"{label} | Frame {i+1}/{n}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color_label, 2, cv2.LINE_AA)
        cv2.putText(frame, "OpenPose 2D Pose",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detection: {detected}/{i+1}",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    rate = detected / n * 100
    print(f"  Saved: {output_mp4.name} ({rate:.1f}% detected)")

print("Membuat video 2D pose OpenPose...")
print("=" * 50)

for video_name, label in [
    ("adl-01-cam0-rgb", "ADL - Aktivitas Normal"),
    ("fall-06-cam0-rgb", "FALL - Jatuh"),
]:
    print(f"Processing {video_name}...")
    create_pose_video(video_name, label)

print("\nSelesai!")
