"""
Konversi JSON MediaPipe -> NPZ untuk input VideoPose3D
Lebih sederhana dari OpenPose karena MediaPipe sudah output 17 keypoint
format Detectron langsung.
"""

import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ============================================================
# KONFIGURASI
# ============================================================
JSON_DIR   = Path("/workspace/mediapipe_pipeline/json")
OUTPUT_NPZ = Path("/workspace/mediapipe_pipeline/data_2d_custom_urfd_mediapipe.npz")

FRAME_W        = 640
FRAME_H        = 480
CONF_THRESHOLD = 0.3

KEYPOINTS_SYMMETRY = [
    [1, 3, 5, 7, 9, 11, 13, 15],
    [2, 4, 6, 8, 10, 12, 14, 16]
]

# ============================================================
# FUNGSI LOAD JSON SATU VIDEO
# ============================================================
def load_json_folder(folder):
    """
    Baca semua JSON MediaPipe dari satu folder.
    MediaPipe sudah output 17 keypoint Detectron langsung,
    tidak perlu mapping seperti OpenPose.
    Return (N_frames, 17, 2)
    """
    json_files = sorted(folder.glob("*_keypoints.json"))
    frames_xy  = []

    for jf in json_files:
        data = json.loads(jf.read_text())

        if len(data['people']) > 0:
            kp = np.array(data['people'][0]['pose_keypoints_2d']).reshape(17, 3)

            # Terapkan confidence threshold
            xy      = kp[:, :2].copy()
            conf    = kp[:, 2].copy()
            xy[conf < CONF_THRESHOLD] = np.nan

        else:
            xy = np.full((17, 2), np.nan)

        frames_xy.append(xy)

    frames_xy = np.array(frames_xy)  # (N, 17, 2)

    # Interpolasi NaN per joint
    N = len(frames_xy)
    for j in range(17):
        for c in range(2):
            col  = frames_xy[:, j, c]
            mask = ~np.isnan(col)

            if mask.sum() >= 2:
                idx_valid  = np.where(mask)[0]
                col_interp = interp1d(
                    idx_valid, col[mask],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(col[mask][0], col[mask][-1])
                )
                frames_xy[:, j, c] = col_interp(np.arange(N))
            elif mask.sum() == 1:
                frames_xy[:, j, c] = col[mask][0]
            else:
                frames_xy[:, j, c] = 0.0

    return frames_xy  # (N, 17, 2)

# ============================================================
# PROSES SEMUA VIDEO
# ============================================================
positions_2d   = {}
video_metadata = {}

# Hanya proses folder yang mengandung -rgb
video_folders = sorted(f for f in JSON_DIR.iterdir() if '-rgb' in f.name)
total = len(video_folders)

print(f"Total video: {total}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print("=" * 50)

for count, folder in enumerate(video_folders, start=1):
    video_name = folder.name

    json_files = list(folder.glob("*_keypoints.json"))
    if len(json_files) == 0:
        print(f"[{count}/{total}] SKIP {video_name}")
        continue

    print(f"[{count}/{total}] Processing {video_name} ({len(json_files)} frames)...")

    kp_array = load_json_folder(folder)

    positions_2d[video_name] = {
        'custom': [kp_array]
    }

    video_metadata[video_name] = {
        'w': FRAME_W,
        'h': FRAME_H
    }

print("=" * 50)

np.savez_compressed(
    OUTPUT_NPZ,
    positions_2d=positions_2d,
    metadata={
        'layout_name'       : 'coco',
        'num_joints'        : 17,
        'keypoints_symmetry': KEYPOINTS_SYMMETRY,
        'video_metadata'    : video_metadata
    }
)

print(f"Tersimpan di: {OUTPUT_NPZ}")
print(f"Total video : {len(positions_2d)}")
