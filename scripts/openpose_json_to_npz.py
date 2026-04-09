"""
Konversi JSON OpenPose -> NPZ untuk input VideoPose3D
Dengan confidence thresholding dan interpolasi keypoint low confidence
"""

import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ============================================================
# KONFIGURASI
# ============================================================
JSON_DIR   = Path("/workspace/awal/json")
OUTPUT_NPZ = Path("/workspace/awal/data_2d_custom_urfd_openpose_new.npz")

FRAME_W    = 640
FRAME_H    = 480
CONF_THRESHOLD = 0.3  # sesuai Paper 3 (IMUTube)

# Mapping OpenPose COCO 18 -> Detectron COCO 17
OPENPOSE_TO_DETECTRON = [
    0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10
]

KEYPOINTS_SYMMETRY = [
    [1, 3, 5, 7, 9, 11, 13, 15],
    [2, 4, 6, 8, 10, 12, 14, 16]
]

# ============================================================
# FUNGSI HELPER
# ============================================================
def load_json_folder(folder):
    """
    Baca semua JSON dari satu folder video.
    Keypoint dengan confidence < threshold di-set NaN,
    lalu diinterpolasi dari frame sekitarnya.
    Return array shape (N_frames, 17, 2)
    """
    json_files = sorted(folder.glob("*_keypoints.json"))
    frames_xy   = []  # koordinat x,y
    frames_conf = []  # confidence

    for jf in json_files:
        data = json.loads(jf.read_text())

        if len(data['people']) > 0:
            kp = np.array(data['people'][0]['pose_keypoints_2d']).reshape(18, 3)

            # Terapkan confidence threshold — set NaN jika low confidence
            xy   = kp[:, :2].copy()
            conf = kp[:, 2].copy()
            xy[conf < CONF_THRESHOLD] = np.nan

        else:
            # Frame kosong
            xy   = np.full((18, 2), np.nan)
            conf = np.zeros(18)

        frames_xy.append(xy)
        frames_conf.append(conf)

    frames_xy   = np.array(frames_xy)    # (N, 18, 2)
    frames_conf = np.array(frames_conf)  # (N, 18)

    # Interpolasi NaN per joint
    N = len(frames_xy)
    for j in range(18):
        for c in range(2):  # x dan y
            col  = frames_xy[:, j, c]
            mask = ~np.isnan(col)

            if mask.sum() >= 2:
                # Ada cukup frame valid untuk interpolasi
                idx_valid = np.where(mask)[0]
                idx_all   = np.arange(N)
                col_interp = interp1d(
                    idx_valid, col[mask],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(col[mask][0], col[mask][-1])
                )
                frames_xy[:, j, c] = col_interp(idx_all)
            elif mask.sum() == 1:
                # Hanya 1 frame valid, isi semua dengan nilai itu
                frames_xy[:, j, c] = col[mask][0]
            else:
                # Tidak ada frame valid, isi 0
                frames_xy[:, j, c] = 0.0

    # Mapping dari 18 keypoint OpenPose ke 17 keypoint Detectron
    kp_17 = frames_xy[:, OPENPOSE_TO_DETECTRON, :]  # (N, 17, 2)

    return kp_17

# ============================================================
# PROSES SEMUA VIDEO
# ============================================================
positions_2d   = {}
video_metadata = {}

video_folders = sorted(JSON_DIR.iterdir())
total = len(video_folders)

print(f"Total video: {total}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print("=" * 50)

for count, folder in enumerate(video_folders, start=1):
    video_name = folder.name

    json_files = list(folder.glob("*_keypoints.json"))
    if len(json_files) == 0:
        print(f"[{count}/{total}] SKIP {video_name} - tidak ada JSON")
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
