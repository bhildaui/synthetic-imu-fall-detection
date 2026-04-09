"""
Generate Synthetic IMU dari 2D keypoint MediaPipe (pixel approach)
Sama dengan pipeline OpenPose tapi menggunakan JSON MediaPipe
"""

import numpy as np
import json
from pathlib import Path
from scipy.signal import savgol_filter

JSON_DIR   = Path("/workspace/mediapipe_pipeline/json")
OUTPUT_DIR = Path("/workspace/mediapipe_pipeline/synthetic_imu")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS            = 30
DT             = 1.0 / FPS
CONF_THRESHOLD = 0.3
SMOOTH_WINDOW  = 15
TRIM_FRAMES    = 10

def load_midhip_pixel(video_name):
    json_dir = JSON_DIR / video_name
    files    = sorted(json_dir.glob("*_keypoints.json"))
    mh = []
    for f in files:
        data = json.loads(f.read_text())
        if len(data['people']) > 0:
            # MediaPipe sudah 17 keypoint Detectron
            # LHip=11, RHip=12
            kp = np.array(data['people'][0]['pose_keypoints_2d']).reshape(17, 3)
            rc, lc = kp[12, 2], kp[11, 2]
            if rc < CONF_THRESHOLD and lc < CONF_THRESHOLD:
                mh.append([np.nan, np.nan])
            elif rc < CONF_THRESHOLD:
                mh.append(kp[11, :2].tolist())
            elif lc < CONF_THRESHOLD:
                mh.append(kp[12, :2].tolist())
            else:
                mh.append(((kp[11, :2] + kp[12, :2]) / 2).tolist())
        else:
            mh.append([np.nan, np.nan])
    return np.array(mh)

def interpolate_nan(signal):
    for i in range(signal.shape[1]):
        mask = np.isnan(signal[:, i])
        if mask.any() and (~mask).sum() >= 2:
            idx = np.arange(len(signal))
            signal[mask, i] = np.interp(idx[mask], idx[~mask], signal[~mask, i])
        elif mask.all():
            signal[:, i] = 0.0
    return signal

def compute_acceleration(positions, dt):
    smoothed = np.zeros_like(positions)
    for i in range(positions.shape[1]):
        smoothed[:, i] = savgol_filter(positions[:, i], SMOOTH_WINDOW, 3)
    accel = np.zeros_like(smoothed)
    for i in range(1, len(smoothed)-1):
        accel[i] = (smoothed[i+1] - 2*smoothed[i] + smoothed[i-1]) / dt**2
    accel[0] = accel[1]; accel[-1] = accel[-2]
    for i in range(accel.shape[1]):
        accel[:, i] = savgol_filter(accel[:, i], 5, 2)
    mag = np.linalg.norm(accel, axis=1, keepdims=True)
    return np.hstack([accel, mag])

# ============================================================
video_folders = sorted(f for f in JSON_DIR.iterdir() if '-rgb' in f.name)
total         = len(video_folders)

print(f"Total video: {total}")
print("=" * 50)

for count, folder in enumerate(video_folders, start=1):
    video_name  = folder.name
    output_file = OUTPUT_DIR / f"{video_name}_imu.npy"

    if output_file.exists():
        output_file.unlink()

    json_files = list(folder.glob("*_keypoints.json"))
    if len(json_files) == 0:
        print(f"[{count}/{total}] SKIP {video_name}")
        continue

    print(f"[{count}/{total}] Processing {video_name}...")

    mh = load_midhip_pixel(video_name)
    n_nan = np.isnan(mh).any(axis=1).sum()
    if n_nan > 0:
        mh = interpolate_nan(mh)
    if len(mh) > 2 * TRIM_FRAMES:
        mh = mh[TRIM_FRAMES:-TRIM_FRAMES]

    accel = compute_acceleration(mh, DT)
    np.save(output_file, accel)

    mag = accel[:, 2]
    print(f"  Frames: {len(accel)}, max={mag.max():.1f}, mean={mag.mean():.1f} pixel/s2")

print("\n" + "=" * 50)
print("SELESAI!")
