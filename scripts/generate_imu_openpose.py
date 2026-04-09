"""
Generate Synthetic IMU dari 3D Pose menggunakan back-projection
Input : JSON OpenPose (2D pixel) + JSON VideoPose3D (depth Z) + camera params
Output: /workspace/awal/synthetic_imu/{video_name}_imu.npy
"""

import numpy as np
import json
from pathlib import Path
from scipy.signal import savgol_filter

# ============================================================
# KONFIGURASI
# ============================================================
JSON_DIR   = Path("/workspace/awal/json")
POSES_DIR  = Path("/workspace/awal/3d_poses")
PARAMS_NPY = Path("/workspace/awal/camera_params.npy")
OUTPUT_DIR = Path("/workspace/awal/synthetic_imu")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS    = 30
DT     = 1.0 / FPS
CX, CY = 320.0, 240.0  # principal point (center image 640x480)
SMOOTH_WINDOW = 15      # window Savitzky-Golay

cam_params = np.load(PARAMS_NPY, allow_pickle=True).item()

# ============================================================
# FUNGSI HELPER
# ============================================================
def load_midhip_2d(video_name):
    """Load posisi mid-hip dalam pixel dari JSON OpenPose"""
    json_dir  = JSON_DIR / video_name
    files     = sorted(json_dir.glob("*_keypoints.json"))
    midhip_2d = []
    for f in files:
        data = json.loads(f.read_text())
        if len(data['people']) > 0:
            kp = np.array(data['people'][0]['pose_keypoints_2d']).reshape(18, 3)
            # RHip=8, LHip=11 dalam OpenPose COCO
            midhip = (kp[8, :2] + kp[11, :2]) / 2
        else:
            midhip = np.array([np.nan, np.nan])
        midhip_2d.append(midhip)
    return np.array(midhip_2d)  # (N, 2)

def load_midhip_depth(video_name):
    """Load depth Z mid-hip dari JSON VideoPose3D"""
    pose_path = POSES_DIR / f"{video_name}_world.json"
    data      = json.loads(pose_path.read_text())
    poses     = np.array([[[kp['x'], kp['y'], kp['z']]
                           for kp in f['keypoints_3d']]
                           for f in data['frames']])
    # Mid-hip depth: rata-rata LHip(11) dan RHip(12)
    return (poses[:, 11, 2] + poses[:, 12, 2]) / 2  # (N,)

def backproject_to_3d(midhip_2d, midhip_z, focal):
    """
    Back-projection dari 2D pixel ke 3D global.
    Rumus: X = (u - cx) * Z / f
           Y = (v - cy) * Z / f
    Justifikasi: menggunakan focal length dari GeoCalib untuk
    konversi koordinat kamera ke koordinat dunia nyata.
    """
    X = (midhip_2d[:, 0] - CX) * midhip_z / focal
    Y = (midhip_2d[:, 1] - CY) * midhip_z / focal
    return np.stack([X, Y, midhip_z], axis=1)  # (N, 3)

def interpolate_nan(signal):
    """Interpolasi linear untuk nilai NaN"""
    for i in range(signal.shape[1]):
        mask = np.isnan(signal[:, i])
        if mask.any():
            idx = np.arange(len(signal))
            signal[mask, i] = np.interp(idx[mask], idx[~mask], signal[~mask, i])
    return signal

def smooth_signal(signal, window=SMOOTH_WINDOW, poly=3):
    if window % 2 == 0:
        window += 1
    smoothed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        smoothed[:, i] = savgol_filter(signal[:, i], window, poly)
    return smoothed

def compute_acceleration(positions, dt):
    """Diferensiasi kedua untuk hitung akselerasi"""
    accel = np.zeros_like(positions)
    for i in range(1, len(positions) - 1):
        accel[i] = (positions[i+1] - 2*positions[i] + positions[i-1]) / dt**2
    accel[0]  = accel[1]
    accel[-1] = accel[-2]
    # Smooth ringan setelah diferensiasi
    for i in range(accel.shape[1]):
        accel[:, i] = savgol_filter(accel[:, i], 5, 2)
    return accel

# ============================================================
# PROSES SEMUA VIDEO
# ============================================================
pose_files = sorted(POSES_DIR.glob("*_world.json"))
total      = len(pose_files)

print(f"Total video: {total}")
print("=" * 50)

for count, pose_file in enumerate(pose_files, start=1):
    video_name  = pose_file.name.replace("_world.json", "")
    output_file = OUTPUT_DIR / f"{video_name}_imu.npy"

    if output_file.exists():
        output_file.unlink()

    print(f"[{count}/{total}] Processing {video_name}...")

    # Load data
    midhip_2d = load_midhip_2d(video_name)
    midhip_z  = load_midhip_depth(video_name)
    focal     = cam_params.get(video_name, {}).get('focal_length', 500.0)

    # Back-projection ke 3D global
    midhip_3d = backproject_to_3d(midhip_2d, midhip_z, focal)

    # Interpolasi NaN (frame kosong)
    n_nan = np.isnan(midhip_3d).any(axis=1).sum()
    if n_nan > 0:
        midhip_3d = interpolate_nan(midhip_3d)
        print(f"  Interpolated {n_nan} NaN frames")

    # Smooth posisi
    midhip_smooth = smooth_signal(midhip_3d)

    # Hitung akselerasi
    accel        = compute_acceleration(midhip_smooth, DT)
    accel[:, 1] += 9.81  # tambah gravitasi di sumbu Y (y:down)

    np.save(output_file, accel)

    mag = np.linalg.norm(accel, axis=1)
    print(f"  Frames: {len(accel)}, max={mag.max():.1f}, mean={mag.mean():.1f} m/s2")

print("")
print("=" * 50)
print("SELESAI!")
