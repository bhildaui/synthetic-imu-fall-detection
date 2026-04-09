"""
Generate Synthetic IMU dari Global 3D Pose (output PnP)
Input : /workspace/awal/3d_poses_global/{video_name}_global.npy
Output: /workspace/awal/synthetic_imu/{video_name}_imu.npy
"""

import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

GLOBAL_DIR = Path("/workspace/awal/3d_poses_global")
OUTPUT_DIR = Path("/workspace/awal/synthetic_imu")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS = 30
DT  = 1.0 / FPS

LHIP_IDX   = 11
RHIP_IDX   = 12

def smooth_signal(signal, window=15, poly=3):
    if window % 2 == 0:
        window += 1
    smoothed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        smoothed[:, i] = savgol_filter(signal[:, i], window, poly)
    return smoothed

def compute_acceleration(positions, dt):
    accel = np.zeros_like(positions)
    for i in range(1, len(positions)-1):
        accel[i] = (positions[i+1] - 2*positions[i] + positions[i-1]) / dt**2
    accel[0]  = accel[1]
    accel[-1] = accel[-2]
    for i in range(accel.shape[1]):
        accel[:, i] = savgol_filter(accel[:, i], 5, 2)
    return accel

# ============================================================
npy_files = sorted(GLOBAL_DIR.glob("*_global.npy"))
total     = len(npy_files)

print(f"Total video: {total}")
print("=" * 50)

for count, npy_file in enumerate(npy_files, start=1):
    video_name  = npy_file.name.replace("_global.npy", "")
    output_file = OUTPUT_DIR / f"{video_name}_imu.npy"

    if output_file.exists():
        output_file.unlink()

    poses  = np.load(npy_file)  # (N, 17, 3)
    midhip = (poses[:, LHIP_IDX, :] + poses[:, RHIP_IDX, :]) / 2

    # Smooth posisi
    midhip_smooth = smooth_signal(midhip, window=15, poly=3)

    # Hitung akselerasi
    accel        = compute_acceleration(midhip_smooth, DT)
    accel[:, 1] += 9.81  # gravitasi di sumbu Y (y:down)

    np.save(output_file, accel)

    mag = np.linalg.norm(accel, axis=1)
    print(f"[{count}/{total}] {video_name}: max={mag.max():.1f}, mean={mag.mean():.1f}")

print("")
print("=" * 50)
print("SELESAI!")
