"""
Generate Synthetic IMU dengan Back-projection + Kalman Filter
Input : JSON OpenPose (2D pixel) + JSON VideoPose3D (depth Z) + camera params
Output: /workspace/awal/synthetic_imu/{video_name}_imu.npy

Pipeline:
1. Back-projection 2D pixel + depth Z -> posisi 3D global mid-hip
2. Kalman filter untuk stabilkan trajectory
3. Diferensiasi kedua -> akselerasi
4. Tambah gravitasi
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
CX, CY = 320.0, 240.0

cam_params = np.load(PARAMS_NPY, allow_pickle=True).item()

OPENPOSE_TO_DETECTRON = [0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10]
CONF_THRESHOLD = 0.3

# ============================================================
# KALMAN FILTER
# ============================================================
class KalmanFilter3D:
    """
    Kalman filter untuk tracking posisi 3D.
    
    State vector: [x, y, z, vx, vy, vz] (posisi + kecepatan)
    Justifikasi:
    - Prediction step: estimasi posisi berdasarkan kecepatan sebelumnya
    - Update step: koreksi menggunakan observasi back-projection
    - Process noise (Q): seberapa besar perubahan kecepatan yang diizinkan
    - Measurement noise (R): seberapa percaya kita pada observasi
    """
    def __init__(self, dt, process_noise=0.1, measurement_noise=1.0):
        self.dt = dt
        self.n  = 6  # state dimension: [x,y,z,vx,vy,vz]

        # State transition matrix F
        # x_new = x + vx*dt, vx_new = vx (constant velocity model)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix H (kita hanya observasi posisi x,y,z)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Process noise covariance Q
        self.Q = np.eye(6) * process_noise
        self.Q[3:, 3:] *= 10  # kecepatan lebih noisy dari posisi

        # Measurement noise covariance R
        self.R = np.eye(3) * measurement_noise

        # Initial state dan covariance
        self.x = np.zeros(6)   # state
        self.P = np.eye(6) * 1.0  # covariance

        self.initialized = False

    def initialize(self, position):
        """Inisialisasi state dengan posisi pertama"""
        self.x[:3] = position
        self.x[3:] = 0
        self.initialized = True

    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]

    def update(self, measurement):
        """Update step dengan observasi baru"""
        # Innovation
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

        return self.x[:3]

    def smooth(self, position):
        """Predict lalu update"""
        if not self.initialized:
            self.initialize(position)
            return position
        self.predict()
        return self.update(position)

# ============================================================
# FUNGSI HELPER
# ============================================================
def load_midhip_2d(video_name):
    """Load posisi mid-hip dalam pixel, return (N, 2) dengan NaN jika kosong"""
    json_dir  = JSON_DIR / video_name
    files     = sorted(json_dir.glob("*_keypoints.json"))
    midhip_2d = []

    for f in files:
        data = json.loads(f.read_text())
        if len(data['people']) > 0:
            kp = np.array(data['people'][0]['pose_keypoints_2d']).reshape(18, 3)
            # Pakai hip yang confidence-nya lebih tinggi
            rhip_conf = kp[8, 2]
            lhip_conf = kp[11, 2]

            if rhip_conf < CONF_THRESHOLD and lhip_conf < CONF_THRESHOLD:
                midhip_2d.append(np.array([np.nan, np.nan]))
            elif rhip_conf < CONF_THRESHOLD:
                midhip_2d.append(kp[11, :2])  # pakai LHip saja
            elif lhip_conf < CONF_THRESHOLD:
                midhip_2d.append(kp[8, :2])   # pakai RHip saja
            else:
                midhip_2d.append((kp[8, :2] + kp[11, :2]) / 2)
        else:
            midhip_2d.append(np.array([np.nan, np.nan]))

    return np.array(midhip_2d)

def load_midhip_depth(video_name):
    """Load depth Z mid-hip dari VideoPose3D"""
    pose_path = POSES_DIR / f"{video_name}_world.json"
    data      = json.loads(pose_path.read_text())
    poses     = np.array([[[kp['x'], kp['y'], kp['z']]
                           for kp in f['keypoints_3d']]
                           for f in data['frames']])
    return (poses[:, 11, 2] + poses[:, 12, 2]) / 2

def interpolate_nan(signal):
    """Interpolasi linear untuk NaN"""
    for i in range(signal.shape[1]):
        mask = np.isnan(signal[:, i])
        if mask.any() and (~mask).sum() >= 2:
            idx = np.arange(len(signal))
            signal[mask, i] = np.interp(
                idx[mask], idx[~mask], signal[~mask, i])
        elif mask.all():
            signal[:, i] = 0.0
    return signal

def compute_acceleration(positions, dt):
    """Diferensiasi kedua untuk akselerasi"""
    accel = np.zeros_like(positions)
    for i in range(1, len(positions)-1):
        accel[i] = (positions[i+1] - 2*positions[i] + positions[i-1]) / dt**2
    accel[0]  = accel[1]
    accel[-1] = accel[-2]
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

    # Back-projection ke 3D
    X = (midhip_2d[:, 0] - CX) * midhip_z / focal
    Y = (midhip_2d[:, 1] - CY) * midhip_z / focal
    midhip_3d = np.stack([X, Y, midhip_z], axis=1)

    # Interpolasi NaN
    n_nan = np.isnan(midhip_3d).any(axis=1).sum()
    if n_nan > 0:
        midhip_3d = interpolate_nan(midhip_3d)

    # Terapkan Kalman filter untuk stabilkan trajectory
    kf        = KalmanFilter3D(dt=DT, process_noise=0.5, measurement_noise=2.0)
    midhip_kf = np.zeros_like(midhip_3d)
    for i in range(len(midhip_3d)):
        midhip_kf[i] = kf.smooth(midhip_3d[i])

    # Hitung akselerasi
    accel        = compute_acceleration(midhip_kf, DT)
    accel[:, 1] += 9.81  # gravitasi di sumbu Y

    np.save(output_file, accel)

    mag = np.linalg.norm(accel, axis=1)
    print(f"  Frames: {len(accel)}, max={mag.max():.1f}, mean={mag.mean():.1f} m/s2")

print("")
print("=" * 50)
print("SELESAI!")
