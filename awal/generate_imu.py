import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

# Path settings
GLOBAL_POSES_DIR = Path("/workspace/awal/3d_poses")
OUTPUT_DIR = Path("/workspace/awal/synthetic_imu")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS = 30  # URFD video 30 FPS
DT = 1.0 / FPS  # time step antar frame

# Index keypoint yang relevan untuk IMU
# Detectron COCO: 0=Nose, 11=LHip, 12=RHip
# Kita pakai MidHip (rata-rata LHip dan RHip) sebagai posisi sensor
LHIP_IDX = 11
RHIP_IDX = 12

def compute_midhip(poses):
    """
    Hitung posisi mid-hip dari LHip dan RHip.
    Justifikasi: Sensor accelerometer URFD dipasang di pinggang,
    jadi kita estimasi posisi sensor dari rata-rata LHip dan RHip.
    """
    return (poses[:, LHIP_IDX, :] + poses[:, RHIP_IDX, :]) / 2

def smooth_signal(signal, window=5, poly=2):
    """Window lebih kecil agar sinyal gerakan tidak hilang"""
    from scipy.signal import savgol_filter
    if window % 2 == 0:
        window += 1
    smoothed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        smoothed[:, i] = savgol_filter(signal[:, i], window, poly)
    return smoothed

def compute_acceleration(positions, dt):
    from scipy.signal import savgol_filter
    accel = np.zeros_like(positions)
    for i in range(1, len(positions) - 1):
        accel[i] = (positions[i+1] - 2*positions[i] + positions[i-1]) / (dt**2)
    accel[0] = accel[1]
    accel[-1] = accel[-2]
    
    # Smooth ringan setelah diferensiasi (window kecil)
    for i in range(accel.shape[1]):
        accel[:, i] = savgol_filter(accel[:, i], 5, 2)
    
    return accel

# Proses semua video
global_files = sorted(GLOBAL_POSES_DIR.glob("*cam0*.npy"))
print(f"Total videos: {len(global_files)}")

for gf in global_files:
    video_name = gf.stem
    output_file = OUTPUT_DIR / f"{video_name}_imu.npy"
    
    if output_file.exists():
        print(f"Skipping {video_name}")
        continue
    
    print(f"Processing {video_name}...")
    
    # Load global 3D poses
    poses = np.load(gf)  # (frames, 17, 3)
    
    # Normalisasi skala ke ukuran tubuh nyata
    # Justifikasi: VideoPose3D menghasilkan pose dalam unit arbitrary,
    # kita normalisasi berdasarkan tinggi tubuh rata-rata 1.7m
    nose = poses[:, 0, :]
    lankle = poses[:, 15, :]
    rankle = poses[:, 16, :]
    ankle = (lankle + rankle) / 2
    height = np.linalg.norm(nose - ankle, axis=1).mean()
    if height > 0:
        scale = 1.7 / height  # 1.7m tinggi rata-rata manusia
        poses = poses * scale
    # Hitung posisi mid-hip
    midhip = compute_midhip(poses)  # (frames, 3)
    
    # Smooth posisi untuk reduce noise
    midhip_smooth = smooth_signal(midhip, window=7, poly=3)
    
    # Hitung akselerasi
    accel = compute_acceleration(midhip_smooth, DT)  # (frames, 3)
    
    # Tambahkan gravitasi (9.81 m/s^2 di sumbu z)
    # Justifikasi: Real accelerometer selalu mengukur gravitasi,
    # saat diam nilai z ≈ 9.81 m/s^2
    accel[:, 2] += 9.81
    
    # Simpan
    np.save(output_file, accel)
    print(f"  Shape: {accel.shape} -> saved")

print("\nDone!")
