"""
PnP Global Tracking untuk mendapatkan posisi global 3D pose
Input : 3D poses dari VideoPose3D (JSON) + 2D keypoints dari OpenPose (JSON)
        + camera params dari GeoCalib (NPY)
Output: /workspace/awal/3d_poses_global/{video_name}_global.npy

Penjelasan PnP:
- VideoPose3D menghasilkan pose 3D root-relative (selalu centered di origin)
- PnP mencari R, t sehingga: pts_2d = K * (R * pts_3d + t)
- Dengan R dan t, kita transform pose ke koordinat kamera global
- Hasilnya: posisi absolut orang dalam ruang 3D
"""

import cv2
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ============================================================
# KONFIGURASI
# ============================================================
POSES_3D_DIR = Path("/workspace/awal/3d_poses")
JSON_2D_DIR  = Path("/workspace/awal/json")
PARAMS_NPY   = Path("/workspace/awal/camera_params.npy")
OUTPUT_DIR   = Path("/workspace/awal/3d_poses_global")
OUTPUT_DIR.mkdir(exist_ok=True)

FRAME_W = 640
FRAME_H = 480
CONF_THRESHOLD = 0.3

# Distortion coefficients (asumsi tidak ada distorsi)
DIST_COEFFS = np.zeros(4)

# Mapping OpenPose COCO 18 -> Detectron COCO 17
# (sama dengan yang dipakai di openpose_json_to_npz.py)
OPENPOSE_TO_DETECTRON = [
    0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10
]

camera_params = np.load(PARAMS_NPY, allow_pickle=True).item()

# ============================================================
# FUNGSI HELPER
# ============================================================
def load_3d_poses(json_path):
    """Load 3D poses dari JSON VideoPose3D, return (N, 17, 3)"""
    data   = json.loads(Path(json_path).read_text())
    poses  = []
    for frame in data['frames']:
        kps    = frame['keypoints_3d']
        coords = [[kp['x'], kp['y'], kp['z']] for kp in kps]
        poses.append(coords)
    return np.array(poses)  # (N, 17, 3)

def load_2d_keypoints(video_name):
    """
    Load 2D keypoints dari JSON OpenPose.
    Keypoint dengan confidence < threshold di-set 0.
    Return (N, 17, 2) dalam pixel coordinates.
    """
    json_dir   = JSON_2D_DIR / video_name
    json_files = sorted(json_dir.glob("*_keypoints.json"))
    kp_all     = []

    for jf in json_files:
        data = json.loads(jf.read_text())
        if len(data['people']) > 0:
            kp18 = np.array(data['people'][0]['pose_keypoints_2d']).reshape(18, 3)
            # Set low confidence keypoints ke 0
            kp18[kp18[:, 2] < CONF_THRESHOLD, :2] = 0
            # Mapping ke 17 keypoint Detectron
            kp17 = kp18[OPENPOSE_TO_DETECTRON, :2]
        else:
            kp17 = np.zeros((17, 2))
        kp_all.append(kp17)

    return np.array(kp_all)  # (N, 17, 2)

def build_camera_matrix(focal, cx=320.0, cy=240.0):
    """Buat camera intrinsic matrix K dari focal length"""
    return np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ]
    ], dtype=np.float64)

def pnp_global_tracking(poses_3d, kp_2d, K):
    """
    Jalankan PnP per frame untuk mendapatkan posisi global.

    Justifikasi:
    - PnP mencari rotasi R dan translasi t kamera
    - Transform: global = R * local_3d + t
    - Menggunakan SOLVEPNP_EPNP untuk stabilitas

    Args:
        poses_3d: (N, 17, 3) - pose 3D lokal dari VideoPose3D
        kp_2d   : (N, 17, 2) - keypoints 2D pixel dari OpenPose
        K       : (3, 3)     - camera intrinsic matrix

    Returns:
        poses_global: (N, 17, 3) - pose dalam koordinat kamera global
    """
    N = len(poses_3d)
    poses_global  = np.zeros_like(poses_3d)
    prev_valid_idx = -1

    for i in range(N):
        pose_3d = poses_3d[i]  # (17, 3)
        kp      = kp_2d[i]     # (17, 2)

        # Filter keypoint yang valid (bukan 0)
        valid = (kp[:, 0] > 0) & (kp[:, 1] > 0)

        if valid.sum() < 6:
            # Tidak cukup keypoint untuk PnP (butuh minimal 4, tapi 6 lebih stabil)
            if prev_valid_idx >= 0:
                poses_global[i] = poses_global[prev_valid_idx]
            else:
                poses_global[i] = pose_3d
            continue

        pts_3d = pose_3d[valid].astype(np.float64)
        pts_2d = kp[valid].astype(np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d, K, DIST_COEFFS,
                flags=cv2.SOLVEPNP_EPNP
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)
                # Transform: global = R * local + t
                poses_global[i] = (R @ pose_3d.T).T + tvec.flatten()
                prev_valid_idx  = i
            else:
                poses_global[i] = poses_global[prev_valid_idx] if prev_valid_idx >= 0 else pose_3d

        except Exception as e:
            poses_global[i] = poses_global[prev_valid_idx] if prev_valid_idx >= 0 else pose_3d

    return poses_global

# ============================================================
# PROSES SEMUA VIDEO
# ============================================================
json_files = sorted(POSES_3D_DIR.glob("*_world.json"))
total      = len(json_files)

print(f"Total video: {total}")
print("=" * 50)

for count, json_file in enumerate(json_files, start=1):
    video_name  = json_file.name.replace("_world.json", "")
    output_file = OUTPUT_DIR / f"{video_name}_global.npy"

    if output_file.exists():
        print(f"[{count}/{total}] Skip {video_name}")
        continue

    print(f"[{count}/{total}] Processing {video_name}...")

    # Load data
    poses_3d = load_3d_poses(json_file)
    kp_2d    = load_2d_keypoints(video_name)

    # Ambil focal length per video dari GeoCalib
    focal = camera_params.get(video_name, {}).get('focal_length', 500.0)
    K     = build_camera_matrix(focal, cx=FRAME_W/2, cy=FRAME_H/2)

    # Sesuaikan jumlah frames
    min_frames = min(len(poses_3d), len(kp_2d))
    poses_3d   = poses_3d[:min_frames]
    kp_2d      = kp_2d[:min_frames]

    # Jalankan PnP
    poses_global = pnp_global_tracking(poses_3d, kp_2d, K)

    np.save(output_file, poses_global)

    # Cek range mid-hip Y untuk validasi
    midhip_y = (poses_global[:, 11, 1] + poses_global[:, 12, 1]) / 2
    print(f"  Frames: {min_frames}, mid-hip Y range: {midhip_y.min():.3f}~{midhip_y.max():.3f}")

print("")
print("=" * 50)
print("SELESAI!")
