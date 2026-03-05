import numpy as np
from pathlib import Path

# Path settings
POSES_3D_DIR = Path("/workspace/awal/3d_poses")
OUTPUT_DIR = Path("/workspace/awal/3d_poses_global")
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameter kamera URFD (estimasi dari resolusi 640x240)
# Justifikasi: focal length = max(w,h) adalah estimasi standar untuk unknown cameras
VIDEO_W = 640
VIDEO_H = 240
# Focal length dari GeoCalib (median dari 70 video)
# Justifikasi: URFD menggunakan fixed camera, median lebih robust dari outlier
FOCAL_LENGTH = 383.94
CX = VIDEO_W / 2  # principal point x = 320
CY = VIDEO_H / 2  # principal point y = 120

# Camera intrinsic matrix
K = np.array([
    [FOCAL_LENGTH, 0, CX],
    [0, FOCAL_LENGTH, CY],
    [0, 0, 1]
], dtype=np.float64)

# Distortion coefficients (asumsi tidak ada distorsi)
dist_coeffs = np.zeros(4)

def project_to_global(poses_3d, keypoints_2d):
    """
    Menggunakan PnP untuk mendapatkan posisi global dari pose 3D lokal.
    
    Penjelasan:
    - VideoPose3D menghasilkan pose 3D dalam koordinat lokal (selalu centered)
    - PnP mencari transformasi (rotasi R + translasi t) yang menjelaskan
      bagaimana pose 3D lokal berhubungan dengan posisi 2D di kamera
    - Dengan R dan t, kita bisa transform pose ke koordinat global
    
    Args:
        poses_3d: array (frames, 17, 3) - pose 3D lokal dari VideoPose3D
        keypoints_2d: array (frames, 17, 2) - keypoints 2D dari OpenPose
    
    Returns:
        poses_global: array (frames, 17, 3) - pose 3D dalam koordinat global
    """
    n_frames = len(poses_3d)
    poses_global = np.zeros_like(poses_3d)
    
    for i in range(n_frames):
        pose_3d = poses_3d[i]  # (17, 3)
        kp_2d = keypoints_2d[i]  # (17, 2)
        
        # Filter keypoints dengan nilai valid (bukan 0)
        valid = (kp_2d[:, 0] > 0) & (kp_2d[:, 1] > 0)
        
        if valid.sum() < 4:
            # Tidak cukup keypoints untuk PnP, pakai frame sebelumnya
            if i > 0:
                poses_global[i] = poses_global[i-1]
            else:
                poses_global[i] = pose_3d
            continue
        
        # Ambil keypoints yang valid
        pts_3d = pose_3d[valid].astype(np.float64)
        pts_2d = kp_2d[valid].astype(np.float64)
        
        # Jalankan PnP
        # PnP mencari R, t sehingga pts_2d = K * (R * pts_3d + t)
        try:
            import cv2
            success, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d, K, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success:
                # Convert rotation vector ke matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Transform semua keypoints ke koordinat global
                # global = R * local + t
                poses_global[i] = (R @ pose_3d.T).T + tvec.T
            else:
                poses_global[i] = pose_3d if i == 0 else poses_global[i-1]
                
        except Exception:
            poses_global[i] = pose_3d if i == 0 else poses_global[i-1]
    
    return poses_global

# Load 2D keypoints
import json
import numpy as np

def load_2d_keypoints(video_name):
    """Load 2D keypoints dari JSON OpenPose"""
    json_dir = Path(f"/workspace/awal/json/{video_name}")
    json_files = sorted(json_dir.glob("*.json"))
    
    keypoints_all = []
    for jf in json_files:
        data = json.load(open(jf))
        if len(data['people']) == 0:
            keypoints_all.append(np.zeros((18, 2)))
        else:
            flat = data['people'][0]['pose_keypoints_2d']
            kp = np.array(flat).reshape(18, 3)
            keypoints_all.append(kp[:, :2])
    
    return np.array(keypoints_all)

# Proses semua video cam0
npy_files = sorted(POSES_3D_DIR.glob("*cam0*.npy"))
print(f"Total videos: {len(npy_files)}")

for npy_file in npy_files:
    video_name = npy_file.stem  # contoh: fall-01-cam0
    output_file = OUTPUT_DIR / f"{video_name}_global.npy"
    
    if output_file.exists():
        print(f"Skipping {video_name} (already done)")
        continue
    
    print(f"Processing {video_name}...")
    
    # Load 3D poses
    poses_3d = np.load(npy_file)  # (frames, 17, 3)
    
    # Load 2D keypoints (mapping ke 17 keypoints)
    kp_2d_18 = load_2d_keypoints(video_name)  # (frames, 18, 2)
    
    # Remap ke 17 keypoints (sama seperti di openpose_to_npz.py)
    openpose_to_detectron = [0, 16, 15, -1, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
    kp_2d_17 = np.zeros((len(kp_2d_18), 17, 2))
    for det_idx, op_idx in enumerate(openpose_to_detectron):
        if op_idx >= 0:
            kp_2d_17[:, det_idx, :] = kp_2d_18[:, op_idx, :]
    
    # Sesuaikan jumlah frames
    min_frames = min(len(poses_3d), len(kp_2d_17))
    poses_3d = poses_3d[:min_frames]
    kp_2d_17 = kp_2d_17[:min_frames]
    
    # Jalankan PnP global tracking
    poses_global = project_to_global(poses_3d, kp_2d_17)
    
    # Simpan
    np.save(output_file, poses_global)
    print(f"  Shape: {poses_global.shape} -> saved to {output_file}")

print("\nDone!")
