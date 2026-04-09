"""
Script batch GeoCalib untuk estimasi focal length semua video URFD
"""

import numpy as np
from pathlib import Path
from geocalib import GeoCalib
import torch
import cv2

RGB_DIR    = Path("/workspace/URFD/rgb")
OUTPUT_NPY = Path("/workspace/awal/camera_params.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan device: {device}")
model = GeoCalib().to(device)
model.eval()

camera_params = {}
video_folders = sorted(RGB_DIR.iterdir())
total         = len(video_folders)

print(f"Total video: {total}")
print("=" * 50)

for count, video_folder in enumerate(video_folders, start=1):
    video_name = video_folder.name
    frame_dir  = video_folder / video_name

    if not frame_dir.exists():
        print(f"[{count}/{total}] SKIP {video_name} - folder tidak ditemukan")
        continue

    frames = sorted(frame_dir.glob("*.png"))
    if len(frames) == 0:
        print(f"[{count}/{total}] SKIP {video_name} - tidak ada frame")
        continue

    mid_frame = frames[len(frames) // 2]
    print(f"[{count}/{total}] Processing {video_name}...")

    try:
        img_bgr    = cv2.imread(str(mid_frame))
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).to(device)

        with torch.no_grad():
            result = model.calibrate(img_tensor)

        # Ambil focal length langsung — sudah dalam pixel
        focal = result['camera'].f.mean().item()

        camera_params[video_name] = {
            'focal_length': focal,
            'res_w': 640,
            'res_h': 480,
            'frame_used': mid_frame.name
        }

        print(f"  Focal length: {focal:.2f} px")

    except Exception as e:
        print(f"  ERROR: {e}")
        camera_params[video_name] = {
            'focal_length': 384.0,
            'res_w': 640,
            'res_h': 480,
            'frame_used': 'default'
        }

print("=" * 50)
np.save(OUTPUT_NPY, camera_params)
print(f"Tersimpan di: {OUTPUT_NPY}")

focals = [v['focal_length'] for v in camera_params.values()]
print(f"Min: {min(focals):.2f}, Max: {max(focals):.2f}, Median: {np.median(focals):.2f}")
