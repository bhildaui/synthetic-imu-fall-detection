#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk mengkonversi OpenPose JSON output dari HumanEva dataset
ke format yang dibutuhkan VideoPose3D

Author: Nay (Thesis S2 UI)
Dataset: HumanEva (15 keypoints)
Target: VideoPose3D 3D Pose Estimation

Justifikasi Script:
- Membaca JSON output dari OpenPose (18 keypoints)
- Convert ke format HumanEva (15 keypoints)
- Handle missing detections dengan interpolasi linear
- Simpan dalam format NPZ yang dibutuhkan VideoPose3D

HumanEva 15 Keypoints Order:
0:Hip, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle,
7:Spine, 8:Thorax, 9:Neck/Nose, 10:Head, 11:LShoulder, 12:LElbow,
13:LWrist, 14:RShoulder, 15:RElbow, 16:RWrist
"""

import numpy as np
import json
import os
import glob
import cv2
from collections import defaultdict

# =============================================================================
# KONFIGURASI
# =============================================================================

# Path ke dataset (sesuaikan dengan lokasi Anda)
DATASET_BASE = '/workspace/URFD'

# Output directory (folder data/ di VideoPose3D)
OUTPUT_DIR = '/workspace/VideoPose3D/data'

# Pilihan konversi format
# True = Convert OpenPose 18 → HumanEva 15
# False = Tetap OpenPose 18 (tidak disarankan untuk HumanEva)
CONVERT_TO_HUMANEVA = True

# Pilihan normalisasi koordinat
# True = Normalize ke [-1, 1]
# False = Tetap pixel coordinates (RECOMMENDED)
USE_NORMALIZATION = False

# FPS video (sesuaikan dengan dataset Anda)
VIDEO_FPS = 60  # HumanEva biasanya 60 fps

# Dataset name untuk output file
DATASET_NAME = 'humaneva'

# =============================================================================
# FUNGSI UTILITY
# =============================================================================

def load_json_keypoints(json_path):
    """
    Membaca file JSON OpenPose dan ekstrak keypoints 2D
    
    Args:
        json_path: path ke file JSON OpenPose
        
    Returns:
        keypoints: array [18, 2] berisi koordinat (x,y) dalam pixels
        confidence: array [18] berisi confidence scores
        
    Justifikasi:
    - OpenPose format: 18 keypoints
    - pose_keypoints_2d berisi [x1,y1,c1, x2,y2,c2, ...] = 54 values
    - Return None jika tidak ada deteksi
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None, None
    
    # Cek apakah ada orang terdeteksi
    if data.get('num_people', 0) == 0:
        return None, None
    
    # Ambil orang pertama (index 0)
    person = data['people'][0]
    pose_data = person['pose_keypoints_2d']
    
    # Hitung jumlah keypoints
    num_keypoints = len(pose_data) // 3
    
    # Validasi
    if num_keypoints != 18:
        print(f'Warning in {json_path}: Expected 18 keypoints, got {num_keypoints}')
    
    # Reshape dari [54] menjadi [18, 3]
    pose_array = np.array(pose_data).reshape(num_keypoints, 3)
    
    keypoints = pose_array[:, :2]  # x, y
    confidence = pose_array[:, 2]   # confidence
    
    return keypoints, confidence


def get_image_dimensions(json_folder):
    """
    Mendapatkan dimensi gambar dari file gambar asli
    
    Args:
        json_folder: folder berisi JSON files
        
    Returns:
        width, height: dimensi gambar dalam pixels
        
    Justifikasi:
    - Baca langsung dari file PNG untuk akurasi
    - Fallback ke estimasi jika file tidak ditemukan
    """
    parent_folder = os.path.dirname(json_folder)
    json_files = sorted(glob.glob(os.path.join(json_folder, '*.json')))
    
    if len(json_files) == 0:
        print("Warning: No JSON files found")
        return 640, 480  # Default
    
    # Coba baca dari file gambar
    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        if 'image' in data:
            image_name = data['image']
            image_path = os.path.join(parent_folder, image_name)
            
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    print(f'  Image dimensions from file: {width}x{height}')
                    return width, height
    except Exception as e:
        print(f"  Warning: Could not read image file: {e}")
    
    # Fallback: estimasi dari keypoints
    print("  Estimating dimensions from keypoints...")
    for json_file in json_files[:10]:
        keypoints, confidence = load_json_keypoints(json_file)
        if keypoints is not None:
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            x_coords = x_coords[x_coords > 0]
            y_coords = y_coords[y_coords > 0]
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                width = int(np.max(x_coords) * 1.2)
                height = int(np.max(y_coords) * 1.2)
                print(f'  Estimated dimensions: {width}x{height}')
                return width, height
    
    # HumanEva default resolution
    print("  Warning: Using default resolution 640x480")
    return 640, 480


def normalize_keypoints(keypoints, width, height):
    """
    Normalisasi koordinat pixel ke range [-1, 1]
    
    Args:
        keypoints: array [num_frames, N, 2] dalam pixels
        width, height: dimensi gambar
        
    Returns:
        normalized_keypoints: array dengan koordinat ternormalisasi
    """
    norm_kp = keypoints.copy()
    norm_kp[:, :, 0] = (keypoints[:, :, 0] / width) * 2 - 1
    norm_kp[:, :, 1] = (keypoints[:, :, 1] / height) * 2 - 1
    return norm_kp


def convert_openpose_to_humaneva(keypoints_18):
    """
    Convert OpenPose 18 keypoints ke HumanEva 15 keypoints
    
    Args:
        keypoints_18: array [num_frames, 18, 2] OpenPose format
        
    Returns:
        keypoints_15: array [num_frames, 15, 2] HumanEva format
        
    Justifikasi:
    - HumanEva menggunakan 15 keypoints berbeda dari COCO
    - Mapping berdasarkan anatomical correspondence
    - Beberapa keypoints perlu di-approximate (misal: Hip = midpoint antara LHip dan RHip)
    
    OpenPose 18 keypoints:
    0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist,
    5:LShoulder, 6:LElbow, 7:LWrist, 8:RHip, 9:RKnee,
    10:RAnkle, 11:LHip, 12:LKnee, 13:LAnkle, 14:REye,
    15:LEye, 16:REar, 17:LEar
    
    HumanEva 15 keypoints (0-indexed):
    0:Hip (center), 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle,
    7:Spine (mid torso), 8:Thorax (upper torso), 9:Neck/Nose, 10:Head (top),
    11:LShoulder, 12:LElbow, 13:LWrist, 14:RShoulder, 15:RElbow, 16:RWrist
    
    Note: HumanEva sebenarnya 15 joints tapi index 0-14, total 15 joints
    Tapi beberapa implementasi mungkin 0-16 (17 joints), perlu disesuaikan
    """
    num_frames = keypoints_18.shape[0]
    keypoints_15 = np.zeros((num_frames, 15, 2), dtype=np.float32)
    
    for i in range(num_frames):
        kp = keypoints_18[i]
        
        # 0: Hip (center) - midpoint antara LHip (11) dan RHip (8)
        keypoints_15[i, 0] = (kp[11] + kp[8]) / 2
        
        # 1: RHip - dari OpenPose RHip (8)
        keypoints_15[i, 1] = kp[8]
        
        # 2: RKnee - dari OpenPose RKnee (9)
        keypoints_15[i, 2] = kp[9]
        
        # 3: RAnkle - dari OpenPose RAnkle (10)
        keypoints_15[i, 3] = kp[10]
        
        # 4: LHip - dari OpenPose LHip (11)
        keypoints_15[i, 4] = kp[11]
        
        # 5: LKnee - dari OpenPose LKnee (12)
        keypoints_15[i, 5] = kp[12]
        
        # 6: LAnkle - dari OpenPose LAnkle (13)
        keypoints_15[i, 6] = kp[13]
        
        # 7: Spine - midpoint antara Hip (calculated) dan Neck (1)
        hip_center = (kp[11] + kp[8]) / 2
        keypoints_15[i, 7] = (hip_center + kp[1]) / 2
        
        # 8: Thorax - midpoint antara shoulders atau gunakan Neck
        # Approximate: midpoint antara LShoulder (5) dan RShoulder (2)
        keypoints_15[i, 8] = (kp[5] + kp[2]) / 2
        
        # 9: Neck/Nose - dari OpenPose Neck (1) atau Nose (0)
        # Preferably Neck karena lebih stabil
        keypoints_15[i, 9] = kp[1]
        
        # 10: Head (top) - approximate dari Nose (0) + offset ke atas
        # atau gunakan midpoint antara eyes
        # Approximate: Nose + vector dari Neck ke Nose
        neck_to_nose = kp[0] - kp[1]
        keypoints_15[i, 10] = kp[0] + neck_to_nose * 0.5
        
        # 11: LShoulder - dari OpenPose LShoulder (5)
        keypoints_15[i, 11] = kp[5]
        
        # 12: LElbow - dari OpenPose LElbow (6)
        keypoints_15[i, 12] = kp[6]
        
        # 13: LWrist - dari OpenPose LWrist (7)
        keypoints_15[i, 13] = kp[7]
        
        # 14: RShoulder - dari OpenPose RShoulder (2)
        keypoints_15[i, 14] = kp[2]
        
    # Note: HumanEva has 15 joints (indices 0-14)
    # Some sources may list it as having indices up to 16
    # Adjust if your specific HumanEva format differs
    
    return keypoints_15


def interpolate_missing_frames(keypoints, valid_indices):
    """
    Interpolasi linear untuk frame yang missing
    
    Args:
        keypoints: array [num_frames, N, 2] dengan NaN pada missing frames
        valid_indices: list of frame indices yang punya deteksi valid
        
    Returns:
        interpolated_keypoints: array dengan NaN sudah diisi
    """
    num_frames = len(keypoints)
    num_joints = keypoints.shape[1]
    indices = np.arange(num_frames)
    
    for joint_idx in range(num_joints):
        for dim in range(2):
            valid_mask = ~np.isnan(keypoints[valid_indices, joint_idx, dim])
            valid_idx_filtered = [valid_indices[i] for i in range(len(valid_indices)) 
                                  if valid_mask[i]]
            
            if len(valid_idx_filtered) == 0:
                keypoints[:, joint_idx, dim] = 0
                continue
            
            if len(valid_idx_filtered) == 1:
                keypoints[:, joint_idx, dim] = keypoints[valid_idx_filtered[0], joint_idx, dim]
                continue
            
            valid_values = keypoints[valid_idx_filtered, joint_idx, dim]
            keypoints[:, joint_idx, dim] = np.interp(
                indices,
                valid_idx_filtered,
                valid_values
            )
    
    return keypoints


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def process_video_sequence(json_folder, convert_to_humaneva=True, normalize=False):
    """
    Proses semua JSON files dari satu video sequence
    
    Args:
        json_folder: folder yang berisi file-file JSON
        convert_to_humaneva: apakah convert OpenPose 18 → HumanEva 15
        normalize: apakah normalisasi coordinates
        
    Returns:
        keypoints_sequence: array [num_frames, N, 2] dimana N=15 atau 18
        metadata: informasi tambahan tentang video
    """
    # Cari dan sort JSON files berdasarkan nomor frame
    json_files = sorted(
        glob.glob(os.path.join(json_folder, '*.json')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[-1])
    )
    
    if len(json_files) == 0:
        print(f'  ERROR: No JSON files found in {json_folder}')
        return None, None
    
    print(f'  Found {len(json_files)} JSON files')
    
    # Deteksi dimensi gambar
    width, height = get_image_dimensions(json_folder)
    
    all_keypoints = []
    valid_indices = []
    
    # Proses setiap frame
    for idx, json_file in enumerate(json_files):
        keypoints, confidence = load_json_keypoints(json_file)
        
        if keypoints is None:
            all_keypoints.append(np.full((18, 2), np.nan, dtype=np.float32))
        else:
            # Handle keypoints dengan confidence = 0
            for i in range(len(keypoints)):
                if confidence[i] == 0.0 or (keypoints[i, 0] == 0 and keypoints[i, 1] == 0):
                    keypoints[i] = [np.nan, np.nan]
            
            all_keypoints.append(keypoints.astype(np.float32))
            valid_indices.append(idx)
    
    # Convert ke numpy array
    keypoints_array = np.array(all_keypoints, dtype=np.float32)
    
    # Interpolasi untuk frame yang missing
    if len(valid_indices) > 0:
        keypoints_array = interpolate_missing_frames(keypoints_array, valid_indices)
        num_interpolated = len(json_files) - len(valid_indices)
        print(f'  {len(json_files)} total frames processed')
        print(f'  {num_interpolated} frames were interpolated')
    else:
        print('  WARNING: No valid detections found!')
        return None, None
    
    # Convert OpenPose 18 → HumanEva 15 (jika diminta)
    if convert_to_humaneva:
        print('  Converting OpenPose 18 → HumanEva 15 keypoints')
        keypoints_array = convert_openpose_to_humaneva(keypoints_array)
    
    # Normalisasi (jika diminta)
    if normalize:
        print('  Normalizing coordinates to [-1, 1]')
        keypoints_array = normalize_keypoints(keypoints_array, width, height)
    
    # Metadata
    metadata = {
        'w': width,
        'h': height,
        'fps': VIDEO_FPS,
    }
    
    return keypoints_array, metadata


def main():
    """
    Main function untuk proses semua video HumanEva
    """
    print("="*70)
    print("HumanEva Dataset Preparation for VideoPose3D")
    print("="*70)
    print(f"Dataset base directory: {DATASET_BASE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Convert to HumanEva: {CONVERT_TO_HUMANEVA}")
    print(f"Use normalization: {USE_NORMALIZATION}")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Dictionary untuk menyimpan semua video
    output = {}
    
    # Metadata global
    if CONVERT_TO_HUMANEVA:
        # HumanEva 15 keypoints
        metadata_all = {
            'layout_name': 'humaneva15',
            'num_joints': 15,
            'keypoints_symmetry': [
                [4, 5, 6, 11, 12, 13],      # Left side: LHip, LKnee, LAnkle, LShoulder, LElbow, LWrist
                [1, 2, 3, 14, 15, 16]       # Right side: RHip, RKnee, RAnkle, RShoulder, RElbow, RWrist
                # Note: Indices adjusted if using 0-14 (15 joints)
            ],
            'video_metadata': {}
        }
    else:
        # OpenPose 18 keypoints
        metadata_all = {
            'layout_name': 'openpose',
            'num_joints': 18,
            'keypoints_symmetry': [
                [1, 2, 3, 4, 8, 9, 10, 15, 17],
                [1, 5, 6, 7, 11, 12, 13, 14, 16]
            ],
            'video_metadata': {}
        }
    
    # Cari semua folder yang berisi JSON outputs
    # Pattern bisa disesuaikan dengan struktur dataset Anda
    video_folders = sorted(glob.glob(os.path.join(DATASET_BASE, '*-out/json')))
    
    # Alternative pattern jika struktur berbeda
    if len(video_folders) == 0:
        video_folders = sorted(glob.glob(os.path.join(DATASET_BASE, '*/json')))
    
    if len(video_folders) == 0:
        print(f"ERROR: No video folders found in {DATASET_BASE}")
        print("Expected pattern: <DATASET_BASE>/*-out/json or <DATASET_BASE>/*/json")
        return
    
    print(f"\nFound {len(video_folders)} video folders to process\n")
    
    # Proses setiap video
    for video_folder in video_folders:
        # Extract nama video
        video_name = os.path.basename(os.path.dirname(video_folder)).replace('-out', '')
        
        print(f"Processing: {video_name}")
        
        # Proses video sequence
        keypoints, video_metadata = process_video_sequence(
            video_folder,
            convert_to_humaneva=CONVERT_TO_HUMANEVA,
            normalize=USE_NORMALIZATION
        )
        
        if keypoints is None:
            print(f"  SKIPPED: {video_name} - no valid detections\n")
            continue
        
        # Format output sesuai VideoPose3D
        output[video_name] = {
            'custom': [keypoints]
        }
        
        metadata_all['video_metadata'][video_name] = video_metadata
        print(f"  SUCCESS: Shape {keypoints.shape}")
        print("-"*70)
    
    # Simpan hasil
    if CONVERT_TO_HUMANEVA:
        format_suffix = 'humaneva15'
    else:
        format_suffix = 'openpose'
    
    norm_suffix = '_normalized' if USE_NORMALIZATION else '_pixel'
    
    output_filename = os.path.join(
        OUTPUT_DIR,
        f'data_2d_custom_{DATASET_NAME}_{format_suffix}{norm_suffix}.npz'
    )
    
    print(f"\nSaving to {output_filename}...")
    
    np.savez_compressed(
        output_filename,
        positions_2d=output,
        metadata=metadata_all
    )
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Processed videos: {len(output)}")
    print(f"Keypoint format: {metadata_all['layout_name']} ({metadata_all['num_joints']} joints)")
    print(f"Coordinate type: {'Normalized [-1,1]' if USE_NORMALIZATION else 'Pixel coordinates'}")
    print(f"Output file: {output_filename}")
    print("="*70)
    
    # Tampilkan sample data
    if len(output) > 0:
        sample_video = list(output.keys())[0]
        sample_data = output[sample_video]['custom'][0]
        sample_meta = metadata_all['video_metadata'][sample_video]
        
        print(f"\nSample data from '{sample_video}':")
        print(f"  Shape: {sample_data.shape}")
        print(f"  Resolution: {sample_meta['w']}x{sample_meta['h']}")
        print(f"  FPS: {sample_meta['fps']}")
        print(f"  First frame, first 5 keypoints:")
        print(sample_data[0, :5, :])
    
    print("\n" + "="*70)
    print("Next steps:")
    print("1. Verify data: python verify_dataset.py")
    print("2. List subjects: python list_subjects.py")
    print(f"3. Run inference: python run.py -d custom -k {DATASET_NAME}_{format_suffix}{norm_suffix}")
    print("="*70)


if __name__ == '__main__':
    main()
