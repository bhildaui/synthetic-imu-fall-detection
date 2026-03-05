#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk mengkonversi OpenPose JSON output dari URFD dataset
ke format yang dibutuhkan VideoPose3D

Author: Nay (Thesis S2 UI)
Dataset: URFD (Fall Detection)
Target: VideoPose3D 3D Pose Estimation

Justifikasi Script:
- Membaca JSON output dari OpenPose (18 keypoints)
- Convert ke format COCO (17 keypoints) untuk kompatibilitas pretrained model
- Handle missing detections dengan interpolasi linear
- Simpan dalam format NPZ yang dibutuhkan VideoPose3D
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

# Path ke dataset URFD
URFD_BASE = '/workspace/URFD'

# Output directory (folder data/ di VideoPose3D)
OUTPUT_DIR = '/workspace/VideoPose3D/data'

# Pilihan konversi format
# True = Convert OpenPose 18 → COCO 17 (untuk pretrained model)
# False = Tetap OpenPose 18 (untuk training sendiri)
CONVERT_TO_COCO = True

# Pilihan normalisasi koordinat
# True = Normalize ke [-1, 1]
# False = Tetap pixel coordinates (RECOMMENDED)
USE_NORMALIZATION = False

# FPS video URFD (sesuaikan dengan dataset Anda)
VIDEO_FPS = 30

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
    # Justifikasi: URFD fall detection biasanya 1 orang per frame
    person = data['people'][0]
    pose_data = person['pose_keypoints_2d']
    
    # Hitung jumlah keypoints
    num_keypoints = len(pose_data) // 3
    
    # Validasi
    if num_keypoints != 18:
        print(f'Warning in {json_path}: Expected 18 keypoints, got {num_keypoints}')
        # Fallback: tetap proses tapi dengan jumlah yang ada
    
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
    - Diperlukan untuk metadata dan normalisasi (opsional)
    """
    # Folder gambar biasanya parent dari folder json
    # Contoh: /workspace/URFD/fall-07-cam0-rgb-out/json
    #      -> /workspace/URFD/fall-07-cam0-rgb-out/ (parent)
    parent_folder = os.path.dirname(json_folder)
    
    # Cari file JSON pertama
    json_files = sorted(glob.glob(os.path.join(json_folder, '*.json')))
    
    if len(json_files) == 0:
        print("Warning: No JSON files found")
        return 640, 480  # Default URFD
    
    # Baca JSON pertama untuk dapat nama file gambar
    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        if 'image' in data:
            image_name = data['image']
            image_path = os.path.join(parent_folder, image_name)
            
            if os.path.exists(image_path):
                # Baca dimensi dari gambar menggunakan OpenCV
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    print(f'  Image dimensions from file: {width}x{height}')
                    return width, height
    except Exception as e:
        print(f"  Warning: Could not read image file: {e}")
    
    # Fallback: estimasi dari keypoints
    print("  Estimating dimensions from keypoints...")
    for json_file in json_files[:10]:  # Cek 10 file pertama
        keypoints, confidence = load_json_keypoints(json_file)
        if keypoints is not None:
            # Filter koordinat yang valid (> 0)
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            x_coords = x_coords[x_coords > 0]
            y_coords = y_coords[y_coords > 0]
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                # Tambah margin 20%
                width = int(np.max(x_coords) * 1.2)
                height = int(np.max(y_coords) * 1.2)
                print(f'  Estimated dimensions: {width}x{height}')
                return width, height
    
    # Last resort: default
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
        
    Justifikasi:
    - Normalisasi membuat model lebih robust
    - Range [-1,1] centered, cocok untuk neural networks
    - OPSIONAL: bisa skip jika pakai pixel coordinates
    """
    norm_kp = keypoints.copy()
    
    # Normalisasi ke [-1, 1]
    # x: [0, width] -> [-1, 1]
    # y: [0, height] -> [-1, 1]
    norm_kp[:, :, 0] = (keypoints[:, :, 0] / width) * 2 - 1
    norm_kp[:, :, 1] = (keypoints[:, :, 1] / height) * 2 - 1
    
    return norm_kp


def convert_openpose_to_coco(keypoints_18):
    """
    Convert OpenPose 18 keypoints ke COCO 17 keypoints
    
    Args:
        keypoints_18: array [num_frames, 18, 2] OpenPose format
        
    Returns:
        keypoints_17: array [num_frames, 17, 2] COCO format
        
    Justifikasi:
    - Pretrained VideoPose3D model menggunakan COCO 17 keypoints
    - OpenPose punya Neck (index 1), COCO tidak
    - Mapping berdasarkan semantic correspondence
    
    OpenPose 18 keypoints order:
    0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist,
    5:LShoulder, 6:LElbow, 7:LWrist, 8:RHip, 9:RKnee,
    10:RAnkle, 11:LHip, 12:LKnee, 13:LAnkle, 14:REye,
    15:LEye, 16:REar, 17:LEar
    
    COCO 17 keypoints order:
    0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder,
    6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist,
    11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
    """
    # Mapping: COCO index -> OpenPose index
    # Justifikasi mapping: berdasarkan anatomical correspondence
    mapping = [
        0,   # 0: Nose -> Nose
        15,  # 1: LEye -> LEye
        14,  # 2: REye -> REye
        17,  # 3: LEar -> LEar
        16,  # 4: REar -> REar
        5,   # 5: LShoulder -> LShoulder
        2,   # 6: RShoulder -> RShoulder
        6,   # 7: LElbow -> LElbow
        3,   # 8: RElbow -> RElbow
        7,   # 9: LWrist -> LWrist
        4,   # 10: RWrist -> RWrist
        11,  # 11: LHip -> LHip
        8,   # 12: RHip -> RHip
        12,  # 13: LKnee -> LKnee
        9,   # 14: RKnee -> RKnee
        13,  # 15: LAnkle -> LAnkle
        10,  # 16: RAnkle -> RAnkle
    ]
    
    # Apply mapping
    keypoints_17 = keypoints_18[:, mapping, :]
    
    return keypoints_17


def interpolate_missing_frames(keypoints, valid_indices):
    """
    Interpolasi linear untuk frame yang missing
    
    Args:
        keypoints: array [num_frames, N, 2] dengan NaN pada missing frames
        valid_indices: list of frame indices yang punya deteksi valid
        
    Returns:
        interpolated_keypoints: array dengan NaN sudah diisi
        
    Justifikasi:
    - Linear interpolation sederhana tapi efektif
    - Menjaga continuity of motion
    - Interpolate per joint per dimensi untuk akurasi
    """
    num_frames = len(keypoints)
    num_joints = keypoints.shape[1]
    indices = np.arange(num_frames)
    
    # Interpolate untuk setiap joint dan setiap dimensi (x, y)
    for joint_idx in range(num_joints):
        for dim in range(2):  # 0=x, 1=y
            # Identifikasi nilai valid (bukan NaN)
            valid_mask = ~np.isnan(keypoints[valid_indices, joint_idx, dim])
            valid_idx_filtered = [valid_indices[i] for i in range(len(valid_indices)) 
                                  if valid_mask[i]]
            
            if len(valid_idx_filtered) == 0:
                # Tidak ada nilai valid untuk joint ini
                # Set ke 0 atau bisa di-skip
                keypoints[:, joint_idx, dim] = 0
                continue
            
            if len(valid_idx_filtered) == 1:
                # Hanya 1 nilai valid, propagate ke semua frame
                keypoints[:, joint_idx, dim] = keypoints[valid_idx_filtered[0], joint_idx, dim]
                continue
            
            # Interpolate linear
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

def process_video_sequence(json_folder, convert_to_coco=True, normalize=False):
    """
    Proses semua JSON files dari satu video sequence
    
    Args:
        json_folder: folder yang berisi file-file JSON
        convert_to_coco: apakah convert OpenPose 18 → COCO 17
        normalize: apakah normalisasi coordinates
        
    Returns:
        keypoints_sequence: array [num_frames, N, 2] dimana N=17 atau 18
        metadata: informasi tambahan tentang video
        
    Workflow:
    1. Load semua JSON files (sorted by frame number)
    2. Extract keypoints dari setiap frame
    3. Handle missing detections (set NaN)
    4. Interpolasi frame yang missing
    5. (Optional) Convert 18→17 keypoints
    6. (Optional) Normalize coordinates
    7. Return array + metadata
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
            # Tidak ada deteksi → NaN
            all_keypoints.append(np.full((18, 2), np.nan, dtype=np.float32))
        else:
            # Handle keypoints dengan confidence = 0
            # Justifikasi: OpenPose set (0,0,0) untuk keypoints tidak terdeteksi
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
    
    # Convert OpenPose 18 → COCO 17 (jika diminta)
    if convert_to_coco:
        print('  Converting OpenPose 18 → COCO 17 keypoints')
        keypoints_array = convert_openpose_to_coco(keypoints_array)
    
    # Normalisasi (jika diminta)
    if normalize:
        print('  Normalizing coordinates to [-1, 1]')
        keypoints_array = normalize_keypoints(keypoints_array, width, height)
    
    # Metadata
    metadata = {
        'w': width,       # Width (key 'w' untuk VideoPose3D)
        'h': height,      # Height (key 'h' untuk VideoPose3D)
        'fps': VIDEO_FPS,
    }
    
    return keypoints_array, metadata


def main():
    """
    Main function untuk proses semua video URFD
    
    Workflow:
    1. Scan semua folder *-out/json di URFD directory
    2. Proses setiap video sequence
    3. Simpan dalam format NPZ untuk VideoPose3D
    
    Output structure:
    {
        'positions_2d': {
            'video_name_1': {'custom': [keypoints_array]},
            'video_name_2': {'custom': [keypoints_array]},
            ...
        },
        'metadata': {
            'layout_name': 'coco' atau 'openpose',
            'num_joints': 17 atau 18,
            'keypoints_symmetry': [...],
            'video_metadata': {
                'video_name_1': {w, h, fps},
                ...
            }
        }
    }
    """
    print("="*70)
    print("URFD Dataset Preparation for VideoPose3D")
    print("="*70)
    print(f"URFD base directory: {URFD_BASE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Convert to COCO: {CONVERT_TO_COCO}")
    print(f"Use normalization: {USE_NORMALIZATION}")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Dictionary untuk menyimpan semua video
    output = {}
    
    # Metadata global
    if CONVERT_TO_COCO:
        # COCO 17 keypoints
        metadata_all = {
            'layout_name': 'coco',
            'num_joints': 17,
            'keypoints_symmetry': [
                [1, 3, 5, 7, 9, 11, 13, 15],   # Left side
                [2, 4, 6, 8, 10, 12, 14, 16]   # Right side
            ],
            'video_metadata': {}
        }
    else:
        # OpenPose 18 keypoints
        metadata_all = {
            'layout_name': 'openpose',
            'num_joints': 18,
            'keypoints_symmetry': [
                [1, 2, 3, 4, 8, 9, 10, 15, 17],      # Left side
                [1, 5, 6, 7, 11, 12, 13, 14, 16]     # Right side (Neck shared)
            ],
            'video_metadata': {}
        }
    
    # Cari semua folder yang berisi JSON outputs
    # Pattern: *-out/json
    video_folders = sorted(glob.glob(os.path.join(URFD_BASE, '*-out/json')))
    
    if len(video_folders) == 0:
        print(f"ERROR: No video folders found in {URFD_BASE}")
        print("Expected pattern: <URFD_BASE>/*-out/json")
        return
    
    print(f"\nFound {len(video_folders)} video folders to process\n")
    
    # Proses setiap video
    for video_folder in video_folders:
        # Extract nama video
        # Contoh: /workspace/URFD/fall-07-cam0-rgb-out/json
        #      -> fall-07-cam0-rgb
        video_name = os.path.basename(os.path.dirname(video_folder)).replace('-out', '')
        
        print(f"Processing: {video_name}")
        
        # Proses video sequence
        keypoints, video_metadata = process_video_sequence(
            video_folder,
            convert_to_coco=CONVERT_TO_COCO,
            normalize=USE_NORMALIZATION
        )
        
        if keypoints is None:
            print(f"  SKIPPED: {video_name} - no valid detections\n")
            continue
        
        # Format output sesuai VideoPose3D
        # Structure: output[video_name]['subject_name'] = [keypoints]
        # 'custom' adalah nama subject default
        output[video_name] = {
            'custom': [keypoints]
        }
        
        metadata_all['video_metadata'][video_name] = video_metadata
        print(f"  SUCCESS: Shape {keypoints.shape}")
        print("-"*70)
    
    # Simpan hasil
    if CONVERT_TO_COCO:
        format_suffix = 'coco'
    else:
        format_suffix = 'openpose'
    
    norm_suffix = '_normalized' if USE_NORMALIZATION else '_pixel'
    
    output_filename = os.path.join(
        OUTPUT_DIR,
        f'data_2d_custom_urfd_{format_suffix}{norm_suffix}.npz'
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
        print(f"  First frame, first 3 keypoints:")
        print(sample_data[0, :3, :])
    
    print("\n" + "="*70)
    print("Next steps:")
    print("1. Verify data: python verify_dataset.py")
    print("2. List subjects: python list_subjects.py")
    print("3. Run inference: python run.py -d custom -k urfd_...[TAB] -str <subject> -ste <subject>")
    print("="*70)


if __name__ == '__main__':
    main()
