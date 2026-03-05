# prepare_urfd_dataset.py
import numpy as np
import json
import os
import glob
from collections import defaultdict

def load_json_keypoints(json_path):
    """
    Membaca file JSON OpenPose dan ekstrak keypoints 2D
    
    Args:
        json_path: path ke file JSON OpenPose
        
    Returns:
        keypoints: array [18, 2] berisi koordinat (x,y) dalam pixels
        confidence: array [18] berisi confidence scores
        
    Justifikasi:
    - OpenPose format: 18 keypoints (BODY_25 simplified atau COCO+Neck)
    - pose_keypoints_2d berisi [x1,y1,c1, x2,y2,c2, ...] = 18*3 = 54 values
    - Kita kembalikan dalam pixel coordinates (belum dinormalisasi)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Cek apakah ada orang terdeteksi
    if data['num_people'] == 0:
        return None, None
    
    # Ambil orang pertama (index 0)
    # Justifikasi: URFD biasanya 1 orang per frame (single subject fall detection)
    person = data['people'][0]
    pose_data = person['pose_keypoints_2d']
    
    # Hitung jumlah keypoints dari panjang array
    # Justifikasi: len(pose_data) = num_keypoints * 3
    num_keypoints = len(pose_data) // 3
    
    # Validasi bahwa ini memang 18 keypoints
    if num_keypoints != 18:
        print(f'Warning: Expected 18 keypoints, got {num_keypoints}')
    
    # Reshape dari [54] menjadi [18, 3]
    # Struktur: [[x1,y1,conf1], [x2,y2,conf2], ...]
    pose_array = np.array(pose_data).reshape(num_keypoints, 3)
    
    keypoints = pose_array[:, :2]  # Kolom 0,1 = x,y coordinates
    confidence = pose_array[:, 2]   # Kolom 2 = confidence score
    
    return keypoints, confidence

def get_image_dimensions(json_folder):
    """
    Mendapatkan dimensi gambar dari file gambar asli
    
    Args:
        json_folder: folder berisi JSON files (biasanya /path/to/video-out/json)
        
    Returns:
        width, height: dimensi gambar dalam pixels
        
    Justifikasi:
    - Lebih akurat daripada estimasi dari keypoints
    - Baca langsung dari file PNG yang disebutkan di JSON
    - Fallback ke estimasi jika file tidak ditemukan
    """
    import cv2  # Tambahkan import di atas script
    
    # Folder gambar biasanya sejajar dengan folder json
    # Contoh: /workspace/URFD/fall-07-cam0-rgb-out/json
    #      -> /workspace/URFD/fall-07-cam0-rgb-out/ (parent)
    parent_folder = os.path.dirname(json_folder)
    
    # Cari file JSON pertama untuk tahu nama file gambar
    json_files = sorted(glob.glob(os.path.join(json_folder, '*.json')))
    
    if len(json_files) == 0:
        print("Warning: No JSON files found")
        return 640, 480  # Default URFD
    
    # Baca JSON pertama untuk dapat nama file gambar
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    if 'image' in data:
        # Coba cari file gambar di parent folder
        image_name = data['image']
        image_path = os.path.join(parent_folder, image_name)
        
        if os.path.exists(image_path):
            # Baca dimensi dari gambar
            img = cv2.imread(image_path)
            if img is not None:
                height, width = img.shape[:2]
                print(f'Image dimensions from file: {width}x{height}')
                return width, height
    
    # Fallback: estimasi dari keypoints (metode lama)
    print("Warning: Could not read image file, estimating from keypoints")
    for json_file in json_files[:10]:  # Cek 10 file pertama
        keypoints, confidence = load_json_keypoints(json_file)
        if keypoints is not None:
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            
            x_coords = x_coords[x_coords > 0]
            y_coords = y_coords[y_coords > 0]
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                width = int(np.max(x_coords) * 1.2)
                height = int(np.max(y_coords) * 1.2)
                print(f'Image dimensions estimated: {width}x{height}')
                return width, height
    
    # Last resort: default URFD resolution
    print("Warning: Using default resolution 640x480")
    return 640, 480

def normalize_keypoints(keypoints, width, height):
    """
    Normalisasi koordinat pixel ke range [-1, 1] atau [0, 1]
    
    Args:
        keypoints: array [num_frames, 18, 2] dalam pixels
        width, height: dimensi gambar
        
    Returns:
        normalized_keypoints: array dengan koordinat ternormalisasi
        
    Justifikasi:
    - Normalisasi membuat model lebih robust terhadap resolusi berbeda
    - Range [-1,1] centered, cocok untuk neural networks
    - VideoPose3D sebenarnya bisa terima pixel atau normalized
    
    CATATAN: Fungsi ini OPTIONAL, bisa di-skip jika mau pakai pixel langsung
    """
    norm_kp = keypoints.copy()
    
    # Normalisasi ke range [-1, 1] dengan center di (0, 0)
    # x: dari [0, width] -> [-1, 1]
    # y: dari [0, height] -> [-1, 1]
    norm_kp[:, :, 0] = (keypoints[:, :, 0] / width) * 2 - 1   # x coordinate
    norm_kp[:, :, 1] = (keypoints[:, :, 1] / height) * 2 - 1  # y coordinate
    
    return norm_kp

def process_video_sequence(json_folder, normalize=False):
    """
    Proses semua JSON files dari satu video sequence
    
    Args:
        json_folder: folder yang berisi file-file JSON
        normalize: apakah mau normalisasi coordinates (True/False)
        
    Returns:
        keypoints_sequence: array [num_frames, 18, 2]
        metadata: informasi tambahan tentang video
        
    PERBAIKAN: Sesuaikan key metadata dengan yang diharapkan VideoPose3D
    """
    # Cari semua file JSON dan urutkan berdasarkan nomor frame
    json_files = sorted(
        glob.glob(os.path.join(json_folder, '*.json')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[-1])
    )
    
    print(f'Found {len(json_files)} JSON files in {os.path.basename(os.path.dirname(json_folder))}')
    
    # Deteksi dimensi gambar
    width, height = get_image_dimensions(json_folder)
    print(f'Image dimensions: {width}x{height}')
    
    all_keypoints = []
    valid_indices = []
    
    # Proses setiap frame
    for idx, json_file in enumerate(json_files):
        keypoints, confidence = load_json_keypoints(json_file)
        
        if keypoints is None:
            all_keypoints.append(np.full((18, 2), np.nan, dtype=np.float32))
        else:
            # Handle keypoints dengan confidence = 0
            for i in range(18):
                if confidence[i] == 0.0:
                    keypoints[i] = [np.nan, np.nan]
            
            all_keypoints.append(keypoints.astype(np.float32))
            valid_indices.append(idx)
    
    # Convert ke numpy array
    keypoints_array = np.array(all_keypoints, dtype=np.float32)
    
    # Interpolasi untuk frame yang missing
    if len(valid_indices) > 0:
        keypoints_array = interpolate_missing_frames(
            keypoints_array, 
            valid_indices
        )
        num_interpolated = len(json_files) - len(valid_indices)
        print(f'{len(json_files)} total frames processed')
        print(f'{num_interpolated} frames were interpolated')
    else:
        print('WARNING: No valid detections found!')
        return None, None
    
    # Optional: Normalisasi coordinates
    if normalize:
        print('Normalizing coordinates to [-1, 1] range')
        keypoints_array = normalize_keypoints(keypoints_array, width, height)
    
    # PERBAIKAN: Metadata dengan key yang sesuai VideoPose3D
    # VideoPose3D expect: 'w', 'h', bukan 'width', 'height'
    metadata = {
        'w': width,      # PERUBAHAN: 'width' -> 'w'
        'h': height,     # PERUBAHAN: 'height' -> 'h'
        'fps': 30,       # Frame per second dari URFD
    }
    
    return keypoints_array, metadata

def interpolate_missing_frames(keypoints, valid_indices):
    """
    Interpolasi linear untuk frame yang missing
    
    Args:
        keypoints: array [num_frames, 18, 2] dengan NaN pada missing frames
        valid_indices: list of frame indices yang punya deteksi valid
        
    Returns:
        interpolated_keypoints: array dengan NaN sudah diisi
    """
    num_frames = len(keypoints)
    indices = np.arange(num_frames)
    
    # Interpolate untuk setiap keypoint (18) dan setiap dimensi (x,y)
    for joint_idx in range(18):  # PERUBAHAN: 17 -> 18
        for dim in range(2):
            # Ambil nilai valid (bukan NaN)
            valid_mask = ~np.isnan(keypoints[valid_indices, joint_idx, dim])
            valid_idx_filtered = [valid_indices[i] for i in range(len(valid_indices)) if valid_mask[i]]
            
            if len(valid_idx_filtered) == 0:
                # Jika tidak ada nilai valid sama sekali untuk joint ini
                # Set ke 0 (atau bisa skip)
                keypoints[:, joint_idx, dim] = 0
                continue
            
            valid_values = keypoints[valid_idx_filtered, joint_idx, dim]
            
            # Interpolate
            keypoints[:, joint_idx, dim] = np.interp(
                indices,
                valid_idx_filtered,
                valid_values
            )
    
    return keypoints

def main():
    """
    Main function untuk proses semua video URFD
    """
    # Konfigurasi
    urfd_base = '/workspace/URFD'
    output_dir = '/workspace/VideoPose3D/data'
    os.makedirs(output_dir, exist_ok=True)
    
    # PILIHAN: normalize atau tidak
    # Justifikasi pemilihan:
    # - normalize=False: lebih sederhana, VideoPose3D bisa handle pixel coords
    # - normalize=True: jika mau pakai pretrained model yang expect normalized
    USE_NORMALIZATION = False
    
    # Dictionary untuk menyimpan semua video
    output = {}
    
    # Metadata global untuk VideoPose3D
    # PENTING: Sesuaikan dengan OpenPose 18 keypoints
    metadata_all = {
        'layout_name': 'openpose',  # PERUBAHAN: coco -> openpose
        'num_joints': 18,            # PERUBAHAN: 17 -> 18
        'keypoints_symmetry': [      # Symmetry untuk OpenPose
            # Left side indices
            [1, 2, 3, 4, 8, 9, 10],   # Left eye, ear, shoulder, elbow, wrist, hip, knee, ankle
            # Right side indices  
            [14, 15, 16, 17, 11, 12, 13]  # Right eye, ear, shoulder, elbow, wrist, hip, knee, ankle
        ],
        'video_metadata': {}
    }
    
    # Cari semua folder JSON
    video_folders = sorted(glob.glob(os.path.join(urfd_base, '*-out/json')))
    
    print(f'Found {len(video_folders)} video folders to process')
    print(f'Normalization: {USE_NORMALIZATION}')
    print('='*60)
    
    for video_folder in video_folders:
        # Extract nama video
        video_name = os.path.basename(os.path.dirname(video_folder)).replace('-out', '')
        
        print(f'\nProcessing: {video_name}')
        
        # Proses video sequence
        keypoints, video_metadata = process_video_sequence(
            video_folder, 
            normalize=USE_NORMALIZATION
        )
        
        if keypoints is None:
            print(f'Skipping {video_name} - no valid detections')
            continue
        
        # Format output sesuai VideoPose3D
        output[video_name] = {
            'custom': [keypoints]
        }
        
        metadata_all['video_metadata'][video_name] = video_metadata
        print('-'*60)
    
    # Simpan hasil
    norm_suffix = '_normalized' if USE_NORMALIZATION else '_pixel'
    output_filename = os.path.join(
        output_dir, 
        f'data_2d_custom_urfd_openpose{norm_suffix}.npz'
    )
    
    print(f'\nSaving to {output_filename}...')
    
    np.savez_compressed(
        output_filename,
        positions_2d=output,
        metadata=metadata_all
    )
    
    print('Done!')
    print(f'\nSummary:')
    print(f'- Processed {len(output)} videos')
    print(f'- Keypoints: 18 (OpenPose format)')
    print(f'- Coordinates: {"Normalized [-1,1]" if USE_NORMALIZATION else "Pixel coordinates"}')
    print(f'- Output file: {output_filename}')
    
    # Tampilkan sample data untuk verifikasi
    sample_video = list(output.keys())[0]
    sample_data = output[sample_video]['custom'][0]
    print(f'\nSample data from {sample_video}:')
    print(f'Shape: {sample_data.shape}')  # Should be [num_frames, 18, 2]
    print(f'First frame, first 3 keypoints:')
    print(sample_data[0, :3, :])

if __name__ == '__main__':
    main()