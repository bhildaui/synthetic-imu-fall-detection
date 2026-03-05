import json
import numpy as np
from pathlib import Path

# Path settings
JSON_DIR = Path("/workspace/awal/json")
OUTPUT_NPZ = "/workspace/awal/data_2d_custom_urfd_openpose.npz"

# Ukuran video URFD (dari hasil ffmpeg tadi: 640x240)
VIDEO_W = 640
VIDEO_H = 240

# COCO memiliki 18 keypoints
NUM_KEYPOINTS = 18

# Mapping dari OpenPose COCO ke Detectron COCO (format VideoPose3D)
# openpose_idx -> detectron_idx
OPENPOSE_TO_DETECTRON = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
# 18 tidak ada di OpenPose, akan diisi NaN (LEar kiri)

def parse_openpose_json(json_folder):
    """
    Baca semua frame JSON dari satu video dan return array keypoints.
    
    Format output: array shape (num_frames, 17, 2)
    Kenapa 17? VideoPose3D pakai 17 keypoints (COCO tanpa keypoint ke-0 yaitu 'neck')
    """
    json_files = sorted(json_folder.glob("*.json"))
    keypoints_all = []

    for jf in json_files:
        data = json.load(open(jf))
        
        if len(data['people']) == 0:
            # Tidak ada orang terdeteksi, isi dengan NaN
            kp = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
        else:
            # Ambil orang dengan confidence tertinggi
            best_person = max(data['people'], 
                            key=lambda p: np.mean([p['pose_keypoints_2d'][i*3+2] 
                                                   for i in range(NUM_KEYPOINTS)]))
            
            flat = best_person['pose_keypoints_2d']
            # Reshape dari flat [x,y,c, x,y,c, ...] menjadi (18, 3)
            kp = np.array(flat).reshape(NUM_KEYPOINTS, 3).astype(np.float32)
        
        keypoints_all.append(kp)
    
    keypoints_all = np.array(keypoints_all)  # shape: (frames, 18, 3)
    
    # Interpolasi frame yang NaN
    mask = ~np.isnan(keypoints_all[:, 0, 0])
    if mask.sum() == 0:
        return None  # Semua frame kosong, skip video ini
    
    indices = np.arange(len(keypoints_all))
    for i in range(NUM_KEYPOINTS):
        for j in range(2):  # x dan y saja
            keypoints_all[:, i, j] = np.interp(
                indices, indices[mask], keypoints_all[mask, i, j]
            )
    
    # Mapping OpenPose index -> Detectron index
    # Detectron: [Nose, LEye, REye, LEar, REar, LShoulder, RShoulder,
    #             LElbow, RElbow, LWrist, RWrist, LHip, RHip,
    #             LKnee, RKnee, LAnkle, RAnkle]
    # OpenPose:  [Nose=0, LEye=16, REye=15, LEar=-1, REar=17,
    #             LShoulder=5, RShoulder=2, LElbow=6, RElbow=3,
    #             LWrist=7, RWrist=4, LHip=12, RHip=9,
    #             LKnee=13, RKnee=10, LAnkle=14, RAnkle=11]
    openpose_to_detectron = [0, 16, 15, -1, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
    
    kp_remapped = np.zeros((len(keypoints_all), 17, 2), dtype=np.float32)
    for det_idx, op_idx in enumerate(openpose_to_detectron):
        if op_idx >= 0:
            kp_remapped[:, det_idx, :] = keypoints_all[:, op_idx, :2]
        # op_idx == -1 berarti tidak ada di OpenPose, biarkan 0
    return kp_remapped

# Proses semua video cam0 saja
positions_2d = {}  # Dictionary untuk menyimpan semua video

# Ambil hanya cam0
cam0_folders = sorted([f for f in JSON_DIR.iterdir() if 'cam0' in f.name])
print(f"Total cam0 videos: {len(cam0_folders)}")

for folder in cam0_folders:
    print(f"Processing {folder.name}...")
    kp = parse_openpose_json(folder)
    
    if kp is None:
        print(f"  Skipping {folder.name} (no detections)")
        continue
    
    # Format untuk VideoPose3D: {'S1': {'video_name': kp_array}}
    # Kita pakai nama video sebagai subject
    subject = folder.name  # contoh: fall-01-cam0
    positions_2d[subject] = {'custom': [kp]}  # dibungkus list
    print(f"  Shape: {kp.shape}")

# Simpan ke NPZ
# Buat video_metadata berisi resolusi tiap video
video_metadata = {}
for subject in positions_2d.keys():
    video_metadata[subject] = {'w': VIDEO_W, 'h': VIDEO_H}

metadata = {
    'layout_name': 'coco',
    'num_joints': 17,  
    'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]],
    'video_metadata': video_metadata
}

np.savez_compressed(OUTPUT_NPZ,
                    positions_2d=positions_2d,
                    metadata=metadata)
print(f"\nSaved to {OUTPUT_NPZ}")

print(f"Total videos processed: {len(positions_2d)}")

