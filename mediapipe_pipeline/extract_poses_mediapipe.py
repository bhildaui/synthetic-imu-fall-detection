import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path

# Path settings
VIDEO_DIR = Path("/workspace/URFD")
OUTPUT_DIR = Path("/workspace/mediapipe_pipeline/json")
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose

# MediaPipe 33 keypoints, kita ambil 17 yang sesuai dengan Detectron COCO
# untuk kompatibilitas dengan VideoPose3D
# MediaPipe index -> nama
# 0=nose, 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow
# 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip
# 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle
# 1=left_eye_inner, 2=left_eye, 5=right_eye, 7=left_ear, 8=right_ear

# Mapping MediaPipe -> Detectron COCO 17 keypoints
# Detectron: Nose, LEye, REye, LEar, REar, LShoulder, RShoulder,
#            LElbow, RElbow, LWrist, RWrist, LHip, RHip,
#            LKnee, RKnee, LAnkle, RAnkle
MEDIAPIPE_TO_DETECTRON = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def process_video(video_path, output_dir):
    """Extract 2D keypoints dari video menggunakan MediaPipe"""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir.mkdir(exist_ok=True)
    
    frame_idx = 0
    detected = 0
    
    with mp_pose.Pose(
        static_image_mode=False,      # video mode, lebih efisien
        model_complexity=2,            # 0=cepat, 1=medium, 2=akurat
        smooth_landmarks=True,         # smooth antar frame
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe butuh RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # Simpan keypoints
            keypoints = []
            if results.pose_landmarks:
                detected += 1
                for idx in MEDIAPIPE_TO_DETECTRON:
                    lm = results.pose_landmarks.landmark[idx]
                    # Konversi dari normalized (0-1) ke pixel
                    x = lm.x * width
                    y = lm.y * height
                    conf = lm.visibility
                    keypoints.extend([x, y, conf])
            else:
                # Tidak ada deteksi, isi 0
                keypoints = [0.0] * (17 * 3)
            
            # Simpan ke JSON (format sama dengan OpenPose)
            output_file = output_dir / f"frame_{frame_idx:04d}_keypoints.json"
            json_data = {
                "version": 1.3,
                "people": [{
                    "pose_keypoints_2d": keypoints
                }] if results.pose_landmarks else []
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_data, f)
            
            frame_idx += 1
    
    cap.release()
    detection_rate = detected / frame_idx * 100 if frame_idx > 0 else 0
    return detection_rate

# Proses semua video (cam0 dan cam1)
all_videos = sorted(VIDEO_DIR.glob("*.mp4"))
print(f"Total videos: {len(all_videos)}")

for i, video in enumerate(all_videos):
    video_name = video.stem
    output_dir = OUTPUT_DIR / video_name
    
    if output_dir.exists() and len(list(output_dir.glob("*.json"))) > 0:
        print(f"[{i+1}/{len(all_videos)}] Skipping {video_name} (already done)")
        continue
    
    print(f"[{i+1}/{len(all_videos)}] Processing {video_name}...")
    rate = process_video(video, output_dir)
    print(f"  Detection rate: {rate:.1f}%")

print("\nDone!")
