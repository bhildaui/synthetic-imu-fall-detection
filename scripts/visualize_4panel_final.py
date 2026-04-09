"""
Video 4-panel final dengan global trajectory:
Atas kiri  : OpenPose 2D pose
Atas kanan : MediaPipe 2D pose
Bawah kiri : OpenPose 3D (root-relative - original)
Bawah kanan: MediaPipe 3D + Global Trajectory (baru)
"""

import cv2
import json
import numpy as np
from pathlib import Path

RGB_DIR     = Path("/workspace/URFD/rgb")
OP_JSON_DIR = Path("/workspace/awal/json")
MP_JSON_DIR = Path("/workspace/mediapipe_pipeline/json")
OP_3D_DIR   = Path("/workspace/awal/3d_poses")
MP_TRAJ_DIR = Path("/workspace/mediapipe_pipeline/visualisasi")
OUT_DIR     = Path("/workspace/mediapipe_pipeline/visualisasi")

SKELETON_OP = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (0,14),(14,16),(0,15),(15,17),
]
SKELETON_MP = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
COLORS = {'head': (255,200,0), 'upper': (0,200,255), 'lower': (0,255,100)}

def get_color_op(i,j):
    head={0,14,15,16,17}; lower={8,9,10,11,12,13}
    if i in head or j in head: return COLORS['head']
    if i in lower or j in lower: return COLORS['lower']
    return COLORS['upper']

def get_color_mp(i,j):
    head={0,1,2,3,4}; lower={11,12,13,14,15,16}
    if i in head or j in head: return COLORS['head']
    if i in lower or j in lower: return COLORS['lower']
    return COLORS['upper']

def draw_op(frame, kp_flat):
    kp = np.array(kp_flat).reshape(18,3)
    for i,j in SKELETON_OP:
        if kp[i,2]>0.3 and kp[j,2]>0.3:
            cv2.line(frame,(int(kp[i,0]),int(kp[i,1])),(int(kp[j,0]),int(kp[j,1])),get_color_op(i,j),2,cv2.LINE_AA)
    for idx in range(18):
        if kp[idx,2]>0.3:
            cv2.circle(frame,(int(kp[idx,0]),int(kp[idx,1])),4,(255,255,255),-1,cv2.LINE_AA)
    return frame

def draw_mp(frame, kp_flat):
    kp = np.array(kp_flat).reshape(17,3)
    for i,j in SKELETON_MP:
        if kp[i,2]>0.3 and kp[j,2]>0.3:
            cv2.line(frame,(int(kp[i,0]),int(kp[i,1])),(int(kp[j,0]),int(kp[j,1])),get_color_mp(i,j),2,cv2.LINE_AA)
    for idx in range(17):
        if kp[idx,2]>0.3:
            cv2.circle(frame,(int(kp[idx,0]),int(kp[idx,1])),4,(255,255,255),-1,cv2.LINE_AA)
    return frame

def add_label(frame, text, color=(255,255,255), pos=(10,28)):
    cv2.putText(frame,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,text,pos,cv2.FONT_HERSHEY_SIMPLEX,0.6,color,1,cv2.LINE_AA)
    return frame

def create_4panel(video_name, label):
    print(f"Processing {video_name}...")

    frame_dir    = RGB_DIR / video_name / video_name
    op_json_dir  = OP_JSON_DIR / video_name
    mp_json_dir  = MP_JSON_DIR / video_name
    op_3d_mp4    = OP_3D_DIR / f"{video_name}.mp4"
    mp_traj_mp4  = MP_TRAJ_DIR / f"3d_trajectory_{video_name}.mp4"
    output_mp4   = OUT_DIR / f"4panel_final_{video_name}.mp4"

    frame_files   = sorted(frame_dir.glob("*.png"))
    op_json_files = sorted(op_json_dir.glob("*_keypoints.json"))
    mp_json_files = sorted(mp_json_dir.glob("*_keypoints.json"))

    cap_op3d   = cv2.VideoCapture(str(op_3d_mp4))
    cap_mp_traj = cv2.VideoCapture(str(mp_traj_mp4))

    PW, PH = 640, 480
    n = min(len(frame_files), len(op_json_files), len(mp_json_files))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(str(output_mp4), fourcc, 30, (PW*2, PH*2))

    color_label = (0,220,100) if 'adl' in video_name else (0,80,255)

    for i in range(n):
        # Panel 1 (atas kiri): OpenPose 2D
        f1 = cv2.imread(str(frame_files[i]))
        f1 = cv2.resize(f1, (PW, PH))
        op_data = json.loads(op_json_files[i].read_text())
        if len(op_data['people']) > 0:
            draw_op(f1, op_data['people'][0]['pose_keypoints_2d'])
        add_label(f1, "OpenPose - 2D Pose", color=(100,200,255))

        # Panel 2 (atas kanan): MediaPipe 2D
        f2 = cv2.imread(str(frame_files[i]))
        f2 = cv2.resize(f2, (PW, PH))
        mp_data = json.loads(mp_json_files[i].read_text())
        if len(mp_data['people']) > 0:
            draw_mp(f2, mp_data['people'][0]['pose_keypoints_2d'])
        add_label(f2, "MediaPipe - 2D Pose", color=(100,255,150))

        # Panel 3 (bawah kiri): OpenPose 3D root-relative
        ret3, f3 = cap_op3d.read()
        if not ret3: f3 = np.zeros((PH,PW,3),dtype=np.uint8)
        f3 = cv2.resize(f3, (PW, PH))
        add_label(f3, "OpenPose - 3D Pose (root-relative)", color=(100,200,255))

        # Panel 4 (bawah kanan): MediaPipe 3D + trajectory
        ret4, f4 = cap_mp_traj.read()
        if not ret4: f4 = np.zeros((PH,PW,3),dtype=np.uint8)
        f4 = cv2.resize(f4, (PW, PH))
        add_label(f4, "MediaPipe - 3D + Global Trajectory", color=(100,255,150))

        # Gabungkan
        top    = np.hstack([f1, f2])
        bottom = np.hstack([f3, f4])
        canvas = np.vstack([top, bottom])

        # Label tengah
        cv2.putText(canvas, f"{label} | Frame {i+1}/{n}",
                    (PW-180, PH*2-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_label, 2, cv2.LINE_AA)

        out.write(canvas)

    out.release()
    cap_op3d.release()
    cap_mp_traj.release()
    print(f"  Saved: {output_mp4.name}")

print("Membuat video 4-panel final...")
print("=" * 50)

for video_name, label in [
    ("adl-01-cam0-rgb", "ADL - Aktivitas Normal"),
    ("fall-06-cam0-rgb", "FALL - Jatuh"),
]:
    create_4panel(video_name, label)

print("\nSelesai! File:")
print("  4panel_final_adl-01-cam0-rgb.mp4")
print("  4panel_final_fall-06-cam0-rgb.mp4")
