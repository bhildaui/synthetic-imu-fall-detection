"""
Video perbandingan 4 panel:
Atas kiri  : OpenPose 2D pose
Atas kanan : OpenPose 3D pose (dari VideoPose3D)
Bawah kiri : MediaPipe 2D pose  
Bawah kanan: MediaPipe 3D pose (dari VideoPose3D)
"""

import cv2
import json
import numpy as np
from pathlib import Path

RGB_DIR     = Path("/workspace/URFD/rgb")
OP_JSON_DIR = Path("/workspace/awal/json")
MP_JSON_DIR = Path("/workspace/mediapipe_pipeline/json")
OP_3D_DIR   = Path("/workspace/awal/3d_poses")
MP_3D_DIR   = Path("/workspace/mediapipe_pipeline/3d_poses")
OUT_DIR     = Path("/workspace/mediapipe_pipeline/visualisasi")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# SKELETON DEFINITIONS
# ============================================================
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

COLORS = {
    'head':  (255, 200, 0),
    'upper': (0, 200, 255),
    'lower': (0, 255, 100),
}

def get_color_op(i, j):
    head  = {0,14,15,16,17}
    lower = {8,9,10,11,12,13}
    if i in head or j in head: return COLORS['head']
    if i in lower or j in lower: return COLORS['lower']
    return COLORS['upper']

def get_color_mp(i, j):
    head  = {0,1,2,3,4}
    lower = {11,12,13,14,15,16}
    if i in head or j in head: return COLORS['head']
    if i in lower or j in lower: return COLORS['lower']
    return COLORS['upper']

def draw_skeleton(frame, kp_flat, n_kp, skeleton, get_color_fn, conf_threshold=0.3):
    kp = np.array(kp_flat).reshape(n_kp, 3)
    for i, j in skeleton:
        if kp[i,2] > conf_threshold and kp[j,2] > conf_threshold:
            pt1 = (int(kp[i,0]), int(kp[i,1]))
            pt2 = (int(kp[j,0]), int(kp[j,1]))
            cv2.line(frame, pt1, pt2, get_color_fn(i,j), 2, cv2.LINE_AA)
    for idx in range(n_kp):
        if kp[idx,2] > conf_threshold:
            pt = (int(kp[idx,0]), int(kp[idx,1]))
            cv2.circle(frame, pt, 4, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, get_color_fn(idx,idx), 1, cv2.LINE_AA)
    return frame

def add_label(frame, text, color=(255,255,255), pos=(10,30)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1, cv2.LINE_AA)
    return frame

def create_4panel_video(video_name, label):
    print(f"Processing {video_name}...")

    # Load semua source
    frame_dir   = RGB_DIR / video_name / video_name
    op_json_dir = OP_JSON_DIR / video_name
    mp_json_dir = MP_JSON_DIR / video_name
    op_3d_mp4   = OP_3D_DIR / f"{video_name}.mp4"
    mp_3d_mp4   = MP_3D_DIR / f"{video_name}.mp4"
    output_mp4  = OUT_DIR / f"4panel_{video_name}.mp4"

    frame_files  = sorted(frame_dir.glob("*.png"))
    op_json_files = sorted(op_json_dir.glob("*_keypoints.json"))
    mp_json_files = sorted(mp_json_dir.glob("*_keypoints.json"))

    # Buka video 3D pose
    cap_op3d = cv2.VideoCapture(str(op_3d_mp4))
    cap_mp3d = cv2.VideoCapture(str(mp_3d_mp4))

    # Ukuran panel: resize semua ke 640x480
    PW, PH = 640, 480

    n = min(len(frame_files), len(op_json_files), len(mp_json_files))

    # Output: 2 kolom x 2 baris = 1280 x 960
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(str(output_mp4), fourcc, 30, (PW*2, PH*2))

    # Warna header
    color_label = (0, 220, 100) if 'adl' in video_name else (0, 80, 255)

    for i in range(n):
        # ---- Panel 1: OpenPose 2D ----
        frame_op2d = cv2.imread(str(frame_files[i]))
        frame_op2d = cv2.resize(frame_op2d, (PW, PH))
        op_data    = json.loads(op_json_files[i].read_text())
        if len(op_data['people']) > 0:
            draw_skeleton(frame_op2d, op_data['people'][0]['pose_keypoints_2d'],
                         18, SKELETON_OP, get_color_op)
        add_label(frame_op2d, "OpenPose - 2D Pose", color=(100,200,255))

        # ---- Panel 2: OpenPose 3D ----
        ret, frame_op3d = cap_op3d.read()
        if not ret:
            frame_op3d = np.zeros((PH, PW, 3), dtype=np.uint8)
        frame_op3d = cv2.resize(frame_op3d, (PW, PH))
        add_label(frame_op3d, "OpenPose - 3D Pose (VideoPose3D)", color=(100,200,255))

        # ---- Panel 3: MediaPipe 2D ----
        frame_mp2d = cv2.imread(str(frame_files[i]))
        frame_mp2d = cv2.resize(frame_mp2d, (PW, PH))
        mp_data    = json.loads(mp_json_files[i].read_text())
        if len(mp_data['people']) > 0:
            draw_skeleton(frame_mp2d, mp_data['people'][0]['pose_keypoints_2d'],
                         17, SKELETON_MP, get_color_mp)
        add_label(frame_mp2d, "MediaPipe - 2D Pose", color=(100,255,150))

        # ---- Panel 4: MediaPipe 3D ----
        ret, frame_mp3d = cap_mp3d.read()
        if not ret:
            frame_mp3d = np.zeros((PH, PW, 3), dtype=np.uint8)
        frame_mp3d = cv2.resize(frame_mp3d, (PW, PH))
        add_label(frame_mp3d, "MediaPipe - 3D Pose (VideoPose3D)", color=(100,255,150))

        # ---- Gabungkan 4 panel ----
        top    = np.hstack([frame_op2d, frame_op3d])
        bottom = np.hstack([frame_mp2d, frame_mp3d])
        canvas = np.vstack([top, bottom])

        # Label video dan frame counter
        cv2.putText(canvas, f"{label} | Frame {i+1}/{n}",
                    (PW - 200, PH*2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_label, 2, cv2.LINE_AA)

        out.write(canvas)

    out.release()
    cap_op3d.release()
    cap_mp3d.release()
    print(f"  Saved: {output_mp4.name}")

# ============================================================
print("Membuat video 4-panel perbandingan...")
print("=" * 50)

for video_name, label in [
    ("adl-01-cam0-rgb", "ADL - Aktivitas Normal"),
    ("fall-06-cam0-rgb", "FALL - Jatuh"),
]:
    create_4panel_video(video_name, label)

print("\nSelesai! File tersimpan di /workspace/mediapipe_pipeline/visualisasi/")
print("  4panel_adl-01-cam0-rgb.mp4")
print("  4panel_fall-06-cam0-rgb.mp4")
