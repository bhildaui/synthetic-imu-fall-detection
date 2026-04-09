"""
Re-render 3D pose dengan menambahkan global trajectory dari posisi 2D pixel.
Cara kerja:
- Ambil posisi mid-hip 2D pixel dari MediaPipe
- Konversi ke koordinat meter menggunakan focal length GeoCalib
- Tambahkan sebagai offset ke semua joint 3D
- Render ulang menggunakan matplotlib
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.signal import savgol_filter
import cv2

MP_JSON_DIR = Path("/workspace/mediapipe_pipeline/json")
POSES_3D_DIR = Path("/workspace/mediapipe_pipeline/3d_poses")
CAM_PARAMS   = Path("/workspace/awal/camera_params.npy")
OUT_DIR      = Path("/workspace/mediapipe_pipeline/visualisasi")
OUT_DIR.mkdir(exist_ok=True)

# Skeleton connections Detectron 17 keypoint
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

cam_params = np.load(CAM_PARAMS, allow_pickle=True).item()
CONF_THRESHOLD = 0.3
CX, CY = 320.0, 240.0
DT = 1/30

def load_3d_poses(video_name):
    data = json.load(open(POSES_3D_DIR / f"{video_name}_world.json"))
    poses = []
    for frame in data['frames']:
        kps = [[kp['x'], kp['y'], kp['z']] for kp in frame['keypoints_3d']]
        poses.append(kps)
    return np.array(poses)  # (N, 17, 3)

def load_midhip_pixel(video_name):
    files = sorted((MP_JSON_DIR / video_name).glob("*_keypoints.json"))
    mh = []
    for f in files:
        d = json.loads(f.read_text())
        if len(d['people']) > 0:
            kp = np.array(d['people'][0]['pose_keypoints_2d']).reshape(17, 3)
            rc, lc = kp[12, 2], kp[11, 2]
            if rc < CONF_THRESHOLD and lc < CONF_THRESHOLD:
                mh.append([np.nan, np.nan])
            elif rc < CONF_THRESHOLD:
                mh.append(kp[11, :2].tolist())
            elif lc < CONF_THRESHOLD:
                mh.append(kp[12, :2].tolist())
            else:
                mh.append(((kp[11, :2] + kp[12, :2]) / 2).tolist())
        else:
            mh.append([np.nan, np.nan])
    mh = np.array(mh)
    # Interpolasi NaN
    for i in range(2):
        mask = np.isnan(mh[:, i])
        if mask.any() and (~mask).sum() >= 2:
            idx = np.arange(len(mh))
            mh[mask, i] = np.interp(idx[mask], idx[~mask], mh[~mask, i])
    return mh

def add_trajectory(poses_3d, mh_pixel, focal):
    """Tambahkan global trajectory dari posisi 2D ke poses 3D"""
    N = min(len(poses_3d), len(mh_pixel))
    poses_3d = poses_3d[:N].copy()
    mh_pixel = mh_pixel[:N]

    # Ambil depth Z dari mid-hip 3D
    mh_z = (poses_3d[:, 11, 2] + poses_3d[:, 12, 2]) / 2

    # Back-projection: pixel -> meter
    traj_x = (mh_pixel[:, 0] - CX) * mh_z / focal
    traj_y = (mh_pixel[:, 1] - CY) * mh_z / focal

    # Smooth trajectory
    traj_x = savgol_filter(traj_x, 15, 3)
    traj_y = savgol_filter(traj_y, 15, 3)

    # Tambahkan offset ke semua joint
    for t in range(N):
        poses_3d[t, :, 0] += traj_x[t]
        poses_3d[t, :, 1] += traj_y[t]

    return poses_3d

def render_3d_video(video_name, label, fps=30):
    print(f"Processing {video_name}...")

    focal = cam_params[video_name]['focal_length']
    poses_3d = load_3d_poses(video_name)
    mh_pixel = load_midhip_pixel(video_name)

    # Tambahkan global trajectory
    poses_traj = add_trajectory(poses_3d, mh_pixel, focal)
    N = len(poses_traj)

    # Setup figure
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')

    color_title = 'green' if 'adl' in video_name else 'red'

    # Hitung range untuk axis yang konsisten
    all_pts = poses_traj.reshape(-1, 3)
    x_range = [all_pts[:,0].min()-0.3, all_pts[:,0].max()+0.3]
    y_range = [all_pts[:,1].min()-0.3, all_pts[:,1].max()+0.3]
    z_range = [all_pts[:,2].min()-0.1, all_pts[:,2].max()+0.1]

    # Warna per bagian skeleton
    def get_seg_color(i, j):
        head  = {0,1,2,3,4}
        lower = {11,12,13,14,15,16}
        if i in head or j in head: return '#FFB300'
        if i in lower or j in lower: return '#00C853'
        return '#00B0FF'

    # Render frame per frame
    output_mp4 = OUT_DIR / f"3d_trajectory_{video_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Render dulu ke PNG buffer lalu gabung ke video
    frames_buf = []
    for t in range(N):
        ax.cla()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"{label}\nFrame {t+1}/{N}", color=color_title, fontsize=11)
        ax.view_init(elev=15, azim=-60)
        ax.grid(True, alpha=0.3)

        pts = poses_traj[t]  # (17, 3)

        # Gambar skeleton
        for i, j in SKELETON:
            color = get_seg_color(i, j)
            ax.plot([pts[i,0], pts[j,0]],
                    [pts[i,1], pts[j,1]],
                    [pts[i,2], pts[j,2]],
                    color=color, linewidth=2)

        # Gambar keypoint
        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                   c='white', s=30, edgecolors='gray', linewidths=0.5, zorder=5)

        # Gambar lantai (referensi)
        floor_y = y_range[1]
        xx, zz = np.meshgrid(x_range, z_range)
        ax.plot_surface(xx, np.full_like(xx, floor_y), zz,
                       alpha=0.1, color='gray')

        # Convert figure ke numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w_fig, h_fig = fig.canvas.get_width_height()
        buf = buf.reshape(h_fig, w_fig, 4)[:, :, :3]
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buf_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        frames_buf.append(buf_bgr)

        if t % 20 == 0:
            print(f"  Rendered {t+1}/{N} frames...")

    # Tulis video
    h, w = frames_buf[0].shape[:2]
    out  = cv2.VideoWriter(str(output_mp4), fourcc, fps, (w, h))
    for frame in frames_buf:
        out.write(frame)
    out.release()
    plt.close(fig)

    print(f"  Saved: {output_mp4.name}")

# ============================================================
print("Render 3D pose dengan global trajectory...")
print("=" * 50)

for video_name, label in [
    ("adl-01-cam0-rgb", "ADL - Aktivitas Normal (MediaPipe + Trajectory)"),
    ("fall-06-cam0-rgb", "FALL - Jatuh (MediaPipe + Trajectory)"),
]:
    render_3d_video(video_name, label)

print("\nSelesai!")
