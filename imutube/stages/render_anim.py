import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil
from imutube.config import RenderConfig

# =============================================================================
# Skeleton + Colors
# =============================================================================

# COCO 17 skeleton (VideoPose3D format)
SKELETON_CONNECTIONS_COCO17 = [
    (0, 1), (0, 2),           # Nose to eyes
    (1, 3), (2, 4),           # Eyes to ears
    (0, 5), (0, 6),           # Nose to shoulders
    (5, 6),                   # Shoulders
    (5, 7), (7, 9),           # Left arm
    (6, 8), (8, 10),          # Right arm
    (5, 11), (6, 12),         # Shoulders to hips
    (11, 12),                 # Hips
    (11, 13), (13, 15),       # Left leg
    (12, 14), (14, 16),       # Right leg
]

# OpenPose COCO-18 / BODY_25 simplified (optional)
SKELETON_CONNECTIONS_COCO18 = [
    (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8),
    (8, 9), (9, 10),
    (1, 11),
    (11, 12), (12, 13),
    (1, 0),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]

# Colors (BGR)
COLORS = {
    "head": (255, 100, 0),
    "torso": (0, 255, 0),
    "left_arm": (0, 100, 255),
    "right_arm": (255, 0, 255),
    "left_leg": (255, 200, 0),
    "right_leg": (0, 255, 255),
}


def get_connection_color(connection, num_joints=17):
    """Color mapping consistent with your reference script."""
    start, end = connection

    # Head
    if start == 0 or end == 0:
        if start <= 4 or end <= 4:
            return COLORS["head"]

    # Arms
    if num_joints == 17:
        if (start, end) in [(5, 7), (7, 9)]:
            return COLORS["left_arm"]
        if (start, end) in [(6, 8), (8, 10)]:
            return COLORS["right_arm"]
    elif num_joints == 18:
        if (start, end) in [(1, 5), (5, 6), (6, 7)]:
            return COLORS["left_arm"]
        if (start, end) in [(1, 2), (2, 3), (3, 4)]:
            return COLORS["right_arm"]

    # Legs (keep simple, but stable)
    if (start, end) in [(11, 13), (13, 15), (8, 11), (11, 12)]:
        return COLORS["left_leg"]
    if (start, end) in [(12, 14), (14, 16), (8, 9), (9, 10)]:
        return COLORS["right_leg"]

    return COLORS["torso"]


# =============================================================================
# IO helpers
# =============================================================================

def load_openpose_json_2d(json_path: Path):
    """Load 2D keypoints from OpenPose JSON; returns (keypoints_xy, conf) or (None, None)."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # OpenPose typical schema: people list, no num_people always
    people = data.get("people", [])
    if not people:
        return None, None

    pose_data = people[0].get("pose_keypoints_2d", [])
    if not pose_data:
        return None, None

    num_kp = len(pose_data) // 3
    arr = np.array(pose_data, dtype=np.float32).reshape(num_kp, 3)
    return arr[:, :2], arr[:, 2]


def load_videpose_json_v2(json_path: Path):
    """
    Load VideoPose3D-style JSON.
    Improvements (aligned with your reference):
    - auto-scale to mm if range indicates meters (threshold < 10)
    - compute FIXED axis limits over whole sequence (valid coords only)
    Returns: (keypoints_3d, metadata)
    """
    print(f"Loading 3D JSON: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    num_frames = len(frames)
    num_joints = data["video_info"]["num_joints"]

    keypoints_3d = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        for joint in frame.get("keypoints_3d", []):
            j = joint["joint_id"]
            keypoints_3d[i, j] = (joint["x"], joint["y"], joint["z"])

    metadata = {
        "joint_names": data.get("joint_names", []),
        "video_info": data.get("video_info", {}),
        "coordinate_system": data.get("coordinate_system", {}),
        "scaled": False,
        "scale_factor": 1,
        "fixed_axis_limits": None,
    }

    coord_range = float(np.abs(keypoints_3d).max())
    print(f"  Frames: {num_frames}, Joints: {num_joints}, MaxAbs: {coord_range:.4f}")

    # Reference script logic: if < 10 => treat as meters and scale to mm
    if coord_range < 10.0:
        print("  Detected meter-scale 3D. Scaling to mm (x1000).")
        keypoints_3d *= 1000.0
        metadata["scaled"] = True
        metadata["scale_factor"] = 1000

    # Compute fixed axis limits across all frames (valid coords only)
    valid_mask = ~np.all(keypoints_3d == 0, axis=2)  # (T, J)
    valid_coords = keypoints_3d[valid_mask]           # (N, 3)

    if valid_coords.shape[0] > 0:
        x_min, x_max = valid_coords[:, 0].min(), valid_coords[:, 0].max()
        y_min, y_max = valid_coords[:, 1].min(), valid_coords[:, 1].max()
        z_min, z_max = valid_coords[:, 2].min(), valid_coords[:, 2].max()

        x_range = float(x_max - x_min)
        y_range = float(y_max - y_min)
        z_range = float(z_max - z_min)

        max_range = max(x_range, y_range, z_range)
        mid_x = float((x_max + x_min) / 2)
        mid_y = float((y_max + y_min) / 2)
        mid_z = float((z_max + z_min) / 2)

        padding_factor = 0.3
        x_pad = x_range * padding_factor
        y_pad = y_range * padding_factor
        z_pad = z_range * padding_factor

        metadata["fixed_axis_limits"] = {
            "x": [mid_x - max_range / 2 - x_pad, mid_x + max_range / 2 + x_pad],
            "y": [mid_y - max_range / 2 - y_pad, mid_y + max_range / 2 + y_pad],
            "z": [mid_z - max_range / 2 - z_pad, mid_z + max_range / 2 + z_pad],
        }

        print("  Fixed axis limits:")
        print(f"    X: {metadata['fixed_axis_limits']['x']}")
        print(f"    Y: {metadata['fixed_axis_limits']['y']}")
        print(f"    Z: {metadata['fixed_axis_limits']['z']}")

    return keypoints_3d, metadata


def list_frame_files(frames_dir: Path):
    """Robust frame listing: jpg/png, case-insensitive, sorted."""
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for pat in exts:
        files = sorted(frames_dir.glob(pat))
        if files:
            return files
    return []


# =============================================================================
# 2D drawing
# =============================================================================

def draw_2d_skeleton(image, keypoints_2d, frame_idx, num_joints=18):
    img = image.copy()
    h, w = img.shape[:2]

    connections = SKELETON_CONNECTIONS_COCO18 if num_joints == 18 else SKELETON_CONNECTIONS_COCO17

    # Connections
    for (s, e) in connections:
        if s >= keypoints_2d.shape[0] or e >= keypoints_2d.shape[0]:
            continue

        p1 = keypoints_2d[s]
        p2 = keypoints_2d[e]

        # Validity: non-zero and within image bounds
        if (
            p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0
            and 0 <= p1[0] < w and 0 <= p1[1] < h
            and 0 <= p2[0] < w and 0 <= p2[1] < h
        ):
            color = get_connection_color((s, e), num_joints=num_joints)
            cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 3, cv2.LINE_AA)

    # Joints
    for p in keypoints_2d:
        if p[0] > 0 and p[1] > 0 and 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, tuple(p.astype(int)), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # Info bar
    cv2.rectangle(img, (5, 5), (200, 45), (0, 0, 0), -1)
    cv2.putText(
        img, f"Frame: {frame_idx}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return img


# =============================================================================
# 3D rendering
# =============================================================================

def create_3d_visualization(
    keypoints_3d,
    frame_idx,
    title,
    fixed_axis_limits=None,
    elev=20,
    azim=45,
    rotate_view=False,
):
    """
    Matplotlib 3D -> image with:
    - thick lines + markers
    - FIXED axis limits if provided (static across frames)
    - optional slow rotation (for world space)
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw skeleton
    for (s, e) in SKELETON_CONNECTIONS_COCO17:
        if s >= keypoints_3d.shape[0] or e >= keypoints_3d.shape[0]:
            continue

        p1 = keypoints_3d[s]
        p2 = keypoints_3d[e]

        # Validity: not all zeros
        if np.all(p1 == 0) or np.all(p2 == 0):
            continue

        color_bgr = get_connection_color((s, e), num_joints=17)
        color_rgb = (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)

        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color_rgb,
            linewidth=3,
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    # Fixed axis limits
    if fixed_axis_limits is not None:
        ax.set_xlim(fixed_axis_limits["x"])
        ax.set_ylim(fixed_axis_limits["y"])
        ax.set_zlim(fixed_axis_limits["z"])
    else:
        # fallback (avoid per-frame jitter where possible; still OK as fallback)
        valid_mask = ~np.all(keypoints_3d == 0, axis=1)
        valid = keypoints_3d[valid_mask]
        if valid.shape[0] > 0:
            x_min, x_max = valid[:, 0].min(), valid[:, 0].max()
            y_min, y_max = valid[:, 1].min(), valid[:, 1].max()
            z_min, z_max = valid[:, 2].min(), valid[:, 2].max()
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            mid_z = (z_max + z_min) / 2
            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.set_xlabel("X (mm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Y (mm)", fontsize=10, fontweight="bold")
    ax.set_zlabel("Z (mm)", fontsize=10, fontweight="bold")
    ax.set_title(f"{title}\nFrame {frame_idx}", fontsize=11, fontweight="bold")

    if rotate_view:
        ax.view_init(elev=elev, azim=azim + frame_idx * 0.3)  # slow rotation
    else:
        ax.view_init(elev=elev, azim=azim)

    ax.grid(True, alpha=0.4, linestyle="--")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_bgr


# =============================================================================
# Layout + labels
# =============================================================================

def _resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _add_label(img, label, bar_h=40):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stack_2x2(img_rgb, img_2d, img_3d_cam, img_3d_world):
    top = np.hstack([img_rgb, img_2d])
    bottom = np.hstack([img_3d_cam, img_3d_world])
    return np.vstack([top, bottom])


# =============================================================================
# Main render()
# =============================================================================

def render(
    frames_dir: Path,
    json2d_dir: Path,
    json3d_world: Path,
    json3d_local: Path,
    out_video: Path,
    cfg: RenderConfig,
):
    """
    Fixed version aligned with your reference script:

    - Proper fixed axis limits (computed in load_videpose_json_v2 -> metadata['fixed_axis_limits'])
    - Frame count is min of available RGB frames / 2D json / 3D arrays (no index drift)
    - Robust 2D loading via load_openpose_json_2d
    - No double-read of RGB frames
    - Labeled 4-panel layout consistent with your reference
    - Optional world rotation via cfg.rotate_world if present (fallback False)
    - Optional keep frames via cfg.keep_frames if present (fallback False)
    """
    frames = list_frame_files(frames_dir)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    json_files = sorted(json2d_dir.glob("*.json"))

    print(f"Loading 3D data:\n  world: {json3d_world}\n  local: {json3d_local}")
    kp_3d_world, meta_world = load_videpose_json_v2(json3d_world)
    kp_3d_local, meta_local = load_videpose_json_v2(json3d_local)

    fixed_world = meta_world.get("fixed_axis_limits")
    fixed_local = meta_local.get("fixed_axis_limits")

    # Determine 2D joint count from first valid json (else default 18)
    num_joints_2d = 18
    if json_files:
        kp2d0, _ = load_openpose_json_2d(json_files[0])
        if kp2d0 is not None:
            num_joints_2d = int(kp2d0.shape[0])

    target_size = (480, 480)

    # Align frame range safely
    n = min(len(frames), len(json_files) if json_files else len(frames), len(kp_3d_world), len(kp_3d_local))
    if n <= 0:
        raise RuntimeError("No overlapping frames among RGB/2D/3D inputs.")

    # Optional start/end if RenderConfig supports them
    start = int(getattr(cfg, "start", 0) or 0)
    end = getattr(cfg, "end", None)
    if end is None:
        end = n
    end = min(int(end), n)
    start = max(0, min(start, end))
    if end - start <= 0:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}, n={n}")

    rotate_world = bool(getattr(cfg, "rotate_world", False))
    keep_frames = bool(getattr(cfg, "keep_frames", False))

    # Temp dir (unique per output to avoid collisions)
    temp_dir = out_video.parent / f"temp_animation_frames_{out_video.stem}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering frames [{start}:{end}) to {temp_dir} ...")

    for i in tqdm(range(start, end), total=(end - start)):
        # RGB
        img_rgb = cv2.imread(str(frames[i]))
        if img_rgb is None:
            # keep alignment: write a blank frame
            img_rgb = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        img_rgb = _resize(img_rgb, target_size)
        img_rgb = _add_label(img_rgb, "RGB Video")

        # 2D
        kp_2d = np.zeros((num_joints_2d, 2), dtype=np.float32)
        if i < len(json_files):
            kp2d_i, _ = load_openpose_json_2d(json_files[i])
            if kp2d_i is not None:
                # If json has more joints than we expect, trim; if fewer, pad
                m = min(kp_2d.shape[0], kp2d_i.shape[0])
                kp_2d[:m] = kp2d_i[:m]

        img_2d = draw_2d_skeleton(img_rgb.copy(), kp_2d, i, num_joints=num_joints_2d)
        img_2d = _resize(img_2d, target_size)
        img_2d = _add_label(img_2d, "2D Keypoints")

        # 3D Camera (local)
        kp_cam = kp_3d_local[i] if i < len(kp_3d_local) else np.zeros((17, 3), dtype=np.float32)
        img_3d_cam = create_3d_visualization(
            kp_cam,
            i,
            "3D Camera Space",
            fixed_axis_limits=fixed_local,
            elev=20,
            azim=45,
            rotate_view=False,
        )
        img_3d_cam = _resize(img_3d_cam, target_size)
        img_3d_cam = _add_label(img_3d_cam, "3D Camera Space")

        # 3D World
        kp_w = kp_3d_world[i] if i < len(kp_3d_world) else np.zeros((17, 3), dtype=np.float32)
        img_3d_world = create_3d_visualization(
            kp_w,
            i,
            "3D World Space",
            fixed_axis_limits=fixed_world,
            elev=20,
            azim=45,
            rotate_view=rotate_world,
        )
        img_3d_world = _resize(img_3d_world, target_size)
        img_3d_world = _add_label(img_3d_world, "3D World Space")

        combined = stack_2x2(img_rgb, img_2d, img_3d_cam, img_3d_world)

        # IMPORTANT: ffmpeg expects 000000.. starting at 0
        out_idx = i - start
        cv2.imwrite(str(temp_dir / f"frame_{out_idx:06d}.png"), combined)

    print(f"Creating video with ffmpeg at {out_video}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate",
        str(cfg.fps),
        "-i",
        str(temp_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(out_video),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed.")
        print(e.stderr)
        raise

    if not keep_frames:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"Keeping rendered frames in: {temp_dir}")

    print(f"Video saved to {out_video}")
