"""
Script batch VideoPose3D untuk semua video MediaPipe
"""

import subprocess
from pathlib import Path
import numpy as np

VIDEOPOSE_DIR = Path("/workspace/VideoPose3D")
OUTPUT_DIR    = Path("/workspace/mediapipe_pipeline/3d_poses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data   = np.load(VIDEOPOSE_DIR / "data/data_2d_custom_urfd_mediapipe.npz", allow_pickle=True)
videos = sorted(data['positions_2d'].item().keys())
total  = len(videos)

print(f"Total video: {total}")
print("=" * 50)

for count, video_name in enumerate(videos, start=1):
    output_json = OUTPUT_DIR / f"{video_name}_world.json"

    if output_json.exists():
        print(f"[{count}/{total}] Skip {video_name}")
        continue

    print(f"[{count}/{total}] Processing {video_name}...")

    cmd = [
        "python", "run.py",
        "-d", "custom",
        "-k", "urfd_mediapipe",
        "-str", video_name,
        "-ste", video_name,
        "--evaluate", "pretrained_h36m_detectron_coco.bin",
        "--render",
        "--viz-subject", video_name,
        "--viz-output", str(OUTPUT_DIR / f"{video_name}.mp4"),
        "--viz-size", "5",
        "-arc", "3,3,3,3,3",
        "-c", "checkpoint",
        "--viz-action", "custom",
        "--viz-camera", "0",
        "--viz-export", str(output_json)
    ]

    subprocess.run(cmd, cwd=str(VIDEOPOSE_DIR))

    if output_json.exists():
        print(f"  OK: {output_json.name}")
    else:
        print(f"  WARNING: output tidak ditemukan!")

print("")
print("=" * 50)
print("SELESAI!")
