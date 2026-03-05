import subprocess
from pathlib import Path
import numpy as np

# Path settings
VIDEOPOSE3D_DIR = "/workspace/VideoPose3D"
OUTPUT_DIR = Path("/workspace/awal/3d_poses")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load daftar video dari NPZ
data = np.load('/workspace/awal/data_2d_custom_urfd_openpose.npz', allow_pickle=True)
positions = data['positions_2d'].item()
videos = sorted(positions.keys())

print(f"Total videos: {len(videos)}")

for i, video in enumerate(videos):
    output_json = OUTPUT_DIR / f"{video}_world.json"
    
    # Skip kalau sudah ada
    if output_json.exists():
        print(f"[{i+1}/{len(videos)}] Skipping {video} (already done)")
        continue
    
    print(f"[{i+1}/{len(videos)}] Processing {video}...")
    
    cmd = [
        "python3", "run.py",
        "-d", "custom",
        "-k", "urfd_openpose",
        "-arc", "3,3,3,3,3",
        "-c", "checkpoint",
        "--evaluate", "pretrained_h36m_detectron_coco.bin",
        "--render",
        "--viz-subject", video,
        "--viz-action", "custom",
        "--viz-camera", "0",
        "--viz-output", str(OUTPUT_DIR / f"{video}.mp4"),
        "--viz-size", "6",
        "--viz-export", str(output_json).replace("_world.json", "")
    ]
    
    subprocess.run(cmd, cwd=VIDEOPOSE3D_DIR)
    print(f"  Done! Saved to {output_json}")

print("\nAll videos processed!")
