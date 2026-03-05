import os
import subprocess
from pathlib import Path

# Path settings
URFD_DIR = Path("/workspace/URFD")
FRAMES_DIR = Path("/workspace/awal/frames")
JSON_DIR = Path("/workspace/awal/json")
OPENPOSE_BIN = "/opt/openpose/build/examples/openpose/openpose.bin"
OPENPOSE_MODELS = "/opt/openpose/models"

# Ambil semua video
videos = sorted(URFD_DIR.glob("*.mp4"))
print(f"Total videos: {len(videos)}")

for i, video in enumerate(videos):
    video_name = video.stem  # contoh: fall-01-cam0
    frames_out = FRAMES_DIR / video_name
    json_out = JSON_DIR / video_name

    print(f"\n[{i+1}/{len(videos)}] Processing: {video_name}")

    # Skip kalau JSON sudah ada
    if json_out.exists() and len(list(json_out.glob("*.json"))) > 0:
        print(f"  Skipping (already processed)")
        continue

    # Buat folder output
    frames_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract frames
    print(f"  Extracting frames...")
    ffmpeg_cmd = [
        "ffmpeg", "-i", str(video),
        str(frames_out / "frame_%04d.jpg"),
        "-q:v", "2", "-y"  # -y = overwrite tanpa tanya
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 2: Run OpenPose
    print(f"  Running OpenPose...")
    openpose_cmd = [
        OPENPOSE_BIN,
        "--image_dir", str(frames_out),
        "--write_json", str(json_out),
        "--model_folder", OPENPOSE_MODELS,
        "--model_pose", "COCO",
        "--display", "0",
        "--render_pose", "0"
    ]
    subprocess.run(openpose_cmd)

    # Hapus frames setelah selesai (hemat disk)
    for f in frames_out.glob("*.jpg"):
        f.unlink()
    frames_out.rmdir()

    print(f"  Done! JSON saved to {json_out}")

print("\nAll videos processed!")
