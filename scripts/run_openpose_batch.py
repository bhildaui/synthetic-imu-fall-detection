"""
Script batch untuk menjalankan OpenPose pada semua video URFD
Input : PNG frames dari /workspace/URFD/rgb/{nama}/{nama}/
Output: JSON keypoints ke /workspace/awal/json/{nama}/
"""

import os
import subprocess
from pathlib import Path

# ============================================================
# KONFIGURASI PATH
# ============================================================
RGB_DIR  = Path("/workspace/URFD/rgb")
JSON_DIR = Path("/workspace/awal/json")
OPENPOSE = "/workspace/openpose/build/examples/openpose/openpose.bin"

JSON_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# AMBIL SEMUA FOLDER VIDEO
# ============================================================
video_folders = sorted(RGB_DIR.iterdir())
total = len(video_folders)

print(f"Total video ditemukan: {total}")
print("=" * 50)

for count, video_folder in enumerate(video_folders, start=1):

    video_name = video_folder.name
    input_dir  = video_folder / video_name
    output_dir = JSON_DIR / video_name

    # Skip jika sudah ada output (pakai *_keypoints.json, bukan adl-*)
    existing = list(output_dir.glob("*_keypoints.json"))
    if output_dir.exists() and len(existing) > 0:
        print(f"[{count}/{total}] Skip {video_name} (sudah ada {len(existing)} file)")
        continue

    if not input_dir.exists():
        print(f"[{count}/{total}] SKIP {video_name} - input tidak ditemukan: {input_dir}")
        continue

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{count}/{total}] Processing {video_name}...")

    cmd = [
        OPENPOSE,
        "--image_dir",         str(input_dir),
        "--write_json",        str(output_dir),
        "--model_pose",        "COCO",
        "--number_people_max", "1",
        "--display",           "0",
        "--render_pose",       "0"
    ]

    subprocess.run(cmd, cwd="/workspace/openpose")

    # Hitung detection rate
    all_json     = list(output_dir.glob("*_keypoints.json"))
    total_frames = len(all_json)

    if total_frames == 0:
        print(f"  WARNING: tidak ada output JSON!")
        continue

    empty = 0
    for f in all_json:
        content = f.read_text()
        if '"people":[]' in content:
            empty += 1

    detected = total_frames - empty
    rate     = detected / total_frames * 100

    print(f"  Detection rate: {rate:.1f}% ({detected}/{total_frames} frames)")

print("")
print("=" * 50)
print("SELESAI! Semua video diproses.")
