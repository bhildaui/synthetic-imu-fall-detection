import os
import json
import numpy as np

# ==========================
# CONFIG
# ==========================
json_dir = "/workspace/URFD/fall-07-cam0-rgb-out/json"
output_npz = "data/data_2d_custom_openpose.npz"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

SUBJECT = "fall_07_cam0"   # canonical_name
ACTION = "custom"          # must match run.py expectation

# ==========================
# LOAD SEQUENCE
# ==========================
frames = sorted(os.listdir(json_dir))
sequence = []

for jf in frames:
    with open(os.path.join(json_dir, jf)) as f:
        data = json.load(f)

    if data["num_people"] == 0:
        sequence.append(np.zeros((17, 2), dtype=np.float32))
        continue

    kpts = np.array(
        data["people"][0]["pose_keypoints_2d"],
        dtype=np.float32
    ).reshape(18, 3)

    vp = np.zeros((17, 2), dtype=np.float32)

    # VideoPose3D 17-joint layout
    vp[0] = (kpts[8][:2] + kpts[11][:2]) / 2   # pelvis

    vp[1] = kpts[8][:2]
    vp[2] = kpts[9][:2]
    vp[3] = kpts[10][:2]
    vp[4] = kpts[11][:2]
    vp[5] = kpts[12][:2]
    vp[6] = kpts[13][:2]

    vp[9]  = kpts[1][:2]
    vp[10] = kpts[0][:2]

    vp[11] = kpts[5][:2]
    vp[12] = kpts[6][:2]
    vp[13] = kpts[7][:2]

    vp[14] = kpts[2][:2]
    vp[15] = kpts[3][:2]
    vp[16] = kpts[4][:2]

    vp[7] = (vp[0] + vp[9]) / 2
    vp[8] = vp[9]

    sequence.append(vp)

sequence = np.stack(sequence).astype(np.float32)

# Normalize to [-1, 1]
sequence[..., 0] = (sequence[..., 0] / IMAGE_WIDTH) * 2 - 1
sequence[..., 1] = (sequence[..., 1] / IMAGE_HEIGHT) * 2 - 1

# ==========================
# BUILD VP3D STRUCTURE (EXACTLY LIKE EXAMPLE)
# ==========================
positions_2d = {
    SUBJECT: {
        ACTION: [
            sequence
        ]
    }
}

metadata = {
    "layout_name": "coco",
    "num_joints": 17,
    "keypoints_symmetry": (
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16]
    ),
    "video_metadata": {
        SUBJECT: {
            "w": IMAGE_WIDTH,
            "h": IMAGE_HEIGHT
        }
    }
}

np.savez_compressed(
    output_npz,
    positions_2d=positions_2d,
    metadata=metadata
)

print("Saved:", output_npz)
print("Sequence shape:", sequence.shape)
