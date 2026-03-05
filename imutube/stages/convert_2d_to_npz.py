import json
import numpy as np
from pathlib import Path

def convert_openpose_json_to_vp3d_npz(
        json_dir: Path,
        out_npz: Path,
        subject: str,
        action: str,
        width: int,
        height: int
):
    files = sorted(json_dir.glob("*.json"))
    seq = []

    for f in files:
        d = json.load(open(f))
        if d["num_people"] == 0:
            seq.append(np.zeros((17, 2)))
            continue

        k = np.array(d["people"][0]["pose_keypoints_2d"]).reshape(18, 3)
        vp = np.zeros((17, 2))

        vp[0] = (k[8,:2] + k[11,:2]) / 2 # Hip (Mid)
        vp[1:7] = k[8:14,:2] # RHip, RKnee, RAnk, LHip, LKnee, LAnk

        vp[7] = (vp[0] + k[1,:2]) / 2 # Spine (Mid Hip - Neck)
        vp[8] = k[1,:2]               # Thorax/Neck
        vp[9] = k[0,:2]               # Nose (Neck/Nose in H36M?)
        
        # Fake Head (10) to match H36M 17-joint layout
        vp[10] = vp[9] + (vp[9] - vp[8])

        vp[11:14] = k[5:8,:2] # Left Arm (LSho, LElb, LWri)
        vp[14:17] = k[2:5,:2] # Right Arm (RSho, RElb, RWri)

        seq.append(vp)

    seq = np.array(seq)
    if width > 0 and height > 0:
        seq[...,0] = seq[...,0] / width * 2 - 1
        seq[...,1] = seq[...,1] / height * 2 - 1

    data = {subject: {action: [seq]}}
    meta = {
        "layout_name": "coco",
        "num_joints": 17,
        "video_metadata": {subject: {"w": width, "h": height}},
        "keypoints_symmetry": [
             [4, 5, 6, 11, 12, 13],  # Left joints
             [1, 2, 3, 14, 15, 16]   # Right joints
        ]
    }

    out_npz.parent.mkdir(exist_ok=True)
    np.savez_compressed(out_npz, positions_2d=data, metadata=meta)