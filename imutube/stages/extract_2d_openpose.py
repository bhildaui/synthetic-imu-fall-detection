import sys
import os
import gc
import json
import time
import cv2
from pathlib import Path
from dataclasses import dataclass
from sys import platform

from imutube.config import OpenPoseConfig

# ==============================
# COCO skeleton definition
# ==============================
COCO_PAIRS = [
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


def draw_pose(image, keypoints, conf_th: float = 0.3):
    """
    keypoints: (N, K, 3)
    """
    output = image.copy()

    for person in keypoints:
        # joints
        for x, y, c in person:
            if c > conf_th:
                cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)

        # skeleton
        for a, b in COCO_PAIRS:
            if person[a][2] > conf_th and person[b][2] > conf_th:
                pt1 = (int(person[a][0]), int(person[a][1]))
                pt2 = (int(person[b][0]), int(person[b][1]))
                cv2.line(output, pt1, pt2, (0, 0, 255), 2)

    return output


@dataclass
class Extract2DResult:
    width: int
    height: int
    n_frames: int



def extract_2d(frames_dir: Path, out_dir: Path, cfg: OpenPoseConfig) -> Extract2DResult:
    import pyopenpose as op
    # ----------------------------
    # Output folders (match your working script)
    # ----------------------------
    img_out_dir = out_dir / "rgb_keypoints"
    json_out_dir = out_dir / "json"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    json_out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # OpenPose params (OOM SAFE) - same as working script
    # ----------------------------
    params = {}
    params["model_folder"] = str(cfg.model_folder)  # keep configurable
    params["net_resolution"] = "-320x176"
    params["scale_number"] = 1
    params["scale_gap"] = 0.25
    params["num_gpu"] = 1  # set 0 for CPU
    params["face"] = False
    params["hand"] = False
    params["model_pose"] = "COCO"  # exact match with working script
    params["disable_blending"] = True

    # ----------------------------
    # Start OpenPose
    # ----------------------------
    opw = op.WrapperPython()
    opw.configure(params)
    opw.start()

    # ----------------------------
    # Frame listing (match old pipeline assumption: PNG frames)
    # ----------------------------
    frames = sorted(frames_dir.glob("*.png"))
    print(f"Found {len(frames)} images")

    start = time.time()
    H, W = None, None

    for idx, frame_path in enumerate(frames):
        base = frame_path.stem
        print(f"[{idx+1}/{len(frames)}] {base}")

        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        # # Resize BEFORE OpenPose (same rule)
        # h, w = image.shape[:2]
        # if w > 640:
        #     s = 640.0 / w
        #     image = cv2.resize(image, (int(w * s), int(h * s)))

        if H is None:
            H, W = image.shape[:2]

        datum = op.Datum()
        datum.cvInputData = image
        opw.emplaceAndPop(op.VectorDatum([datum]))

        # ----------------------------
        # Draw keypoints on RGB (same behavior)
        # ----------------------------
        if datum.poseKeypoints is not None:
            drawn = draw_pose(image, datum.poseKeypoints)
        else:
            drawn = image

        cv2.imwrite(str(img_out_dir / f"{base}_rgb_pose.png"), drawn)

        # ----------------------------
        # Save JSON keypoints (same schema + flat list)
        # ----------------------------
        people = []
        if datum.poseKeypoints is not None:
            for pid, person in enumerate(datum.poseKeypoints):
                flat = []
                for x, y, c in person:
                    flat.extend([float(x), float(y), float(c)])
                people.append(
                    {
                        "person_id": pid,
                        "pose_keypoints_2d": flat,
                    }
                )

        json_data = {
            "image": frame_path.name,
            "num_people": len(people),
            "people": people,
        }

        with open(json_out_dir / f"{base}.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # Cleanup (same spirit)
        del datum, image, drawn
        gc.collect()

    print(f"Done in {time.time() - start:.2f}s")
    return Extract2DResult(width=W or 0, height=H or 0, n_frames=len(frames))
