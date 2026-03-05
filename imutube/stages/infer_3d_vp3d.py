import sys
import torch
import numpy as np
from pathlib import Path
from imutube.config import VP3DConfig
from imutube.utils.paths import ROOT
from imutube.vendor.library import ensure_vp3d_on_path

def infer_3d(npz_path: Path, out_json: Path, cfg: VP3DConfig):
    # 1. Ensure VideoPose3D is importable
    # ensure_vp3d_on_path(ROOT)

    from common.model import TemporalModel
    from common.custom_dataset import CustomDataset
    from common.camera import normalize_screen_coordinates, camera_to_world
    from common.generators import UnchunkedGenerator

    # 2. Load dataset and metadata
    #    CustomDataset loads the npz internally to get metadata/resolutions
    dataset = CustomDataset(str(npz_path))
    
    #    We also need to load it manually to get the actual 2D positions similarly to run.py
    #    (CustomDataset stores "custom" structure but we need to follow generic extraction)
    raw_data = np.load(npz_path, allow_pickle=True)
    metadata = raw_data['metadata'].item()
    keypoints = raw_data['positions_2d'].item()

    #    Get symmetry and skeleton info
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

    # 3. Select specific subject/action/camera
    subject = cfg.subject
    action = cfg.action
    # VideoPose3D default viz-camera is 0
    cam_idx = 0 

    if subject not in keypoints:
        raise ValueError(f"Subject {subject} not found in {npz_path}")
    if action not in keypoints[subject]:
        raise ValueError(f"Action {action} not found for subject {subject}")

    input_keypoints = keypoints[subject][action][cam_idx].copy()
    
    # 4. Normalize keypoints
    #    Get camera resolution from dataset
    cam = dataset.cameras()[subject][cam_idx]
    input_keypoints[..., :2] = normalize_screen_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

    # 5. Initialize Model (Using defaults from run.py/arguments.py)
    filter_widths = [3, 3, 3, 3, 3]
    dropout = 0.25
    channels = 1024
    causal = False
    dense = False
    
    #    We use TemporalModel as in run.py (unless optimized/stride=1 logic triggers, 
    #    but we are doing inference, run.py uses TemporalModel for evaluation mostly)
    #    Actually run.py uses TemporalModel for model_pos in evaluation.
    model_pos = TemporalModel(
        input_keypoints.shape[-2], 
        input_keypoints.shape[-1], 
        dataset.skeleton().num_joints(),
        filter_widths=filter_widths, 
        causal=causal, 
        dropout=dropout, 
        channels=channels,
        dense=dense
    )

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2
    if causal:
        causal_shift = pad
    else:
        causal_shift = 0

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    # 6. Load Checkpoint
    #    cfg.checkpoint is the full path to the checkpoint file
    print(f"Loading checkpoint {cfg.checkpoint}")
    checkpoint = torch.load(cfg.checkpoint, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])
    model_pos.eval()

    # 7. Generate Predictions
    #    Using UnchunkedGenerator just like run.py does for rendering
    gen = UnchunkedGenerator(
        None, None, [input_keypoints],
        pad=pad, causal_shift=causal_shift, augment=True, # test_time_augmentation default is True
        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
    )

    prediction = EvaluateNet(gen, model_pos, joints_left, joints_right, return_predictions=True)

    # 8. Post-process (Camera to World)
    #    Invert camera transformation
    #    (If ground truth is not available, take the camera extrinsic params from a random subject... 
    #     but here we use the specific camera from the dataset)
    
    #    Save Camera Space Predictions first
    local_out_path = out_json.parent / (out_json.stem + "_local.json")
    print(f"Exporting camera space joint positions to {local_out_path}")
    
    # Get fps from cfg or default 30
    fps = 30
    
    export_predictions_to_json(
        prediction, input_keypoints, subject, action, cam_idx, local_out_path, fps, cam 
    )

    #    run.py logic for visualization:
    #    rot = dataset.cameras()[subject][cam_idx]['orientation']
    #    prediction = camera_to_world(prediction, R=rot, t=0)
    #    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    
    rot = cam['orientation']
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    # 9. Export World Space
    print(f"Exporting world space joint positions to {out_json}")
    export_predictions_to_json(
        prediction, input_keypoints, subject, action, cam_idx, out_json, fps, cam
    )

def EvaluateNet(test_generator, model_pos, joints_left, joints_right, return_predictions=False):
    # Helper similar to evaluate() in run.py, but stripped down for simple inference
    with torch.no_grad():
        model_pos.eval()
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled in generator)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
            
    return None

import json

def export_predictions_to_json(prediction, input_keypoints, subject, action, camera_idx, 
                                output_path, fps=30, cam_params=None):
    """
    Export 3D pose predictions ke JSON format
    """
    
    # COCO 17 joint names
    joint_names = [
        'Nose', 'LEye', 'REye', 'LEar', 'REar',
        'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
        'LWrist', 'RWrist', 'LHip', 'RHip',
        'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
    ]
    
    # Skeleton connections
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (0, 5), (0, 6), (5, 6),           # Shoulders
        (5, 7), (7, 9),                   # Left arm
        (6, 8), (8, 10),                  # Right arm
        (5, 11), (6, 12), (11, 12),       # Torso
        (11, 13), (13, 15),               # Left leg
        (12, 14), (14, 16),               # Right leg
    ]
    
    num_frames, num_joints, _ = prediction.shape
    
    # Build JSON structure
    output_data = {
        "video_info": {
            "subject": subject,
            "action": action,
            "camera": int(camera_idx),
            "num_frames": int(num_frames),
            "num_joints": int(num_joints),
            "fps": fps,
            "keypoint_format": "coco"
        },
        "coordinate_system": {
            "3d": "camera coordinate system (x: right, y: down, z: forward) in millimeters",
            "2d": "normalized coordinates [-1, 1] (origin: center)"
        },
        "joint_names": joint_names,
        "skeleton_connections": skeleton_connections,
        "frames": []
    }
    
    # Add camera parameters if available
    if cam_params is not None:
        output_data["camera_parameters"] = {
            "resolution_w": int(cam_params.get('res_w', 0)),
            "resolution_h": int(cam_params.get('res_h', 0)),
            "azimuth": float(cam_params.get('azimuth', 0))
        }
        
        if 'orientation' in cam_params:
            if isinstance(cam_params['orientation'], np.ndarray):
                output_data["camera_parameters"]["orientation"] = cam_params['orientation'].tolist()
            else:
                 output_data["camera_parameters"]["orientation"] = cam_params['orientation']

        if 'translation' in cam_params:
            if isinstance(cam_params['translation'], np.ndarray):
                 output_data["camera_parameters"]["translation"] = cam_params['translation'].tolist()
            else:
                 output_data["camera_parameters"]["translation"] = cam_params['translation']
    
    # Process each frame
    for frame_idx in range(num_frames):
        timestamp = frame_idx / fps
        
        frame_data = {
            "frame_id": int(frame_idx),
            "timestamp": float(timestamp),
            "keypoints_3d": [],
            "keypoints_2d": []
        }
        
        # Add 3D keypoints
        for joint_idx in range(num_joints):
            joint_3d = prediction[frame_idx, joint_idx]
            
            keypoint_3d = {
                "joint_id": int(joint_idx),
                "name": joint_names[joint_idx],
                "x": float(joint_3d[0]),
                "y": float(joint_3d[1]),
                "z": float(joint_3d[2])
            }
            
            frame_data["keypoints_3d"].append(keypoint_3d)
        
        # Add 2D keypoints
        if input_keypoints is not None and frame_idx < len(input_keypoints):
            for joint_idx in range(num_joints):
                joint_2d = input_keypoints[frame_idx, joint_idx]
                
                keypoint_2d = {
                    "joint_id": int(joint_idx),
                    "name": joint_names[joint_idx],
                    "x": float(joint_2d[0]),
                    "y": float(joint_2d[1])
                }
                
                frame_data["keypoints_2d"].append(keypoint_2d)
        
        output_data["frames"].append(frame_data)
    
    # Write JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
