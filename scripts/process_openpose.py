"""
Script untuk process OpenPose JSON output menjadi pose sequence
Disesuaikan dengan struktur URFD dataset
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class URFDOpenPoseProcessor:
    def __init__(self):
        # OpenPose BODY_25 atau COCO format
        # Kita asumsikan COCO (18 keypoints)
        self.joint_names_coco = [
            'Nose',           # 0
            'Neck',           # 1
            'RShoulder',      # 2
            'RElbow',         # 3
            'RWrist',         # 4
            'LShoulder',      # 5
            'LElbow',         # 6
            'LWrist',         # 7
            'RHip',           # 8
            'RKnee',          # 9
            'RAnkle',         # 10
            'LHip',           # 11
            'LKnee',          # 12
            'LAnkle',         # 13
            'REye',           # 14
            'LEye',           # 15
            'REar',           # 16
            'LEar'            # 17
        ]
        
        # Joint penting untuk fall detection
        self.important_joint_indices = {
            'Nose': 0,
            'Neck': 1,
            'RShoulder': 2,
            'LShoulder': 5,
            'RHip': 8,
            'LHip': 11,
            'RKnee': 9,
            'LKnee': 12,
            'RAnkle': 10,
            'LAnkle': 13
        }
    
    def load_json_sequence(self, json_folder):
        """
        Load semua JSON dari folder OpenPose output
        """
        json_folder = Path(json_folder)
        
        if not json_folder.exists():
            print(f"❌ Folder tidak ditemukan: {json_folder}")
            return None, None
        
        # Cari semua file JSON yang berisi 'keypoints'
        json_files = sorted(json_folder.glob("*_keypoints.json"))
        
        if len(json_files) == 0:
            print(f"❌ Tidak ada file *_keypoints.json di {json_folder}")
            return None, None
        
        print(f"📂 Found {len(json_files)} JSON files")
        print(f"   First: {json_files[0].name}")
        print(f"   Last:  {json_files[-1].name}")
        
        pose_sequence = []
        frame_names = []
        missing_frames = 0
        
        for i, json_file in enumerate(json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if len(data['people']) > 0:
                # Ambil person pertama
                keypoints = data['people'][0]['pose_keypoints_2d']
                
                # Tentukan jumlah joints
                num_joints = len(keypoints) // 3
                
                # Reshape ke [num_joints, 3]
                keypoints = np.array(keypoints).reshape(num_joints, 3)
                
                pose_sequence.append(keypoints)
                frame_names.append(json_file.stem)
            else:
                # Tidak ada deteksi
                missing_frames += 1
                num_joints = 18
                keypoints = np.full((num_joints, 3), np.nan)
                
                pose_sequence.append(keypoints)
                frame_names.append(json_file.stem)
        
        pose_sequence = np.array(pose_sequence)
        
        print(f"\n✅ Pose sequence loaded:")
        print(f"   Shape: {pose_sequence.shape}")
        print(f"   Missing detections: {missing_frames}/{len(json_files)} frames")
        
        return pose_sequence, frame_names
    
    def interpolate_missing_poses(self, pose_sequence):
        """
        Interpolasi untuk frame yang missing detection
        """
        T, J, _ = pose_sequence.shape
        interpolated = pose_sequence.copy()
        
        for j in range(J):
            for coord in range(2):  # x dan y saja
                data = pose_sequence[:, j, coord]
                
                if np.any(np.isnan(data)):
                    mask = ~np.isnan(data)
                    
                    if np.any(mask):
                        indices = np.arange(len(data))
                        valid_indices = indices[mask]
                        valid_data = data[mask]
                        
                        interpolated[:, j, coord] = np.interp(
                            indices, 
                            valid_indices, 
                            valid_data
                        )
                    else:
                        interpolated[:, j, coord] = 0
        
        # Confidence
        for t in range(T):
            if np.any(np.isnan(pose_sequence[t, :, 2])):
                interpolated[t, :, 2] = 0.5
        
        print(f"✅ Interpolation done")
        
        return interpolated
    
    def add_midhip_joint(self, pose_sequence):
        """
        Tambahkan MidHip sebagai joint tambahan
        """
        rhip = pose_sequence[:, 8, :]
        lhip = pose_sequence[:, 11, :]
        
        midhip = (rhip + lhip) / 2.0
        midhip[:, 2] = np.minimum(rhip[:, 2], lhip[:, 2])
        
        # Insert setelah LHip
        pose_with_midhip = np.insert(pose_sequence, 12, midhip, axis=1)
        
        print(f"✅ Added MidHip joint")
        print(f"   New shape: {pose_with_midhip.shape}")
        
        return pose_with_midhip
    
    def extract_important_joints(self, pose_sequence):
        """
        Ekstrak joint penting
        """
        indices = list(self.important_joint_indices.values())
        joint_names = list(self.important_joint_indices.keys())
        
        important_pose = pose_sequence[:, indices, :]
        
        print(f"✅ Extracted important joints:")
        print(f"   Joints: {joint_names}")
        print(f"   Shape: {important_pose.shape}")
        
        return important_pose, joint_names
    
    def visualize_pose_trajectory(self, pose_sequence, joint_names=None, save_path=None):
        """
        Visualisasi trajectory dari key joints
        """
        T, J, _ = pose_sequence.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Trajectory semua joints (top view)
        ax = axes[0, 0]
        for j in range(J):
            x = pose_sequence[:, j, 0]
            y = pose_sequence[:, j, 1]
            
            if not np.all(np.isnan(x)):
                label = joint_names[j] if joint_names else f'Joint {j}'
                ax.plot(x, y, 'o-', label=label, markersize=2, linewidth=0.5)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('2D Pose Trajectory (Top View)')
        ax.legend(fontsize=6, loc='best')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # 2. Vertical position over time (KEY!)
        ax = axes[0, 1]
        
        # Hip vertical position
        # Cari index hip
        if joint_names and 'RHip' in joint_names:
            hip_idx = joint_names.index('RHip')
        else:
            hip_idx = 4  # default
        
        y_hip = pose_sequence[:, hip_idx, 1]
        time = np.arange(T)
        
        ax.plot(time, y_hip, 'b-', linewidth=2, label='Hip Y position')
        ax.axhline(y=np.mean(y_hip[~np.isnan(y_hip)]), color='r', 
                   linestyle='--', label='Mean Y')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('⭐ Vertical Position Over Time (Hip) - FALL INDICATOR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        # 3. Confidence over time
        ax = axes[1, 0]
        
        avg_conf = np.nanmean(pose_sequence[:, :, 2], axis=1)
        
        ax.plot(time, avg_conf, 'g-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold 0.5')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Detection Confidence Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 4. Body height
        ax = axes[1, 1]
        
        nose_idx = joint_names.index('Nose') if joint_names and 'Nose' in joint_names else 0
        nose_y = pose_sequence[:, nose_idx, 1]
        
        height = np.abs(y_hip - nose_y)
        
        ax.plot(time, height, 'm-', linewidth=2)
        ax.axhline(y=np.nanmean(height), color='r', 
                   linestyle='--', label='Mean height')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Body Height (pixels)')
        ax.set_title('Body Height Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Plot saved: {save_path}")
        
        plt.close()
    
    def save_processed_pose(self, pose_sequence, output_path, 
                           joint_names=None, metadata=None):
        """
        Save processed pose sequence
        """
        save_dict = {
            'pose_2d': pose_sequence,
            'joint_names': joint_names if joint_names else [],
            'shape': pose_sequence.shape
        }
        
        if metadata:
            save_dict.update(metadata)
        
        np.savez_compressed(output_path, **save_dict)
        
        print(f"\n💾 Saved to: {output_path}")
        print(f"   Shape: {pose_sequence.shape}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Path disesuaikan dengan struktur Docker
    JSON_FOLDER = "/workspace/URFD/fall-10-cam1_keypoints"
    OUTPUT_FOLDER = "/workspace/processed_poses"
    
    # Buat output folder
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PROCESSING OPENPOSE OUTPUT: fall-10-cam1")
    print("=" * 60)
    
    # Initialize processor
    processor = URFDOpenPoseProcessor()
    
    # 1. Load JSON sequence
    print("\n🔄 Step 1: Loading JSON sequence...")
    pose_seq, frame_names = processor.load_json_sequence(JSON_FOLDER)
    
    if pose_seq is None:
        print("\n❌ Failed to load pose sequence!")
        exit()
    
    # 2. Interpolate missing poses
    print("\n🔄 Step 2: Interpolating missing poses...")
    pose_seq = processor.interpolate_missing_poses(pose_seq)
    
    # 3. Add MidHip joint
    print("\n🔄 Step 3: Adding MidHip joint...")
    pose_seq = processor.add_midhip_joint(pose_seq)
    
    # 4. Extract important joints
    print("\n🔄 Step 4: Extracting important joints...")
    pose_important, joint_names = processor.extract_important_joints(pose_seq)
    
    # 5. Visualize
    print("\n🔄 Step 5: Visualizing trajectory...")
    processor.visualize_pose_trajectory(
        pose_important,
        joint_names=joint_names,
        save_path=f"{OUTPUT_FOLDER}/fall-10-cam1-trajectory.png"
    )
    
    # 6. Save processed pose
    print("\n🔄 Step 6: Saving processed pose...")
    
    video_name = "fall-10-cam1"
    
    metadata = {
        'video_name': video_name,
        'num_frames': len(frame_names),
        'frame_names': frame_names
    }
    
    processor.save_processed_pose(
        pose_important,
        f"{OUTPUT_FOLDER}/{video_name}_pose2d.npz",
        joint_names=joint_names,
        metadata=metadata
    )
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  📊 {OUTPUT_FOLDER}/fall-10-cam1-trajectory.png")
    print(f"  💾 {OUTPUT_FOLDER}/{video_name}_pose2d.npz")
    print("\n📌 IMPORTANT: Check the plot!")
    print("   Look at 'Vertical Position Over Time' chart")
    print("   You should see a SUDDEN INCREASE in Y position (fall down)")
