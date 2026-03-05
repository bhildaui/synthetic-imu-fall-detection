"""
Analisis detail dari pose 2D yang sudah diproses
"""

import numpy as np
import matplotlib.pyplot as plt

# Path yang BENAR
NPZ_FILE = "/workspace/processed_poses/fall-10-cam1_pose2d.npz"

print("=" * 60)
print("ANALYZING POSE 2D DATA")
print("=" * 60)

# Load data
data = np.load(NPZ_FILE)

print("\n📂 File contents:")
for key in data.keys():
    print(f"   - {key}")

# Extract data
pose_2d = data['pose_2d']  # Shape: (T, J, 3)
joint_names = data['joint_names'].tolist()

print(f"\n📊 Data shape:")
print(f"   pose_2d: {pose_2d.shape}")
print(f"   Frames (T): {pose_2d.shape[0]}")
print(f"   Joints (J): {pose_2d.shape[1]}")
print(f"   Coords: {pose_2d.shape[2]} (x, y, confidence)")

print(f"\n🦴 Joints:")
for i, name in enumerate(joint_names):
    print(f"   [{i}] {name}")

# Analyze Hip vertical position
print("\n" + "=" * 60)
print("FALL DETECTION ANALYSIS")
print("=" * 60)

# Find RHip index
if 'RHip' in joint_names:
    hip_idx = joint_names.index('RHip')
else:
    hip_idx = 4  # default

print(f"\nAnalyzing joint: {joint_names[hip_idx]}")

# Get Y coordinate (vertical position)
y_hip = pose_2d[:, hip_idx, 1]

# Statistics
print(f"\n📊 Vertical Position (Y coordinate):")
print(f"   Min Y: {np.min(y_hip):.1f} pixels")
print(f"   Max Y: {np.max(y_hip):.1f} pixels")
print(f"   Mean Y: {np.mean(y_hip):.1f} pixels")
print(f"   Std Y: {np.std(y_hip):.1f} pixels")
print(f"   Range: {np.max(y_hip) - np.min(y_hip):.1f} pixels")

# Velocity (first derivative)
dy = np.diff(y_hip)

print(f"\n📈 Vertical Velocity (dy/dt):")
print(f"   Max downward: {np.max(dy):.1f} pixels/frame (↓)")
print(f"   Max upward: {np.min(dy):.1f} pixels/frame (↑)")
print(f"   Mean velocity: {np.mean(dy):.1f} pixels/frame")

# Detect sudden changes
threshold = 20  # pixels per frame
sudden_moves = np.where(np.abs(dy) > threshold)[0]

print(f"\n⚡ Sudden movements (>{threshold} px/frame):")
if len(sudden_moves) > 0:
    print(f"   Found {len(sudden_moves)} sudden movements")
    for idx in sudden_moves[:10]:  # Show max 10
        direction = "DOWN ↓" if dy[idx] > 0 else "UP ↑"
        print(f"   Frame {idx:3d} → {idx+1:3d}: {dy[idx]:+7.1f} px ({direction})")
else:
    print(f"   No sudden movements detected")

# Fall pattern detection
# Pattern: stable high position → sudden drop → stable low position
print(f"\n🎯 FALL PATTERN DETECTION:")

# Divide into 3 segments
segment_size = len(y_hip) // 3
seg1 = y_hip[:segment_size]
seg2 = y_hip[segment_size:2*segment_size]
seg3 = y_hip[2*segment_size:]

seg1_mean = np.mean(seg1)
seg2_mean = np.mean(seg2)
seg3_mean = np.mean(seg3)

print(f"   Segment 1 (frames 0-{segment_size}): Mean Y = {seg1_mean:.1f}")
print(f"   Segment 2 (frames {segment_size}-{2*segment_size}): Mean Y = {seg2_mean:.1f}")
print(f"   Segment 3 (frames {2*segment_size}-{len(y_hip)}): Mean Y = {seg3_mean:.1f}")

# Check if there's a clear drop
drop_1_to_3 = seg3_mean - seg1_mean

print(f"\n   Total drop (Seg 1 → Seg 3): {drop_1_to_3:+.1f} pixels")

if drop_1_to_3 > 80:
    print(f"   ✅ FALL DETECTED! (person moved DOWN significantly)")
elif drop_1_to_3 < -80:
    print(f"   ⚠️  Unusual: person moved UP significantly")
else:
    print(f"   ⚠️  No clear fall pattern")

# Find exact fall moment
if drop_1_to_3 > 80:
    # Find frame with maximum downward velocity
    max_fall_frame = np.argmax(dy)
    print(f"\n   💥 Fall impact estimated at frame: {max_fall_frame}")
    print(f"      Velocity at impact: {dy[max_fall_frame]:+.1f} pixels/frame")

# Confidence analysis
avg_conf = np.mean(pose_2d[:, :, 2])
min_conf = np.min(pose_2d[:, :, 2])
max_conf = np.max(pose_2d[:, :, 2])

print(f"\n🎯 Detection Confidence:")
print(f"   Average: {avg_conf:.3f}")
print(f"   Min: {min_conf:.3f}")
print(f"   Max: {max_conf:.3f}")

if avg_conf > 0.7:
    quality = "EXCELLENT"
elif avg_conf > 0.5:
    quality = "GOOD"
else:
    quality = "FAIR"

print(f"   Quality: {quality}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n✅ 2D Pose Estimation: SUCCESS")
print(f"   - All {pose_2d.shape[0]} frames detected")
print(f"   - Average confidence: {avg_conf:.3f} ({quality})")
print(f"   - Fall pattern: {'DETECTED' if drop_1_to_3 > 80 else 'NOT CLEAR'}")

if drop_1_to_3 > 80:
    print(f"\n🎯 READY FOR NEXT STEP: 3D Pose Lifting")
    print(f"   This 2D pose data is valid for fall detection research!")
else:
    print(f"\n⚠️  May need to check video or re-run OpenPose")

print("\n" + "=" * 60)
