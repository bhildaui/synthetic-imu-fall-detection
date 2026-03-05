import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load synthetic IMU
synthetic = np.load('/workspace/awal/synthetic_imu/fall-01-cam0_imu.npy')

# Hitung SVM dari synthetic
# SVM = sqrt(ax^2 + ay^2 + az^2)
synthetic_svm = np.sqrt(np.sum(synthetic**2, axis=1))

# Load ground truth
gt = pd.read_csv('/workspace/URFD/fall-01-data.csv',
                 header=None,
                 names=['idx', 'timestamp_ms', 'svm'])

# Plot perbandingan
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Ground truth
axes[0].plot(gt['svm'].values, color='blue', label='Ground Truth SVM')
axes[0].axhline(y=2.0, color='r', linestyle='--', label='Fall threshold')
axes[0].set_title('Ground Truth Accelerometer SVM - fall-01')
axes[0].set_ylabel('SVM (g)')
axes[0].legend()
axes[0].grid(True)

# Synthetic
axes[1].plot(synthetic_svm, color='orange', label='Synthetic IMU SVM')
axes[1].axhline(y=2.0, color='r', linestyle='--', label='Fall threshold')
axes[1].set_title('Synthetic IMU SVM - fall-01')
axes[1].set_ylabel('SVM (g)')
axes[1].set_xlabel('Frame')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('/workspace/awal/compare_imu_fall01.png')
print("Plot saved!")
print(f"GT SVM max: {gt['svm'].max():.3f}")
print(f"Synthetic SVM max: {synthetic_svm.max():.3f}")