"""
Visualisasi perbandingan sinyal IMU:
1. Real IMU vs OpenPose vs MediaPipe (sinyal time series)
2. Bar chart perbandingan detection rate
3. Bar chart perbandingan akurasi klasifikasi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import savgol_filter
import json

MP_IMU  = Path("/workspace/mediapipe_pipeline/synthetic_imu")
OP_IMU  = Path("/workspace/awal/synthetic_imu")
CSV_DIR = Path("/workspace/URFD/csv")
OUT_DIR = Path("/workspace/awal/visualisasi")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# ============================================================
# PLOT 1: Perbandingan Sinyal Time Series (ADL vs Fall)
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Perbandingan Sinyal IMU: ADL vs Fall', fontsize=14, fontweight='bold')

# Sample video yang representatif
adl_sample  = 'adl-09-cam0-rgb'
fall_sample = 'fall-08-cam0-rgb'

# --- Real IMU ---
def load_real(name):
    f = CSV_DIR / f"{name.replace('-cam0-rgb','').replace('-cam0','')}-acc.csv"
    if not f.exists():
        # coba tanpa cam0
        base = name.split('-cam')[0]
        f = CSV_DIR / f"{base}-acc.csv"
    data = np.loadtxt(f, delimiter=',')
    return np.linalg.norm(data[:, 1:4], axis=1) * 9.81

# Real IMU ADL
try:
    real_adl  = load_real(adl_sample)
    real_fall = load_real(fall_sample)
    t_adl  = np.linspace(0, len(real_adl)/50, len(real_adl))
    t_fall = np.linspace(0, len(real_fall)/50, len(real_fall))
    axes[0,0].plot(t_adl, real_adl, 'b-', linewidth=1.5)
    axes[0,0].set_title('Real IMU - ADL (Normal)', fontweight='bold')
    axes[0,0].axhline(y=np.mean(real_adl), color='r', linestyle='--', alpha=0.7, label=f'Mean={np.mean(real_adl):.1f}')
    axes[0,0].set_ylabel('Akselerasi (m/s²)')
    axes[0,0].legend(fontsize=9)
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t_fall, real_fall, 'r-', linewidth=1.5)
    axes[0,1].set_title('Real IMU - FALL (Jatuh)', fontweight='bold')
    axes[0,1].axhline(y=np.mean(real_fall), color='b', linestyle='--', alpha=0.7, label=f'Mean={np.mean(real_fall):.1f}')
    axes[0,1].set_ylabel('Akselerasi (m/s²)')
    axes[0,1].legend(fontsize=9)
    axes[0,1].grid(True, alpha=0.3)
except Exception as e:
    print(f"Real IMU error: {e}")

# OpenPose IMU
op_adl  = np.load(OP_IMU / f"{adl_sample}_imu.npy")[:, 2]
op_fall = np.load(OP_IMU / f"{fall_sample}_imu.npy")[:, 2]
t_op_adl  = np.linspace(0, len(op_adl)/30, len(op_adl))
t_op_fall = np.linspace(0, len(op_fall)/30, len(op_fall))

axes[1,0].plot(t_op_adl, op_adl, 'b-', linewidth=1.5)
axes[1,0].set_title('OpenPose Synthetic - ADL', fontweight='bold')
axes[1,0].axhline(y=np.mean(op_adl), color='r', linestyle='--', alpha=0.7, label=f'Mean={np.mean(op_adl):.1f}')
axes[1,0].set_ylabel('Akselerasi (pixel/s²)')
axes[1,0].legend(fontsize=9)
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(t_op_fall, op_fall, 'r-', linewidth=1.5)
axes[1,1].set_title('OpenPose Synthetic - FALL', fontweight='bold')
axes[1,1].axhline(y=np.mean(op_fall), color='b', linestyle='--', alpha=0.7, label=f'Mean={np.mean(op_fall):.1f}')
axes[1,1].set_ylabel('Akselerasi (pixel/s²)')
axes[1,1].legend(fontsize=9)
axes[1,1].grid(True, alpha=0.3)

# MediaPipe IMU
mp_adl  = np.load(MP_IMU / f"{adl_sample}_imu.npy")[:, 2]
mp_fall = np.load(MP_IMU / f"{fall_sample}_imu.npy")[:, 2]
t_mp_adl  = np.linspace(0, len(mp_adl)/30, len(mp_adl))
t_mp_fall = np.linspace(0, len(mp_fall)/30, len(mp_fall))

axes[2,0].plot(t_mp_adl, mp_adl, 'b-', linewidth=1.5)
axes[2,0].set_title('MediaPipe Synthetic - ADL', fontweight='bold')
axes[2,0].axhline(y=np.mean(mp_adl), color='r', linestyle='--', alpha=0.7, label=f'Mean={np.mean(mp_adl):.1f}')
axes[2,0].set_ylabel('Akselerasi (pixel/s²)')
axes[2,0].set_xlabel('Waktu (detik)')
axes[2,0].legend(fontsize=9)
axes[2,0].grid(True, alpha=0.3)

axes[2,1].plot(t_mp_fall, mp_fall, 'r-', linewidth=1.5)
axes[2,1].set_title('MediaPipe Synthetic - FALL', fontweight='bold')
axes[2,1].axhline(y=np.mean(mp_fall), color='b', linestyle='--', alpha=0.7, label=f'Mean={np.mean(mp_fall):.1f}')
axes[2,1].set_ylabel('Akselerasi (pixel/s²)')
axes[2,1].set_xlabel('Waktu (detik)')
axes[2,1].legend(fontsize=9)
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot1_sinyal_imu.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 1 selesai: sinyal time series")

# ============================================================
# PLOT 2: Bar Chart Perbandingan Semua Metrik
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Perbandingan Pipeline OpenPose vs MediaPipe vs Real IMU', 
             fontsize=14, fontweight='bold')

colors = ['#2196F3', '#FF5722', '#4CAF50']
labels = ['Real IMU', 'OpenPose', 'MediaPipe']

# --- Detection Rate ---
ax = axes[0]
detection = [100, 84.2, 100]
bars = ax.bar(labels, detection, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Detection Rate (%)', fontweight='bold')
ax.set_ylabel('Detection Rate (%)')
ax.set_ylim([0, 115])
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, detection):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# --- Ratio FALL/ADL ---
ax = axes[1]
ratios = [3.19, 1.27, 3.18]
bars = ax.bar(labels, ratios, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Ratio Peak FALL/ADL\n(lebih tinggi = lebih baik)', fontweight='bold')
ax.set_ylabel('Ratio FALL/ADL')
ax.set_ylim([0, 4.0])
ax.axhline(y=3.19, color='#2196F3', linestyle='--', alpha=0.7, label='Real IMU target')
for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}x', ha='center', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# --- SVM Accuracy ---
ax = axes[2]
accuracy = [95.7, 80.0, 95.7]
bars = ax.bar(labels, accuracy, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('SVM Accuracy\nFall Detection (%)', fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([0, 110])
ax.axhline(y=95.7, color='#2196F3', linestyle='--', alpha=0.7, label='Real IMU baseline')
for bar, val in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot2_perbandingan_metrik.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 selesai: bar chart perbandingan")

# ============================================================
# PLOT 3: Distribusi IMU ADL vs Fall (Box Plot)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Distribusi Magnitude IMU: ADL vs Fall', fontsize=14, fontweight='bold')

def get_all_means(imu_dir, pattern):
    vals = []
    for f in sorted(imu_dir.glob(pattern)):
        imu = np.load(f)
        if np.isnan(imu).any(): continue
        vals.append(imu[:, 2].mean())
    return vals

# OpenPose
op_adl_means  = get_all_means(OP_IMU, 'adl*cam0*_imu.npy')
op_fall_means = get_all_means(OP_IMU, 'fall*cam0*_imu.npy')

# MediaPipe
mp_adl_means  = get_all_means(MP_IMU, 'adl*cam0*_imu.npy')
mp_fall_means = get_all_means(MP_IMU, 'fall*cam0*_imu.npy')

# Box plot OpenPose
ax = axes[0]
bp = ax.boxplot([op_adl_means, op_fall_means],
                labels=['ADL', 'FALL'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#90CAF9')
bp['boxes'][1].set_facecolor('#EF9A9A')
ax.set_title('OpenPose Synthetic IMU', fontweight='bold')
ax.set_ylabel('Mean Magnitude (pixel/s²)')
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.05, 0.95, f'Ratio: {np.mean(op_fall_means)/np.mean(op_adl_means):.2f}x',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='top', color='darkred')

# Box plot MediaPipe
ax = axes[1]
bp = ax.boxplot([mp_adl_means, mp_fall_means],
                labels=['ADL', 'FALL'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#90CAF9')
bp['boxes'][1].set_facecolor('#EF9A9A')
ax.set_title('MediaPipe Synthetic IMU', fontweight='bold')
ax.set_ylabel('Mean Magnitude (pixel/s²)')
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.05, 0.95, f'Ratio: {np.mean(mp_fall_means)/np.mean(mp_adl_means):.2f}x',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='top', color='darkred')

plt.tight_layout()
plt.savefig(OUT_DIR / 'plot3_distribusi_imu.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 selesai: distribusi box plot")

print(f"\nSemua plot tersimpan di: {OUT_DIR}")
print("File:")
for f in sorted(OUT_DIR.glob('*.png')):
    print(f"  {f.name}")
