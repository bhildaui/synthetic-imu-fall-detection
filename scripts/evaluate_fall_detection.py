"""
Evaluasi Fall Detection menggunakan Synthetic IMU vs Real IMU
Classifier: SVM dan Random Forest
Metode evaluasi: Leave-One-Out Cross Validation (LOOCV)
sesuai dengan standar penelitian fall detection
"""

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

MP_IMU  = Path("/workspace/mediapipe_pipeline/synthetic_imu")
OP_IMU  = Path("/workspace/awal/synthetic_imu")
CSV_DIR = Path("/workspace/URFD/csv")

# ============================================================
# FUNGSI EKSTRAK FITUR
# ============================================================
def extract_features(signal):
    """
    Ekstrak fitur statistik dari sinyal IMU.
    Sesuai dengan paper fall detection standar.
    
    signal: array (N, 3) atau (N,) untuk magnitude
    """
    if signal.ndim == 2:
        mag = signal[:, 2] if signal.shape[1] > 2 else np.linalg.norm(signal[:, :2], axis=1)
    else:
        mag = signal

    features = [
        mag.mean(),           # rata-rata akselerasi
        mag.std(),            # standar deviasi
        mag.max(),            # nilai puncak
        mag.min(),            # nilai minimum
        mag.max() - mag.min(),# range
        np.percentile(mag, 75) - np.percentile(mag, 25),  # IQR
        np.percentile(mag, 90),  # persentil 90
        np.percentile(mag, 95),  # persentil 95
        # Fitur temporal - perubahan akselerasi
        np.diff(mag).std(),   # variabilitas perubahan
        np.diff(mag).max(),   # perubahan maksimum
    ]
    return np.array(features)

def load_synthetic_imu(imu_dir, pattern_adl='adl*cam0*_imu.npy',
                       pattern_fall='fall*cam0*_imu.npy'):
    """Load synthetic IMU dan ekstrak fitur"""
    X, y = [], []

    for f in sorted(imu_dir.glob(pattern_adl)):
        imu = np.load(f)
        if np.isnan(imu).any(): continue
        X.append(extract_features(imu))
        y.append(0)  # 0 = ADL (tidak jatuh)

    for f in sorted(imu_dir.glob(pattern_fall)):
        imu = np.load(f)
        if np.isnan(imu).any(): continue
        X.append(extract_features(imu))
        y.append(1)  # 1 = Fall (jatuh)

    return np.array(X), np.array(y)

def load_real_imu(csv_dir):
    """Load real IMU dari CSV dan ekstrak fitur"""
    X, y = [], []

    for f in sorted(csv_dir.glob('adl-*-acc.csv')):
        try:
            data = np.loadtxt(f, delimiter=',')
            mag  = np.linalg.norm(data[:, 1:4], axis=1) * 9.81
            X.append(extract_features(mag))
            y.append(0)
        except: pass

    for f in sorted(csv_dir.glob('fall-*-acc.csv')):
        try:
            data = np.loadtxt(f, delimiter=',')
            mag  = np.linalg.norm(data[:, 1:4], axis=1) * 9.81
            X.append(extract_features(mag))
            y.append(1)
        except: pass

    return np.array(X), np.array(y)

# ============================================================
# EVALUASI
# ============================================================
def evaluate(X, y, name):
    """Evaluasi dengan SVM dan Random Forest menggunakan LOOCV"""
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"Samples: {len(y)} ({sum(y==0)} ADL, {sum(y==1)} Fall)")
    print(f"{'='*50}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for clf_name, clf in [
        ("SVM (RBF)", SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))
    ]:
        loo    = LeaveOneOut()
        scores = cross_val_score(clf, X_scaled, y, cv=loo, scoring='accuracy')
        f1     = cross_val_score(clf, X_scaled, y, cv=loo, scoring='f1')

        print(f"\n{clf_name}:")
        print(f"  Accuracy : {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
        print(f"  F1 Score : {f1.mean():.3f} ± {f1.std():.3f}")

        results[clf_name] = {
            'accuracy': scores.mean()*100,
            'f1': f1.mean()
        }

    return results

# ============================================================
# MAIN
# ============================================================
print("EVALUASI FALL DETECTION - SYNTHETIC IMU vs REAL IMU")
print("Metode: Leave-One-Out Cross Validation (LOOCV)")
print("Classifier: SVM dan Random Forest")

# 1. Real IMU
X_real, y_real = load_real_imu(CSV_DIR)
r = evaluate(X_real, y_real, "Real IMU (Sensor Fisik)")

# 2. OpenPose IMU
X_op, y_op = load_synthetic_imu(OP_IMU)
o = evaluate(X_op, y_op, "OpenPose Synthetic IMU")

# 3. MediaPipe IMU
X_mp, y_mp = load_synthetic_imu(MP_IMU)
m = evaluate(X_mp, y_mp, "MediaPipe Synthetic IMU")

# Ringkasan
print(f"\n{'='*60}")
print("RINGKASAN HASIL")
print(f"{'='*60}")
print(f"{'Dataset':25s} {'SVM Acc':10s} {'RF Acc':10s} {'SVM F1':8s} {'RF F1':8s}")
print(f"{'-'*60}")
for name, res in [("Real IMU", r), ("OpenPose", o), ("MediaPipe", m)]:
    svm_acc = res['SVM (RBF)']['accuracy']
    rf_acc  = res['Random Forest']['accuracy']
    svm_f1  = res['SVM (RBF)']['f1']
    rf_f1   = res['Random Forest']['f1']
    print(f"{name:25s} {svm_acc:8.1f}%  {rf_acc:8.1f}%  {svm_f1:6.3f}   {rf_f1:6.3f}")

