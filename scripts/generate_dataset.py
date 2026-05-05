"""
Dataset Generator — Synthetic High-Dimensional Classification Dataset
=====================================================================
Generates a 1000×1000 dataset (1000 samples, 1000 features) with a
binary classification target derived from non-linear feature interactions.

Uses ONLY Python standard library (no numpy/pandas required).

The dataset includes:
  • 100 informative features (truly predictive, strong signal)
  • 200 redundant features (linear combos of informative)
  • 700 noise features
  • Non-linear decision boundary with clear separation
  • Saved as CSV for consumption by the Julia ML pipeline
"""

import csv
import math
import os
import random
import statistics
import time

SEED = 42
N_SAMPLES = 1000
N_FEATURES = 1000
N_INFORMATIVE = 100
N_REDUNDANT = 200

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_dataset.csv")
METADATA_FILE = os.path.join(DATA_DIR, "dataset_metadata.txt")

random.seed(SEED)

def randn():
    """Generate a single standard normal random variable (Box-Muller)."""
    u1 = random.random()
    u2 = random.random()
    while u1 == 0:
        u1 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

def generate_dataset():
    print("=" * 70)
    print("  SYNTHETIC DATASET GENERATOR")
    print("  1000 samples × 1000 features — Binary Classification")
    print("  (Pure Python — no external dependencies)")
    print("=" * 70)
    start = time.time()

    print("\n[1/7] Generating informative features...")
    n_per_class = N_SAMPLES // 2
    labels_raw = [0] * n_per_class + [1] * n_per_class

    random.seed(SEED)
    
    center_0 = [randn() * 0.5 for _ in range(N_INFORMATIVE)]
    center_1 = [center_0[i] + (1.5 + abs(randn()) * 0.5) * (1 if i % 2 == 0 else -1) 
                for i in range(N_INFORMATIVE)]

    X_informative = []
    for i in range(N_SAMPLES):
        label = labels_raw[i]
        if label == 0:
            row = [center_0[j] + randn() * 0.6 for j in range(N_INFORMATIVE)]
        else:
            row = [center_1[j] + randn() * 0.6 for j in range(N_INFORMATIVE)]
        X_informative.append(row)

    print("[2/7] Creating redundant features...")
    mixing = [[randn() * 0.3 for _ in range(N_REDUNDANT)] for _ in range(N_INFORMATIVE)]

    X_redundant = []
    for i in range(N_SAMPLES):
        row = []
        for j in range(N_REDUNDANT):
            val = sum(X_informative[i][k] * mixing[k][j] for k in range(N_INFORMATIVE))
            val += randn() * 0.05
            row.append(val)
        X_redundant.append(row)

    n_noise = N_FEATURES - N_INFORMATIVE - N_REDUNDANT
    print(f"[3/7] Generating {n_noise} noise features...")
    noise_scales = [random.uniform(0.5, 2.0) for _ in range(n_noise)]
    X_noise = [[randn() * noise_scales[j] for j in range(n_noise)] for _ in range(N_SAMPLES)]

    print("[4/7] Assembling and shuffling feature matrix...")
    X = []
    for i in range(N_SAMPLES):
        row = X_informative[i] + X_redundant[i] + X_noise[i]
        X.append(row)

    col_perm = list(range(N_FEATURES))
    random.shuffle(col_perm)
    X = [[row[col_perm[j]] for j in range(N_FEATURES)] for row in X]

    print("[5/7] Computing non-linear target variable...")
    z = [0.0] * N_SAMPLES

    for i in range(N_SAMPLES):
        xi = X_informative[i]

        for k in range(0, min(40, N_INFORMATIVE), 2):
            z[i] += xi[k] * xi[k + 1] * 0.3

        for k in range(0, N_INFORMATIVE):
            z[i] += xi[k] * 0.15

        for k in range(40, min(60, N_INFORMATIVE)):
            z[i] += math.sin(xi[k] * 1.5) * 0.5

        for k in range(60, min(80, N_INFORMATIVE)):
            z[i] += xi[k] ** 2 * 0.1

        for k in range(80, N_INFORMATIVE):
            z[i] += (1.5 if xi[k] > 0 else -1.5)

        z[i] += randn() * 0.3

    median_z = statistics.median(z)
    y = [1 if zi > median_z else 0 for zi in z]

    print("[6/7] Shuffling and splitting into train/test...")
    indices = list(range(N_SAMPLES))
    random.shuffle(indices)

    split_idx = int(N_SAMPLES * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    print("[7/7] Saving datasets to CSV...")
    os.makedirs(DATA_DIR, exist_ok=True)

    feature_names = [f"feature_{i:04d}" for i in range(N_FEATURES)]
    header = feature_names + ["target"]

    def write_csv(filepath, row_indices):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx in row_indices:
                row = [f"{X[idx][j]:.8f}" for j in range(N_FEATURES)] + [str(y[idx])]
                writer.writerow(row)

    write_csv(TRAIN_FILE, train_indices)
    write_csv(TEST_FILE, test_indices)

    train_y = [y[i] for i in train_indices]
    test_y = [y[i] for i in test_indices]
    train_pos = sum(train_y)
    train_neg = len(train_y) - train_pos
    test_pos = sum(test_y)
    test_neg = len(test_y) - test_pos

    all_vals = []
    for idx in train_indices[:100]:
        all_vals.extend(X[idx])
    mean_val = statistics.mean(all_vals)
    std_val = statistics.stdev(all_vals)
    min_val = min(all_vals)
    max_val = max(all_vals)

    elapsed = time.time() - start

    metadata = f"""Dataset Metadata
================
Total samples:          {N_SAMPLES}
Total features:         {N_FEATURES}
Informative features:   {N_INFORMATIVE}
Redundant features:     {N_REDUNDANT}
Noise features:         {N_FEATURES - N_INFORMATIVE - N_REDUNDANT}
Task:                   Binary Classification
Random seed:            {SEED}
Generation time:        {elapsed:.1f}s

Training Set ({len(train_indices)} samples):
  Class 0: {train_neg}
  Class 1: {train_pos}

Test Set ({len(test_indices)} samples):
  Class 0: {test_neg}
  Class 1: {test_pos}

Feature statistics (sample):
  Mean:  {mean_val:.6f}
  Std:   {std_val:.6f}
  Min:   {min_val:.6f}
  Max:   {max_val:.6f}
"""

    with open(METADATA_FILE, "w") as f:
        f.write(metadata)

    print(f"\n{'─' * 70}")
    print(f"   Training data saved: {TRAIN_FILE}")
    print(f"     └─ Shape: ({len(train_indices)}, {N_FEATURES + 1})")
    print(f"   Test data saved:     {TEST_FILE}")
    print(f"     └─ Shape: ({len(test_indices)}, {N_FEATURES + 1})")
    print(f"   Metadata saved:      {METADATA_FILE}")
    print(f"  ⏱  Generated in {elapsed:.1f}s")
    print(f"{'─' * 70}")
    print(metadata)

if __name__ == "__main__":
    generate_dataset()
