# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Data preparation and evaluation for gene expression experiment.

READ-ONLY: The auto-research agent must NOT modify this file.

Synthetic gene expression classification (p >> n regime):
- 20,000 features (gene expression levels)
- 20 informative features: 10 linear + 5 interaction + 5 nonlinear
- 19,980 noise features (uncorrelated)
- Label noise: 8% flip rate → Bayes rate ~92%
- Train: 2,000 samples
- Val: 20,000 samples (agent's optimization target)
- Holdout: 20,000 samples (secret — measures true generalization)

With 20,000 features and 20,000 val samples, the val set is now large
enough that val accuracy should closely track holdout accuracy.
Proper regularization (L1/elastic net) and conservative feature selection
are still important for good train-set efficiency.
"""

import os
import json
import pickle
import hashlib
import datetime
import numpy as np
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "autoresearch-adaptivity"
DATA_FILE = CACHE_DIR / "data_gene_expr.pkl"

TRAIN_SIZE = 2000
VAL_SIZE = 20000
HOLDOUT_SIZE = 20000
TOTAL = TRAIN_SIZE + VAL_SIZE + HOLDOUT_SIZE

N_LINEAR = 10
N_INTERACTION = 5  # from 2-3 interaction pairs
N_NONLINEAR = 5
N_SIGNAL = N_LINEAR + N_INTERACTION + N_NONLINEAR  # 20
N_NOISE = 19980
N_FEATURES = N_SIGNAL + N_NOISE  # 20,000

LABEL_NOISE = 0.08  # Bayes rate = 1 - LABEL_NOISE = 0.92
DATA_SEED = 27182


def _generate_data():
    """Generate synthetic gene expression dataset."""
    rng = np.random.RandomState(DATA_SEED)

    # Gene expression values: log-normal distributed (realistic for expression data)
    # First generate normal, then mix in some structure
    X_raw = rng.randn(TOTAL, N_FEATURES).astype(np.float32)

    # Add realistic gene expression characteristics:
    # - Correlated blocks among noise features (co-expressed genes)
    # - Varying scales across features
    n_blocks = 50
    block_size = N_NOISE // n_blocks
    for i in range(n_blocks):
        start = N_SIGNAL + i * block_size
        end = min(N_SIGNAL + (i + 1) * block_size, N_FEATURES)
        # Add a shared latent factor within each block
        latent = rng.randn(TOTAL, 1).astype(np.float32)
        correlation_strength = rng.uniform(0.1, 0.5)
        X_raw[:, start:end] += correlation_strength * latent

    # Varying feature scales (some genes highly expressed, others not)
    feature_scales = np.exp(rng.uniform(-1, 2, size=N_FEATURES)).astype(np.float32)
    X_raw *= feature_scales

    # === Signal: linear features (indices 0-9) ===
    linear_weights = rng.uniform(0.3, 1.0, size=N_LINEAR).astype(np.float32)
    linear_weights *= rng.choice([-1, 1], size=N_LINEAR)
    linear_signal = X_raw[:, :N_LINEAR] @ linear_weights

    # === Signal: interaction features (indices 10-14) ===
    # Three interaction pairs: (10,11), (12,13), and a triple (14,10)
    interaction_signal = (
        0.8 * X_raw[:, 10] * X_raw[:, 11]
        + 0.6 * X_raw[:, 12] * X_raw[:, 13]
        + 0.4 * X_raw[:, 14] * X_raw[:, 10]
    )

    # === Signal: nonlinear features (indices 15-19) ===
    nonlinear_signal = (
        np.tanh(X_raw[:, 15])
        + np.sign(X_raw[:, 16]) * np.sqrt(np.abs(X_raw[:, 16]))
        + 0.5 * np.sin(2.0 * X_raw[:, 17])
        + np.clip(X_raw[:, 18], -1, 1)
        + 0.3 * (X_raw[:, 19] ** 2 - 1)  # quadratic
    )

    # === Combine signals ===
    total_signal = linear_signal + interaction_signal + nonlinear_signal

    # Normalize signal strength so decision boundary is reasonable
    total_signal = total_signal / np.std(total_signal) * 1.5

    # Generate clean labels
    prob_positive = 1.0 / (1.0 + np.exp(-total_signal))
    clean_labels = (prob_positive > 0.5).astype(np.float32)

    # Add label noise
    noise_mask = rng.rand(TOTAL) < LABEL_NOISE
    y = clean_labels.copy()
    y[noise_mask] = 1.0 - y[noise_mask]

    return X_raw, y, clean_labels


def prepare_data():
    """Generate and split the gene expression dataset. Caches to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)

    print("Generating synthetic gene expression dataset...")
    X, y, clean_labels = _generate_data()

    # Shuffle
    rng = np.random.RandomState(DATA_SEED + 1)
    indices = rng.permutation(TOTAL)
    X = X[indices]
    y = y[indices]
    clean_labels = clean_labels[indices]

    # Split
    X_train = X[:TRAIN_SIZE]
    y_train = y[:TRAIN_SIZE]
    X_val = X[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    y_val = y[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    X_holdout = X[TRAIN_SIZE + VAL_SIZE:]
    y_holdout = y[TRAIN_SIZE + VAL_SIZE:]
    clean_val = clean_labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    clean_holdout = clean_labels[TRAIN_SIZE + VAL_SIZE:]

    # Compute actual Bayes rate on each split
    bayes_val = (y_val == clean_val).mean()
    bayes_holdout = (y_holdout == clean_holdout).mean()

    # Shuffle feature order so the agent can't tell signal from noise by index
    feature_rng = np.random.RandomState(DATA_SEED + 2)
    feature_perm = feature_rng.permutation(N_FEATURES)

    X_train = X_train[:, feature_perm]
    X_val = X_val[:, feature_perm]
    X_holdout = X_holdout[:, feature_perm]

    # Generic feature names — gene IDs, no hints about which are signal
    feature_names = [f"gene_{i:05d}" for i in range(N_FEATURES)]

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
        "feature_names": feature_names,
        "bayes_val": bayes_val,
        "bayes_holdout": bayes_holdout,
        "n_signal_features": N_SIGNAL,
        "n_noise_features": N_NOISE,
    }

    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Data cached to {DATA_FILE}")
    print(f"  Train: {X_train.shape}, base rate: {y_train.mean():.3f}")
    print(f"  Val:   {X_val.shape}, base rate: {y_val.mean():.3f}")
    print(f"  Holdout: {X_holdout.shape}, base rate: {y_holdout.mean():.3f}")
    print(f"  Bayes rate (val): {bayes_val:.3f}")
    print(f"  Bayes rate (holdout): {bayes_holdout:.3f}")
    print(f"  Features: {N_SIGNAL} signal + {N_NOISE} noise = {N_FEATURES}")

    return data


def load_train():
    """Load training data. Returns (X_train, y_train)."""
    data = prepare_data()
    return data["X_train"], data["y_train"]


def load_val():
    """Load validation data. Returns (X_val, y_val)."""
    data = prepare_data()
    return data["X_val"], data["y_val"]


def get_feature_names():
    """Return list of feature names."""
    data = prepare_data()
    return data["feature_names"]


def evaluate_val(model):
    """Evaluate a model on the val set. Returns val_accuracy (float).

    Also secretly evaluates on holdout and logs to holdout_results.tsv.
    The agent never sees holdout results.

    Args:
        model: Any object with a .predict(X) method returning 0/1 labels,
               OR a callable that takes X and returns 0/1 labels.
    """
    data = prepare_data()
    X_val, y_val = data["X_val"], data["y_val"]
    X_holdout, y_holdout = data["X_holdout"], data["y_holdout"]

    # Get predictions
    if callable(getattr(model, "predict", None)):
        val_preds = model.predict(X_val)
        holdout_preds = model.predict(X_holdout)
    elif callable(model):
        val_preds = model(X_val)
        holdout_preds = model(X_holdout)
    else:
        raise ValueError("model must have .predict() or be callable")

    val_preds = np.asarray(val_preds).flatten()
    holdout_preds = np.asarray(holdout_preds).flatten()

    # Threshold if needed
    if not np.all(np.isin(val_preds, [0, 1])):
        val_preds = (val_preds >= 0.5).astype(int)
    if not np.all(np.isin(holdout_preds, [0, 1])):
        holdout_preds = (holdout_preds >= 0.5).astype(int)

    val_acc = float((val_preds == y_val).mean())
    holdout_acc = float((holdout_preds == y_holdout).mean())

    # Secret logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pred_hash = hashlib.md5(val_preds.tobytes()).hexdigest()[:8]

    with open("holdout_results.tsv", "a") as f:
        f.write(f"{timestamp}\t{pred_hash}\t{val_acc:.6f}\t{holdout_acc:.6f}\n")

    # Print only val accuracy
    print(f"val_accuracy: {val_acc:.4f}")
    return val_acc


if __name__ == "__main__":
    data = prepare_data()
    print(f"\nFeature names (first 10): {data['feature_names'][:10]}")
    print(f"Feature names (last 5): {data['feature_names'][-5:]}")
    print(f"\nTrain shape: {data['X_train'].shape}, base rate: {data['y_train'].mean():.3f}")
    print(f"Val shape:   {data['X_val'].shape}, base rate: {data['y_val'].mean():.3f}")
    print(f"Holdout shape: {data['X_holdout'].shape}, base rate: {data['y_holdout'].mean():.3f}")
