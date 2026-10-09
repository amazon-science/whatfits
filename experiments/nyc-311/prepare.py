# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Data preparation and evaluation for the NYC 311 (January 2026) experiment.

READ-ONLY: The auto-research agent must NOT modify this file.

Real tabular data from NYC Open Data 311 Service Requests (erm2-nwe9),
filtered to created_date in January 2026 (post-dates the model's training
cutoff of August 2025).

Task: predict whether a 311 service request is CLOSED WITHIN 72 HOURS of
creation, using ONLY information available at request-creation time.
  target = (closed_date - created_date) <= 72 hours

Leakage discipline: features use only creation-time metadata (agency,
complaint type/descriptor, location type, geography, channel, and derived
created-time fields). Resolution/SLA fields (status, due_date,
resolution_description, update timestamps) are NOT used, and closed_date is
used ONLY to build the label.

- Train: 10,000 examples
- Val: 20,000 examples (the agent's optimization target)
- Holdout: 20,000 examples (secret — measures true generalization)

Categorical features are integer-coded (stable factorization); numeric
features are median-imputed. The agent receives a dense float32 matrix plus
feature_names and is free to re-encode, select features, and choose models.
"""

import os
import datetime
import hashlib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "autoresearch-nyc311"
DATA_FILE = CACHE_DIR / "data_nyc311.pkl"
RAW_CSV = CACHE_DIR / "raw_nyc_311_jan2026.csv"

TRAIN_SIZE = 10000
VAL_SIZE = 20000
HOLDOUT_SIZE = 20000
DATA_SEED = 42

CLOSURE_THRESHOLD_HOURS = 72.0

# Creation-time features only. Order here defines the column order of X.
CATEGORICAL_COLS = [
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "city",
    "borough",
    "community_board",
    "council_district",
    "police_precinct",
    "open_data_channel_type",
]
NUMERIC_RAW_COLS = ["latitude", "longitude"]
# Derived-from-created_date numeric features (always available at creation).
DERIVED_NUMERIC_COLS = ["created_hour", "created_dow", "created_day", "created_week"]


def _build_from_raw():
    """Load the staged raw CSV, build features + label, and split."""
    if not RAW_CSV.exists():
        raise RuntimeError(
            f"Raw CSV not found: {RAW_CSV}\n"
            f"Stage the January-2026 NYC 311 extract there first: "
            f"cd experiments/nyc-311 && uv run python download_data.py"
        )

    df = pd.read_csv(RAW_CSV, low_memory=False)

    created = pd.to_datetime(df["created_date"], errors="coerce")
    closed = pd.to_datetime(df["closed_date"], errors="coerce")

    # Label: only rows with a usable closed_date are labeled.
    duration_h = (closed - created).dt.total_seconds() / 3600.0
    labeled = created.notna() & closed.notna() & (duration_h >= 0)
    df = df[labeled].reset_index(drop=True)
    created = created[labeled].reset_index(drop=True)
    duration_h = duration_h[labeled].reset_index(drop=True)

    y = (duration_h <= CLOSURE_THRESHOLD_HOURS).astype(int).to_numpy()

    # Derived creation-time numeric features.
    df["created_hour"] = created.dt.hour
    df["created_dow"] = created.dt.dayofweek
    df["created_day"] = created.dt.day
    df["created_week"] = created.dt.isocalendar().week.astype(int)

    feature_names = []
    columns = []

    # Categorical -> stable integer codes. Missing -> its own code (-1 shifted).
    for col in CATEGORICAL_COLS:
        raw = df[col].astype("string").fillna("__MISSING__")
        # Deterministic factorization by sorted unique value.
        uniques = sorted(raw.unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        codes = raw.map(mapping).to_numpy(dtype=np.float32)
        columns.append(codes)
        feature_names.append(col)

    # Numeric raw features -> median imputation.
    for col in NUMERIC_RAW_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        med = vals.median()
        vals = vals.fillna(med).to_numpy(dtype=np.float32)
        columns.append(vals)
        feature_names.append(col)

    # Derived numeric features (never missing).
    for col in DERIVED_NUMERIC_COLS:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        columns.append(vals)
        feature_names.append(col)

    X = np.column_stack(columns).astype(np.float32)

    # Shuffle with fixed seed, then take the fixed-size splits.
    rng = np.random.RandomState(DATA_SEED)
    total_needed = TRAIN_SIZE + VAL_SIZE + HOLDOUT_SIZE
    if len(X) < total_needed:
        raise RuntimeError(
            f"Not enough labeled rows: have {len(X)}, need {total_needed}."
        )
    perm = rng.permutation(len(X))[:total_needed]
    X = X[perm]
    y = y[perm]

    X_train = X[:TRAIN_SIZE]
    y_train = y[:TRAIN_SIZE]
    X_val = X[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    y_val = y[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    X_holdout = X[TRAIN_SIZE + VAL_SIZE:]
    y_holdout = y[TRAIN_SIZE + VAL_SIZE:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
        "feature_names": feature_names,
    }


def prepare_data():
    """Build, split, and cache the dataset."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)

    print("Building NYC 311 (Jan 2026) dataset from raw CSV...")
    data = _build_from_raw()

    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Data cached to {DATA_FILE}")
    print(f"  Train: {data['X_train'].shape}, base rate: {data['y_train'].mean():.3f}")
    print(f"  Val:   {data['X_val'].shape}, base rate: {data['y_val'].mean():.3f}")
    print(f"  Holdout: {data['X_holdout'].shape}, base rate: {data['y_holdout'].mean():.3f}")
    print(f"  Features ({len(data['feature_names'])}): {data['feature_names']}")

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


def save_model(model):
    """Save the trained model for external evaluation."""
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model.pkl")


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

    if not np.all(np.isin(val_preds, [0, 1])):
        val_preds = (val_preds >= 0.5).astype(int)
    if not np.all(np.isin(holdout_preds, [0, 1])):
        holdout_preds = (holdout_preds >= 0.5).astype(int)

    val_acc = float((val_preds == y_val).mean())
    holdout_acc = float((holdout_preds == y_holdout).mean())

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pred_hash = hashlib.md5(val_preds.tobytes()).hexdigest()[:8]

    with open("holdout_results.tsv", "a") as f:
        f.write(f"{timestamp}\t{pred_hash}\t{val_acc:.6f}\t{holdout_acc:.6f}\n")

    print(f"val_accuracy: {val_acc:.4f}")
    return val_acc


if __name__ == "__main__":
    data = prepare_data()
    print(f"\nFeature names: {data['feature_names']}")
    print(f"Train shape: {data['X_train'].shape}, base rate: {data['y_train'].mean():.3f}")
    print(f"Val shape:   {data['X_val'].shape}, base rate: {data['y_val'].mean():.3f}")
    print(f"Holdout shape: {data['X_holdout'].shape}, base rate: {data['y_holdout'].mean():.3f}")
