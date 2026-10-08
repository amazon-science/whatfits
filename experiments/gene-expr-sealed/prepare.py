# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Data interface for the gene-expression experiment (SEALED variant).

READ-ONLY: The auto-research agent must NOT modify this file.

This is a pure DATA LOADER. Unlike the original gene-expr prepare.py, it
contains NO data-generating code: no random seed, no signal-column indices,
no generative weights, and no description of which features carry signal.
It only loads a pre-materialized, cached dataset and exposes the standard
explorer interface. This prevents an agent from reverse-engineering the
label-generating process by reading the preparation code (the true model must
be *learned* from data, not copied from the generator).

Task: binary classification on gene-expression features (high-dimensional,
p >> n). Train on load_train(); optimize val via evaluate_val(model).

- Train: 2,000 samples x 20,000 features
- Val:   20,000 samples (optimization target)
- Holdout: 20,000 samples (secret; measures true generalization)
"""

import os
import datetime
import hashlib
import pickle
import numpy as np
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "autoresearch-geneexpr-sealed"
DATA_FILE = CACHE_DIR / "data_gene_expr_sealed.pkl"


def _load_data():
    """Load the cached, pre-materialized dataset."""
    if not DATA_FILE.exists():
        raise RuntimeError(
            f"Data file not found: {DATA_FILE}\n"
            f"The sealed gene-expression dataset must be staged there first."
        )
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


def load_train():
    """Load training data. Returns (X_train, y_train)."""
    d = _load_data()
    return d["X_train"], d["y_train"]


def load_val():
    """Load validation data. Returns (X_val, y_val)."""
    d = _load_data()
    return d["X_val"], d["y_val"]


def get_feature_names():
    """Return list of feature names."""
    d = _load_data()
    return d["feature_names"]


def evaluate_val(model):
    """Evaluate a model on the val set. Returns val_accuracy (float).

    Also secretly evaluates on holdout and logs to holdout_results.tsv.
    The agent never sees holdout results.
    """
    d = _load_data()
    X_val, y_val = d["X_val"], d["y_val"]
    X_holdout, y_holdout = d["X_holdout"], d["y_holdout"]

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


def save_model(model):
    """Save the trained model for external evaluation."""
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model.pkl")


if __name__ == "__main__":
    d = _load_data()
    print(f"Train: {d['X_train'].shape}, base rate: {d['y_train'].mean():.3f}")
    print(f"Val:   {d['X_val'].shape}, base rate: {d['y_val'].mean():.3f}")
    print(f"Holdout: {d['X_holdout'].shape}, base rate: {d['y_holdout'].mean():.3f}")
