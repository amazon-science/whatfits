# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Reproducer data interface — training data + model saving only.

Self-contained: does NOT import from prepare.py.
- load_train() — training data (full access, call as many times as you want)
- save_model(model) — save your trained model for external evaluation
- get_feature_names() — feature names
- NO evaluate_val, NO load_val — the reproducer never sees validation data
"""

import os
import pickle
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autoresearch-adaptivity"
DATA_FILE = CACHE_DIR / "data_gene_expr.pkl"

N_FEATURES = 20000

_MODEL_PATH = "model.pkl"


# ---------------------------------------------------------------------------
# Internal data loading
# ---------------------------------------------------------------------------

def _load_data():
    """Load cached gene expression dataset."""
    if not DATA_FILE.exists():
        raise RuntimeError(
            f"Data file not found: {DATA_FILE}\n"
            f"Run 'python prepare.py' in the explorer directory first."
        )
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Public interface (training data only)
# ---------------------------------------------------------------------------

def load_train():
    """Load training data. Returns (X_train, y_train).

    X_train: (2000, 20000) float32 — gene expression levels
    y_train: (2000,) float32 — binary labels (0/1)

    You can call this as many times as you want.
    """
    data = _load_data()
    return data["X_train"], data["y_train"]


def get_feature_names():
    """Return list of feature names (gene_00000, gene_00001, ...)."""
    data = _load_data()
    return data["feature_names"]


def save_model(model):
    """Save the trained model for external evaluation.

    The model must have a .predict(X) method that returns class labels (0/1),
    OR be a callable that takes X and returns labels.
    Call this once at the end of training.
    """
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {_MODEL_PATH}")


# ---------------------------------------------------------------------------
# External evaluation (called by pipeline AFTER the agent exits)
# ---------------------------------------------------------------------------

def _evaluate_saved_model():
    """Load the saved model and evaluate on val + holdout."""
    import datetime
    import hashlib

    if not os.path.exists(_MODEL_PATH):
        print("ERROR: No saved model found. The reproducer must call save_model(model).")
        return None, None

    # Build a "train" module holding only the reproducer's class/function
    # definitions so pickled __main__.ClassName references resolve. We AST-filter
    # train.py to defs/classes/imports and never execute its top-level body —
    # a bare `import train` would re-run train.py (including its top-level
    # save_model(...)), truncating model.pkl while we read it.
    import ast
    import sys
    import types

    _train_mod = None
    if os.path.exists("train.py"):
        try:
            with open("train.py") as _tf:
                _src = _tf.read()
            _tree = ast.parse(_src)
            _filtered = [
                n for n in _tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef, ast.Import, ast.ImportFrom))
            ]
            _train_mod = types.ModuleType("train")
            _train_mod.__file__ = os.path.abspath("train.py")
            exec(compile(ast.Module(body=_filtered, type_ignores=[]),
                         "train.py", "exec"), _train_mod.__dict__)
            sys.modules["train"] = _train_mod
        except Exception:
            _train_mod = None

    class _TrainAwareUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and _train_mod is not None:
                try:
                    return getattr(_train_mod, name)
                except AttributeError:
                    pass
            return super().find_class(module, name)

    with open(_MODEL_PATH, "rb") as f:
        model = _TrainAwareUnpickler(f).load()

    data = _load_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_holdout, y_holdout = data["X_holdout"], data["y_holdout"]

    # Get predictions
    if callable(getattr(model, "predict", None)):
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        holdout_preds = model.predict(X_holdout)
    elif callable(model):
        train_preds = model(X_train)
        val_preds = model(X_val)
        holdout_preds = model(X_holdout)
    else:
        raise ValueError("model must have .predict() or be callable")

    train_preds = np.asarray(train_preds).flatten()
    val_preds = np.asarray(val_preds).flatten()
    holdout_preds = np.asarray(holdout_preds).flatten()

    if not np.all(np.isin(train_preds, [0, 1])):
        train_preds = (train_preds >= 0.5).astype(int)
    if not np.all(np.isin(val_preds, [0, 1])):
        val_preds = (val_preds >= 0.5).astype(int)
    if not np.all(np.isin(holdout_preds, [0, 1])):
        holdout_preds = (holdout_preds >= 0.5).astype(int)

    train_acc = float((train_preds == y_train).mean())
    val_acc = float((val_preds == y_val).mean())
    holdout_acc = float((holdout_preds == y_holdout).mean())

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pred_hash = hashlib.md5(val_preds.tobytes()).hexdigest()[:8]

    with open("holdout_results.tsv", "a") as f:
        f.write(f"{timestamp}\t{val_acc:.6f}\t{holdout_acc:.6f}\t{train_acc:.6f}\n")

    print(f"train_accuracy: {train_acc:.4f}")
    print(f"val_accuracy: {val_acc:.4f}")
    print(f"holdout_accuracy: {holdout_acc:.4f}")
    return val_acc, holdout_acc


if __name__ == "__main__":
    _evaluate_saved_model()
