#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Materialize the sealed gene-expression dataset cache.

The sealed experiment's prepare.py is intentionally only a loader. This staging
helper lives outside the files copied into agent work directories and writes the
same fixed synthetic split to:
    ~/.cache/autoresearch-geneexpr-sealed/data_gene_expr_sealed.pkl
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np


CACHE_DIR = Path.home() / ".cache" / "autoresearch-geneexpr-sealed"
DATA_FILE = CACHE_DIR / "data_gene_expr_sealed.pkl"
GENERATOR_PATH = Path(__file__).resolve().parents[1] / "gene-expr" / "prepare.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("gene_expr_prepare", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator module: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_data() -> dict:
    gen = load_generator()
    X, y, clean_labels = gen._generate_data()

    rng = np.random.RandomState(gen.DATA_SEED + 1)
    indices = rng.permutation(gen.TOTAL)
    X = X[indices]
    y = y[indices]
    clean_labels = clean_labels[indices]

    X_train = X[: gen.TRAIN_SIZE]
    y_train = y[: gen.TRAIN_SIZE]
    X_val = X[gen.TRAIN_SIZE : gen.TRAIN_SIZE + gen.VAL_SIZE]
    y_val = y[gen.TRAIN_SIZE : gen.TRAIN_SIZE + gen.VAL_SIZE]
    X_holdout = X[gen.TRAIN_SIZE + gen.VAL_SIZE :]
    y_holdout = y[gen.TRAIN_SIZE + gen.VAL_SIZE :]

    feature_rng = np.random.RandomState(gen.DATA_SEED + 2)
    feature_perm = feature_rng.permutation(gen.N_FEATURES)
    X_train = X_train[:, feature_perm]
    X_val = X_val[:, feature_perm]
    X_holdout = X_holdout[:, feature_perm]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
        "feature_names": [f"gene_{i:05d}" for i in range(gen.N_FEATURES)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the sealed gene-expression cache")
    parser.add_argument("--output", type=Path, default=DATA_FILE)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cache")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Already staged: {args.output}")
        print("Use --force to overwrite.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = build_data()
    with args.output.open("wb") as f:
        pickle.dump(data, f)

    print(f"Data cached to {args.output}")
    print(f"  Train: {data['X_train'].shape}, base rate: {data['y_train'].mean():.3f}")
    print(f"  Val:   {data['X_val'].shape}, base rate: {data['y_val'].mean():.3f}")
    print(f"  Holdout: {data['X_holdout'].shape}, base rate: {data['y_holdout'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
