# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Baseline training script for gene expression classification.

The auto-research agent modifies this file to improve val_accuracy.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from prepare import load_train, evaluate_val


def train():
    X_train, y_train = load_train()

    # Simple L2-regularized logistic regression as baseline
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    train_acc = (model.predict(X_train) == y_train).mean()
    print(f"train_accuracy: {train_acc:.4f}")

    val_acc = evaluate_val(model)
    return val_acc


if __name__ == "__main__":
    train()
