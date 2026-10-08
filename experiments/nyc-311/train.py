# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Baseline: Logistic regression on NYC 311 (Jan 2026) 72h-closure prediction.

The auto-research agent modifies this file to improve val_accuracy.
"""

from prepare import load_train, evaluate_val, save_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X_train, y_train = load_train()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)

evaluate_val(model)
save_model(model)
