# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Ladder mechanism: binary feedback wrapper for evaluate_val.

Instead of returning the actual val accuracy, returns only whether the
submitted model improves on the best-so-far. This leaks at most 1 bit
per query. After T queries with K improvements, the total information
leaked is K * log2(T) bits (encoding which K of T queries improved).

This file wraps an existing experiment's prepare.py. To use:

    from prepare_ladder import load_train, evaluate_val, save_model, get_feature_names

evaluate_val(model) prints "IMPROVED (K/T)" or "NO IMPROVEMENT (K/T)"
and returns True/False. The actual val/holdout scores are logged secretly.

State is persisted in ladder_state.json between calls.
"""

import json
import os
import sys
import datetime
import hashlib
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

STATE_FILE = "ladder_state.json"


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"best_val": None, "n_queries": 0, "n_improvements": 0, "history": []}


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Delegation to the real prepare.py
# ---------------------------------------------------------------------------

# The real prepare.py is expected to be in the same directory
# (copied from the experiment, with evaluate_val renamed to _real_evaluate_val)
# OR we import it and wrap.

_real_module = None


def _get_real_module():
    global _real_module
    if _real_module is not None:
        return _real_module

    # Import the real prepare module (named _prepare_real.py in this dir)
    import importlib.util
    real_path = os.path.join(os.path.dirname(__file__), "_prepare_real.py")
    if not os.path.exists(real_path):
        # Fallback: try prepare_original.py
        real_path = os.path.join(os.path.dirname(__file__), "prepare_original.py")
    if not os.path.exists(real_path):
        raise FileNotFoundError(
            "Cannot find _prepare_real.py or prepare_original.py. "
            "Copy the experiment's prepare.py and rename it."
        )
    spec = importlib.util.spec_from_file_location("_prepare_real", real_path)
    _real_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_real_module)
    return _real_module


# ---------------------------------------------------------------------------
# Public API (matching the standard prepare.py interface)
# ---------------------------------------------------------------------------

def load_train(*args, **kwargs):
    """Load training data. Delegates to real prepare.py."""
    return _get_real_module().load_train(*args, **kwargs)


def get_feature_names(*args, **kwargs):
    """Return list of feature names."""
    return _get_real_module().get_feature_names(*args, **kwargs)


def save_model(*args, **kwargs):
    """Save the trained model for external evaluation."""
    return _get_real_module().save_model(*args, **kwargs)


def __getattr__(name):
    """Pass through any other function to the real module.

    This handles GPU experiments that have additional functions like
    make_train_loader, make_val_loader, get_tokenizer, etc.
    We intercept any evaluate_* function (evaluate_val, evaluate_bpb, etc.)
    to enforce binary feedback.
    """
    if name.startswith("evaluate_"):
        return _make_ladder_wrapper(name)
    real = _get_real_module()
    if hasattr(real, name):
        return getattr(real, name)
    raise AttributeError(f"module 'prepare' has no attribute '{name}'")


def _ladder_eval(real_fn_name, model, **kwargs):
    """Evaluate model on val with BINARY FEEDBACK only.

    Returns True if the model improves on the best-so-far, False otherwise.
    Prints "IMPROVED (K/T)" or "NO IMPROVEMENT (K/T)" where:
        K = number of improvements so far
        T = total number of queries so far

    The actual val metric is NEVER shown to the agent.

    This works for both tabular (sklearn) and GPU (PyTorch) experiments by
    delegating to the real eval function and intercepting its output.
    """
    import io
    import sys

    real = _get_real_module()
    real_fn = getattr(real, real_fn_name)

    # Capture the real function's stdout (which prints the val metric)
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        result = real_fn(model, **kwargs)
    finally:
        sys.stdout = old_stdout

    # Parse the val metric from the captured output
    captured_text = captured.getvalue()
    val_acc = None
    if result is not None:
        val_acc = float(result)
    else:
        # Try to parse from printed output
        for line in captured_text.split("\n"):
            if "accuracy" in line.lower() or "bpb" in line.lower() or "logloss" in line.lower():
                import re
                nums = re.findall(r"[\d.]+", line)
                if nums:
                    val_acc = float(nums[-1])
                    break

    if val_acc is None:
        print(f"WARNING: Could not determine val metric from {real_fn_name}")
        print(f"NO IMPROVEMENT (?/?)")
        return False

    # Determine direction from metric name
    # Lower is better for BPB, logloss; higher is better for accuracy
    metric_direction = "higher"  # default
    lower_metrics = ["bpb", "logloss", "loss", "perplexity"]
    for lm in lower_metrics:
        if lm in real_fn_name.lower() or lm in captured_text.lower():
            metric_direction = "lower"
            break

    # Load state
    state = _load_state()
    state["n_queries"] += 1
    T = state["n_queries"]

    # Binary feedback: improved?
    improved = False
    if state["best_val"] is None:
        improved = True
    elif metric_direction == "higher" and val_acc > state["best_val"]:
        improved = True
    elif metric_direction == "lower" and val_acc < state["best_val"]:
        improved = True

    if improved:
        state["best_val"] = val_acc
        state["n_improvements"] += 1

    K = state["n_improvements"]

    # Log to history (secret)
    state["history"].append({
        "query": T,
        "improved": improved,
        "val_acc": val_acc,
    })
    _save_state(state)

    # Print ONLY binary feedback
    if improved:
        print(f"IMPROVED ({K}/{T})")
    else:
        print(f"NO IMPROVEMENT ({K}/{T})")

    return improved


def _make_ladder_wrapper(fn_name):
    """Create a ladder-wrapped version of any evaluate_* function."""
    def wrapper(model, **kwargs):
        return _ladder_eval(fn_name, model, **kwargs)
    wrapper.__name__ = fn_name
    wrapper.__doc__ = f"Ladder-wrapped {fn_name}: returns binary feedback only."
    return wrapper


def evaluate_val(model, **kwargs):
    """Evaluate model on val with BINARY FEEDBACK only."""
    return _ladder_eval("evaluate_val", model, **kwargs)


# ---------------------------------------------------------------------------
# Bound computation
# ---------------------------------------------------------------------------

def compute_ladder_bound(n_val: int, T: int, K: int, delta: float = 0.05) -> float:
    """Compute the MDL generalization bound from the ladder mechanism.

    The trajectory (which K of T queries improved) can be encoded in
    K * log2(T) bits. This plugs into the MDL bound:

        |val - true| <= sqrt((k * ln(2) + ln(1/delta)) / (2 * n))

    where k = K * log2(T).

    Args:
        n_val: Validation set size.
        T: Total number of queries.
        K: Number of improvements.
        delta: Confidence parameter.

    Returns:
        Bound on |val_accuracy - true_accuracy|.
    """
    import math
    if T <= 0 or K <= 0:
        return 0.0
    k_bits = K * math.log2(T)
    return math.sqrt((k_bits * math.log(2) + math.log(1.0 / delta)) / (2 * n_val))


def print_summary():
    """Print the ladder state summary with bound computation."""
    import math
    state = _load_state()
    T = state["n_queries"]
    K = state["n_improvements"]
    best_val = state["best_val"]

    print(f"\nLadder Summary:")
    print(f"  Total queries (T): {T}")
    print(f"  Improvements (K):  {K}")
    print(f"  Best val accuracy: {best_val:.4f}" if best_val else "  Best val: (none)")
    if K > 0 and T > 0:
        k_bits = K * math.log2(T)
        print(f"  Description length: K * log2(T) = {K} * {math.log2(T):.2f} = {k_bits:.1f} bits")


if __name__ == "__main__":
    print_summary()
