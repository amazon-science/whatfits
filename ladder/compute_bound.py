#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Compute ladder mechanism generalization bounds.

After T queries with K improvements, the explorer's trajectory can be
encoded in K * log2(T) bits. The MDL bound gives:

    |val - true| <= sqrt((K * log2(T) * ln(2) + ln(1/delta)) / (2 * n_val))

Usage:
    python ladder/compute_bound.py folktables-10k ladder-analyst
    python ladder/compute_bound.py folktables-10k ladder-analyst --plot
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_ladder_state(explorer_dir: Path) -> dict:
    state_path = explorer_dir / "ladder_state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return None


def load_holdout_data(explorer_dir: Path) -> list[dict]:
    holdout_path = explorer_dir / "holdout_results.tsv"
    entries = []
    if holdout_path.exists():
        with open(holdout_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    entries.append({
                        "timestamp": parts[0],
                        "hash": parts[1],
                        "val": float(parts[2]),
                        "holdout": float(parts[3]),
                    })
    return entries


def ladder_bound(K: int, T: int, n_val: int, delta: float = 0.05) -> float:
    """MDL bound from K improvements in T queries."""
    if K <= 0 or T <= 1:
        return 0.0
    k_bits = K * math.log2(T)
    return math.sqrt((k_bits * math.log(2) + math.log(1.0 / delta)) / (2 * n_val))


def main():
    parser = argparse.ArgumentParser(description="Compute ladder bounds")
    parser.add_argument("experiment", help="Experiment directory")
    parser.add_argument("condition", help="Condition (e.g., ladder-analyst)")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.experiment
    explorer_dir = dataset_dir / f"explore-agent-{args.condition}"

    if not explorer_dir.exists():
        print(f"ERROR: {explorer_dir} not found")
        sys.exit(1)

    # Detect n_val
    n_val = 200
    prep = dataset_dir / "prepare.py"
    if prep.exists():
        with open(prep) as f:
            for line in f:
                if "VAL_SIZE" in line and "=" in line:
                    try:
                        n_val = int(line.split("=")[1].strip())
                    except ValueError:
                        pass

    # Load state
    state = load_ladder_state(explorer_dir)
    holdout = load_holdout_data(explorer_dir)

    if state:
        T = state["n_queries"]
        K = state["n_improvements"]
        best_val = state["best_val"]
    else:
        # Infer from holdout data
        T = len(holdout)
        K = 0
        best_val = 0
        for h in holdout:
            if h["val"] > best_val:
                K += 1
                best_val = h["val"]

    k_bits = K * math.log2(T) if T > 1 else 0
    bound = ladder_bound(K, T, n_val, args.delta)

    print(f"{'='*60}")
    print(f"Ladder Mechanism Bound")
    print(f"{'='*60}")
    print(f"  Experiment:    {args.experiment} ({args.condition})")
    print(f"  n_val:         {n_val}")
    print(f"  Total queries: T = {T}")
    print(f"  Improvements:  K = {K}")
    print(f"  Codelength:    K * log2(T) = {K} * {math.log2(T):.2f} = {k_bits:.1f} bits")
    print(f"  MDL bound:     ±{bound:.4f} ({bound*100:.1f}pp)")
    print(f"  Best val:      {best_val:.4f}" if best_val else "  Best val: (none)")

    if holdout:
        best_holdout = max(holdout, key=lambda h: h["val"])
        print(f"  Holdout at best val: {best_holdout['holdout']:.4f}")
        print(f"  Val-holdout gap:     {(best_holdout['val'] - best_holdout['holdout'])*100:.1f}pp")

        # Check if holdout falls within bound
        ci_lo = best_val - bound
        ci_hi = best_val + bound
        in_ci = ci_lo <= best_holdout["holdout"] <= ci_hi
        print(f"  CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Holdout in CI: {'YES' if in_ci else 'NO'}")

    # Theoretical projections
    print(f"\n--- Bound at different T (n_val={n_val}, K={K}) ---")
    for t in [10, 20, 50, 100, 200, 500]:
        b = ladder_bound(K, t, n_val, args.delta)
        bits = K * math.log2(t) if t > 1 else 0
        print(f"  T={t:>4d}: {bits:>6.1f} bits → ±{b:.4f} ({b*100:.1f}pp)")

    # Comparison with score-only (each query leaks full val accuracy ≈ 10 bits)
    print(f"\n--- Comparison ---")
    score_only_bits = T * 10  # ~10 bits per float val accuracy
    score_only_bound = math.sqrt((score_only_bits * math.log(2) + math.log(1.0 / args.delta)) / (2 * n_val))
    print(f"  Ladder:     {k_bits:.1f} bits → ±{bound:.4f}")
    print(f"  Score-only: ~{score_only_bits} bits → ±{score_only_bound:.4f}")
    print(f"  Savings:    {score_only_bits - k_bits:.0f} bits ({(1 - k_bits/max(score_only_bits,1))*100:.0f}% reduction)")

    if args.plot:
        _make_plot(holdout, n_val, K, args.delta, args.experiment, args.condition)


def _make_plot(holdout, n_val, K_final, delta, experiment, condition):
    """Plot the ladder trajectory with running bounds."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not holdout:
        print("No holdout data for plot")
        return

    iters = list(range(1, len(holdout) + 1))
    vals = [h["val"] for h in holdout]
    holds = [h["holdout"] for h in holdout]

    # Compute running K and bounds
    best_so_far = 0
    K_running = []
    bounds_running = []
    for i, v in enumerate(vals):
        if v > best_so_far:
            best_so_far = v
            K_running.append(K_running[-1] + 1 if K_running else 1)
        else:
            K_running.append(K_running[-1] if K_running else 0)
        T = i + 1
        K = K_running[-1]
        bounds_running.append(ladder_bound(K, T, n_val, delta))

    # Best-so-far envelope
    best_env = []
    cur_best = 0
    for v in vals:
        cur_best = max(cur_best, v)
        best_env.append(cur_best)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Val trajectory
    ax.plot(iters, vals, color="#B0B0B0", linewidth=0.8, alpha=0.5)
    ax.scatter(iters, vals, s=12, color="#B0B0B0", alpha=0.4, edgecolors="none")

    # Holdout trajectory
    ax.plot(iters, holds, color="#9B59B6", linewidth=1.0, alpha=0.5)

    # Best-so-far
    ax.step(iters, best_env, where="post", color="#2C5F8A", linewidth=1.5,
            alpha=0.4, linestyle="--")

    # Running CI band around best-so-far
    ci_upper = [b + bnd for b, bnd in zip(best_env, bounds_running)]
    ci_lower = [b - bnd for b, bnd in zip(best_env, bounds_running)]
    ax.fill_between(iters, ci_lower, ci_upper, color="#4A90D9", alpha=0.1)

    # Mark improvements
    imp_iters = [i+1 for i in range(len(vals)) if (i == 0 and vals[i] > 0) or (i > 0 and vals[i] > max(vals[:i]))]
    imp_vals = [vals[i-1] for i in imp_iters]
    ax.scatter(imp_iters, imp_vals, s=60, color="#2ECC71", zorder=5,
               edgecolors="white", linewidths=0.8)

    K = K_running[-1] if K_running else 0
    T = len(holdout)
    k_bits = K * math.log2(T) if T > 1 else 0
    bound = bounds_running[-1] if bounds_running else 0

    ax.set_title(f"{experiment} ({condition}) — Ladder Mechanism\n"
                 f"T={T} queries, K={K} improvements, {k_bits:.1f} bits, "
                 f"bound ±{bound*100:.1f}pp",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Query", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.grid(True, alpha=0.15)

    legend_elements = [
        Line2D([0], [0], color="#B0B0B0", linewidth=0.8, label="Val (secret)"),
        Line2D([0], [0], color="#9B59B6", linewidth=1.0, label="Holdout (secret)"),
        Line2D([0], [0], color="#2C5F8A", linewidth=1.5, linestyle="--", label="Best-so-far"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ECC71",
               markersize=8, label=f"Improvement (K={K})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    out_dir = PROJECT_ROOT / "plots" / "ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment}_{condition}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
