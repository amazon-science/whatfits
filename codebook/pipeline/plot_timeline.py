#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Codebook MDL timeline plot.

For a given experiment directory, visualizes:
- Explorer val + holdout trajectories
- Codebook-compressed checkpoints with reproducer val, holdout, and MDL CI
- Compression gap gradient coloring (green → gray → red)

Usage:
    python codebook/pipeline/plot_timeline.py codebook/experiments/folktables-5k--full-aggressive/
    python codebook/pipeline/plot_timeline.py codebook/experiments/folktables-5k--full-analyst/
"""

import argparse
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
CODEBOOK_ROOT = PIPELINE_DIR.parent
PROJECT_ROOT = CODEBOOK_ROOT.parent
sys.path.insert(0, str(CODEBOOK_ROOT))
from schema import mdl_bound

# Colors
C_EXPLORER = "#B0B0B0"
C_HOLDOUT_EXPL = "#9B59B6"
C_BEST = "#2C5F8A"
C_PASS = "#2ECC71"
C_FAIL = "#E74C3C"
C_REPR_HOLD = "#E67E22"
C_CI_BAND = "#4A90D9"

GAP_CMAP = LinearSegmentedColormap.from_list("gap",
    [(0.0, C_PASS), (0.3, "#AAAAAA"), (1.0, C_FAIL)])


def parse_results_tsv(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    rows.append((parts[0].strip(), float(parts[1]), parts[2].strip()))
                except ValueError:
                    continue  # skip non-numeric metrics
    return rows


def parse_holdout_tsv(path, results_rows):
    holdout_map = {}
    if not os.path.exists(path):
        return holdout_map
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                entries.append((float(parts[2]), float(parts[3])))
    used = set()
    for commit, metric, _ in results_rows:
        for j, (val, holdout) in enumerate(entries):
            if j in used:
                continue
            if abs(val - metric) < 1e-5:
                holdout_map[commit[:7]] = holdout
                used.add(j)
                break
    return holdout_map


def load_checkpoint_data(exp_dir: Path, n_val: int, delta: float,
                         bit_budget: float) -> dict:
    """Load compress + reproduce results for all checkpoints."""
    data = {}
    ckpts_dir = exp_dir / "checkpoints"
    if not ckpts_dir.exists():
        return data
    for ckpt_dir in sorted(ckpts_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        ckpt = ckpt_dir.name
        compress_result = ckpt_dir / "compress" / "codebook_result.json"
        reproduce_holdout = ckpt_dir / "reproduce" / "holdout_results.tsv"
        if not compress_result.exists() or not reproduce_holdout.exists():
            continue
        with open(compress_result) as f:
            cr = json.load(f)
        bits = cr.get("codelength_bits", 0)
        with open(reproduce_holdout) as f:
            lines = f.readlines()
        if not lines:
            continue
        last = lines[-1].strip().split("\t")
        repr_val = float(last[0]) if len(last) > 0 else None
        repr_hold = float(last[1]) if len(last) > 1 else None
        if repr_val is None:
            continue
        # Use bit_budget (not actual bits) for the bound — the budget is
        # the constructive description length the reproducer received
        data[ckpt] = {
            "val": repr_val,
            "holdout": repr_hold,
            "bits": bits,
            "bound": mdl_bound(bit_budget, n_val, delta),
        }
    return data


def plot(exp_dir: Path, output_path: Path = None, max_iter: int = None):
    """Generate the timeline plot for one experiment."""
    # Load config
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return
    with open(config_path) as f:
        config = json.load(f)

    experiment = config["experiment"]
    condition = config["condition"]
    n_val = config["n_val"]
    delta = config.get("delta", 0.05)
    beta = config.get("beta", 20)
    bit_budget = config.get("bit_budget", 16)

    # Load explorer data
    explore_dir = PROJECT_ROOT / config["explorer_dir"]
    results = parse_results_tsv(explore_dir / "results.tsv")
    holdout_map = parse_holdout_tsv(explore_dir / "holdout_results.tsv", results)

    # Truncate if requested
    if max_iter is not None:
        results = results[:max_iter]

    iters = list(range(1, len(results) + 1))
    val_metrics = [r[1] for r in results]
    hashes = [r[0][:7] for r in results]
    holdouts = [holdout_map.get(h) for h in hashes]

    # Best-so-far
    best_so_far = []
    cur_best = None
    for m in val_metrics:
        if cur_best is None or m > cur_best:
            cur_best = m
        best_so_far.append(cur_best)

    # Load checkpoint reproduce data
    reproduce_data = load_checkpoint_data(exp_dir, n_val, delta, bit_budget)

    # Phase 0 baseline
    phase0 = config.get("phase0_baseline")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # Phase 0 baseline horizontal line + union bound CI band
    if phase0:
        p0_val = phase0["val_accuracy"]
        p0_holdout = phase0.get("holdout_accuracy")
        n_configs = phase0.get("n_configs", 12)
        # Union bound: ε = sqrt(ln(2N/δ) / (2n))
        import math
        union_bound = math.sqrt(math.log(2 * n_configs / delta) / (2 * n_val))

        ax.axhline(y=p0_val, color="#2C5F8A", linewidth=1.5, linestyle="-.",
                   alpha=0.6, zorder=1)
        ax.fill_between([0, len(results) + 1],
                        [p0_val - union_bound] * 2, [p0_val + union_bound] * 2,
                        color="#2C5F8A", alpha=0.08, zorder=0)
        # Label
        ax.text(len(results) + 0.3, p0_val, f"Phase 0 baseline\n{p0_val:.3f} ± {union_bound*100:.1f}pp",
                fontsize=7, color="#2C5F8A", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#2C5F8A", alpha=0.8))

    # Explorer val trajectory
    ax.plot(iters, val_metrics, color=C_EXPLORER, linewidth=1.0, alpha=0.6, zorder=1)
    ax.scatter(iters, val_metrics, s=15, color=C_EXPLORER, alpha=0.4,
               zorder=2, edgecolors="none")

    # Explorer holdout trajectory
    holdout_iters = [it for it, h in zip(iters, holdouts) if h is not None]
    holdout_vals = [h for h in holdouts if h is not None]
    if holdout_vals:
        ax.plot(holdout_iters, holdout_vals, color=C_HOLDOUT_EXPL, linewidth=1.2,
                alpha=0.5, zorder=1)
        ax.scatter(holdout_iters, holdout_vals, s=12, color=C_HOLDOUT_EXPL,
                   alpha=0.3, zorder=2, edgecolors="none")

    # Best-so-far envelope
    ax.step(iters, best_so_far, where="post", color=C_BEST, linewidth=1.5,
            alpha=0.3, linestyle="--", zorder=2)

    # Codebook checkpoints
    for h_idx, (h, it) in enumerate(zip(hashes, iters)):
        if h not in reproduce_data:
            continue

        rd = reproduce_data[h]
        repr_val = rd["val"]
        repr_hold = rd["holdout"]
        bits = rd["bits"]
        bound = rd["bound"]
        expl_val = val_metrics[h_idx]

        # Compression gap → gradient color
        gap = expl_val - repr_val
        gap_norm = np.clip(gap / 0.10, 0.0, 1.0)
        dot_color = GAP_CMAP(gap_norm)

        # CI band
        ci_lo = repr_val - bound
        ci_hi = repr_val + bound
        holdout_in_ci = ci_lo <= repr_hold <= ci_hi if repr_hold else False

        ax.fill_between([it - 0.3, it + 0.3], [ci_lo, ci_lo], [ci_hi, ci_hi],
                        color=C_CI_BAND, alpha=0.15, zorder=3)
        ax.plot([it - 0.3, it + 0.3, it + 0.3, it - 0.3, it - 0.3],
                [ci_lo, ci_lo, ci_hi, ci_hi, ci_lo],
                color=C_CI_BAND, linewidth=0.8, alpha=0.4, zorder=3)

        # Red ring if holdout outside CI
        if repr_hold is not None and not holdout_in_ci:
            ax.scatter([it], [repr_hold], s=120, facecolors="none",
                       edgecolors=C_FAIL, linewidths=2.0, zorder=9)

        # Reproducer val (triangle, gap-colored)
        ax.scatter([it], [repr_val], s=80, color=[dot_color], marker="v",
                   zorder=7, edgecolors="white", linewidths=0.8)

        # Reproducer holdout (open diamond)
        if repr_hold is not None:
            ax.scatter([it], [repr_hold], s=70, color=C_REPR_HOLD, marker="D",
                       zorder=7, edgecolors=C_REPR_HOLD, facecolors="none",
                       linewidths=1.5)

        # Line from explorer to reproducer (gap-colored)
        ax.plot([it, it], [expl_val, repr_val], color=dot_color, linewidth=2,
                alpha=0.6, zorder=4)

        # Explorer dot (gap-colored)
        ax.scatter([it], [expl_val], s=80, color=[dot_color], zorder=6,
                   edgecolors="white", linewidths=1.0)

        # Annotation
        ax.annotate(f"{bit_budget:.0f}b ±{bound*100:.1f}pp",
                    (it, ci_hi), textcoords="offset points",
                    xytext=(8, 6), fontsize=6.5, color="#555555",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#DDDDDD", alpha=0.8))

    # Title
    title = f"{experiment} ({condition}) — Codebook MDL Bounds"
    subtitle = f"bit budget={bit_budget}, n_val={n_val:,}, δ={delta}, β={beta}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_xlim(0.5, len(results) + 4)

    # Y-axis padding
    all_y = val_metrics + [v for v in holdout_vals if v]
    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.1)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#2C5F8A", linewidth=1.5, linestyle="-.", alpha=0.6,
               label="Phase 0 baseline (train-only)"),
        Line2D([0], [0], color=C_EXPLORER, linewidth=1.0, alpha=0.6,
               label="Explorer val"),
        Line2D([0], [0], color=C_HOLDOUT_EXPL, linewidth=1.2, alpha=0.5,
               label="Explorer holdout"),
        Line2D([0], [0], color=C_BEST, linewidth=1.5, linestyle="--", alpha=0.3,
               label="Best-so-far"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#AAAAAA",
               markersize=8, label="Reproducer val"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
               markeredgecolor=C_REPR_HOLD, markersize=7, markeredgewidth=1.5,
               label="Reproducer holdout"),
        mpatches.Patch(facecolor=C_CI_BAND, alpha=0.2,
                       label=f"MDL {int((1-delta)*100)}% CI"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=7.5,
               frameon=True, fancybox=True, ncol=6,
               bbox_to_anchor=(0.42, -0.02))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=GAP_CMAP, norm=plt.Normalize(0, 10))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.82, 0.01, 0.12, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Compression gap (pp)", fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    # Save
    if output_path is None:
        out_dir = PROJECT_ROOT / "plots" / "codebook"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{experiment}_{condition}_codebook_mdl.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Codebook MDL timeline plot")
    parser.add_argument("experiment_dir", help="Path to experiment dir (e.g., codebook/experiments/folktables-5k--full-aggressive/)")
    parser.add_argument("--output", type=str, default=None, help="Output path for plot")
    parser.add_argument("--max-iter", type=int, default=None, help="Truncate plot at this iteration")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    output_path = Path(args.output) if args.output else None
    plot(exp_dir, output_path, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
