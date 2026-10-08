#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Build a performance-weighted codebook from exploration configs.

Reads configs.jsonl (from Phase 0 sequential exploration) and computes a
spike-and-slab codebook where prior probabilities are weighted by CV performance.

Each config gets Boltzmann weight: w_j = exp(beta * cv_accuracy_j).
Then:
  - P(menu_i = value_v) ∝ Σ_{configs where menu_i = v} w_j
  - π_i (selection prob) = Σ_{configs specifying menu_i} w_j / Σ_all w_j

At beta=0, this recovers frequency-based priors. At high beta, the prior
concentrates on whatever the best-performing configs used.

Usage:
    python codebook/build_codebook_from_configs.py codebook/build-folktables/configs.jsonl \
        --beta 10 20 50 --n_val 200 --output codebook/folktables-perf.json

    # Compare codelengths under different betas:
    python codebook/build_codebook_from_configs.py codebook/build-folktables/configs.jsonl \
        --beta 0 10 50 --compare-strategies codebook/strategies/folktables-full-analyst-*.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema import compute_sparse_codelength, mdl_bound


def load_configs(path: str) -> list[dict]:
    configs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                configs.append(json.loads(line))
    return configs


def build_codebook(configs: list[dict], beta: float, name: str = "codebook",
                   exclude_menus: set[str] | None = None) -> dict:
    """Build a performance-weighted spike-and-slab codebook.

    Args:
        configs: List of dicts with 'cv_accuracy' and 'specified_menus'.
        beta: Boltzmann temperature. 0 = frequency-based, higher = more peaked.
        name: Codebook name.

    Returns:
        Codebook dict in the standard JSON format.
    """
    # Compute Boltzmann weights
    accuracies = [c["cv_accuracy"] for c in configs]
    weights = [math.exp(beta * a) for a in accuracies]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    # Collect all menu names and their values
    menu_values = defaultdict(lambda: defaultdict(float))  # menu -> value -> weight
    menu_specified_weight = defaultdict(float)  # menu -> total weight when specified

    if exclude_menus is None:
        exclude_menus = set()

    for config, w in zip(configs, norm_weights):
        menus = config["specified_menus"]
        for menu_name, value in menus.items():
            if menu_name in exclude_menus:
                continue
            key = json.dumps(value) if not isinstance(value, str) else value
            menu_values[menu_name][key] += w
            menu_specified_weight[menu_name] += w

    # Build menus
    codebook_menus = []
    for menu_name in sorted(menu_values.keys()):
        values = menu_values[menu_name]
        pi = menu_specified_weight[menu_name]  # already normalized

        # Determine menu type from observed values
        all_keys = list(values.keys())
        sample_values = []
        for k in all_keys:
            try:
                sample_values.append(json.loads(k))
            except (json.JSONDecodeError, TypeError):
                sample_values.append(k)

        # Classify: all strings → categorical, all numeric → continuous/integer
        if all(isinstance(v, str) for v in sample_values):
            # Categorical
            total_val_weight = sum(values.values())
            options = {}
            for k, w in values.items():
                options[k] = round(w / total_val_weight, 4)
            # Ensure probabilities sum to 1
            remainder = 1.0 - sum(options.values())
            if options:
                first_key = next(iter(options))
                options[first_key] = round(options[first_key] + remainder, 4)

            codebook_menus.append({
                "name": menu_name,
                "type": "categorical",
                "selection_probability": round(pi, 4),
                "options": options,
            })
        elif all(isinstance(v, int) for v in sample_values):
            # Integer
            int_vals = [int(v) for v in sample_values]
            codebook_menus.append({
                "name": menu_name,
                "type": "integer",
                "selection_probability": round(pi, 4),
                "min": min(int_vals),
                "max": max(int_vals),
            })
        elif all(isinstance(v, (int, float)) for v in sample_values):
            # Continuous
            float_vals = [float(v) for v in sample_values]
            lo, hi = min(float_vals), max(float_vals)
            # Expand range slightly if all same value
            if lo == hi:
                lo = lo * 0.5 if lo > 0 else lo - 0.1
                hi = hi * 1.5 if hi > 0 else hi + 0.1
            scale = "log" if lo > 0 and hi / lo > 10 else "linear"
            codebook_menus.append({
                "name": menu_name,
                "type": "continuous",
                "selection_probability": round(pi, 4),
                "min": lo,
                "max": hi,
                "scale": scale,
            })
        else:
            # Fallback: treat as categorical with string keys
            total_val_weight = sum(values.values())
            options = {k: round(w / total_val_weight, 4) for k, w in values.items()}
            codebook_menus.append({
                "name": menu_name,
                "type": "categorical",
                "selection_probability": round(pi, 4),
                "options": options,
            })

    # Sort menus by selection probability (descending)
    codebook_menus.sort(key=lambda m: -m["selection_probability"])

    return {
        "name": name,
        "description": f"Performance-weighted spike-and-slab prior (beta={beta})",
        "menus": codebook_menus,
        "escape_penalty_bits": 12.0,
        "beta": beta,
        "n_configs_explored": len(configs),
        "best_cv_accuracy": max(accuracies),
        "weight_summary": {
            "beta": beta,
            "top_config_weight": round(max(norm_weights), 4),
            "bottom_config_weight": round(min(norm_weights), 4),
            "effective_sample_size": round(1.0 / sum(w**2 for w in norm_weights), 1),
        },
    }


def print_codebook_summary(codebook: dict):
    """Print a human-readable summary."""
    ws = codebook.get("weight_summary", {})
    print(f"\n  Codebook: {codebook['name']}")
    print(f"  Beta: {codebook.get('beta', '?')}")
    print(f"  Configs: {codebook['n_configs_explored']}, "
          f"best CV: {codebook['best_cv_accuracy']:.4f}")
    if ws:
        print(f"  Weight range: [{ws['bottom_config_weight']:.4f}, "
              f"{ws['top_config_weight']:.4f}], "
              f"ESS: {ws['effective_sample_size']:.1f}/{codebook['n_configs_explored']}")

    print(f"\n  {'Menu':<25s} {'Type':<12s} {'π':>6s}  Details")
    print(f"  {'-'*70}")
    for m in codebook["menus"]:
        detail = ""
        if m["type"] == "categorical":
            top = sorted(m["options"].items(), key=lambda x: -x[1])[:3]
            detail = ", ".join(f"{k}={v:.2f}" for k, v in top)
        elif m["type"] in ("integer", "continuous"):
            detail = f"[{m['min']}, {m['max']}]"
        print(f"  {m['name']:<25s} {m['type']:<12s} {m['selection_probability']:>6.3f}  {detail}")


def encode_strategy_for_codebook(strategy: dict, codebook: dict) -> dict:
    """Map a strategy JSON to codebook menu selections.

    The strategy JSON has flat key-value pairs. We map each to the
    corresponding codebook menu, skipping null/None values.
    """
    menu_names = {m["name"] for m in codebook["menus"]}
    specified = {}
    for key, value in strategy.items():
        if key in menu_names and value is not None:
            # Skip empty lists, "none" strings, default-looking values
            if isinstance(value, list) and len(value) == 0:
                continue
            if isinstance(value, str) and value.lower() == "none":
                continue
            specified[key] = value
    return specified


def main():
    parser = argparse.ArgumentParser(
        description="Build performance-weighted codebook from configs.jsonl")
    parser.add_argument("configs", help="Path to configs.jsonl")
    parser.add_argument("--beta", type=float, nargs="+", default=[0, 10, 50],
                        help="Boltzmann beta values to try (default: 0 10 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for codebook JSON (uses last beta)")
    parser.add_argument("--n_val", type=int, default=200,
                        help="Val set size for bound computation")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Confidence parameter")
    parser.add_argument("--compare-strategies", type=str, nargs="*", default=[],
                        help="Strategy JSON files to encode under each codebook")
    parser.add_argument("--name", type=str, default="codebook",
                        help="Codebook name prefix")
    parser.add_argument("--exclude-menus", type=str, nargs="*", default=["random_seed"],
                        help="Menu names to exclude (default: random_seed)")
    args = parser.parse_args()

    configs = load_configs(args.configs)
    print(f"Loaded {len(configs)} configs from {args.configs}")

    # Load strategy files if provided
    strategies = {}
    for sp in args.compare_strategies:
        p = Path(sp)
        with open(p) as f:
            strategies[p.stem] = json.load(f)

    codebooks = {}
    for beta in args.beta:
        exclude = set(args.exclude_menus) if args.exclude_menus else set()
        cb = build_codebook(configs, beta, name=f"{args.name}-beta{beta}",
                            exclude_menus=exclude)
        codebooks[beta] = cb
        print_codebook_summary(cb)

        if strategies:
            print(f"\n  Strategy codelengths (beta={beta}):")
            print(f"  {'Strategy':<45s} {'Bits':>7s} {'Bound':>7s}")
            print(f"  {'-'*60}")
            for sname, strategy in strategies.items():
                specified = encode_strategy_for_codebook(strategy, cb)
                bits, breakdown = compute_sparse_codelength(cb, specified)
                bound = mdl_bound(bits, args.n_val, args.delta)
                print(f"  {sname:<45s} {bits:>7.1f} {bound:>7.4f}")

    # Save the last (or only) codebook
    if args.output:
        last_beta = args.beta[-1]
        cb = codebooks[last_beta]
        with open(args.output, "w") as f:
            json.dump(cb, f, indent=2)
        print(f"\nSaved codebook (beta={last_beta}) to {args.output}")


if __name__ == "__main__":
    main()
