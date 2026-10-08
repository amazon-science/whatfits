#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""End-to-end test: codebook encoding → MDL bound → holdout comparison.

For each checkpoint in a sweep:
1. Read the compressed prompt
2. Auto-encode it as codebook selections
3. Compute codelength under the performance-weighted prior
4. Compute MDL bound → confidence interval on true accuracy
5. Compare with actual holdout accuracy

Usage:
    python codebook/end_to_end_test.py folktables --codebook codebook/folktables-perf.json
    python codebook/end_to_end_test.py gene-expr --codebook codebook/gene-expr-perf.json --n_val 1000
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codebook"))

from schema import compute_sparse_codelength, mdl_bound


# ---------------------------------------------------------------------------
# Strategy auto-encoding from compressed prompts
# ---------------------------------------------------------------------------

def auto_encode_folktables(prompt_text: str) -> dict:
    """Parse a Folktables compressed prompt into codebook menu selections.

    Extracts model, hyperparameters, preprocessing, feature engineering, etc.
    from the free-form compressed prompt text.
    """
    text = prompt_text.lower()
    selections = {}

    # Model family (check more specific first)
    if "logisticregression" in text or "logistic_regression" in text:
        selections["model"] = "LogisticRegression"
    elif "gradientboostingclassifier" in text or "gradientboosting" in text:
        selections["model"] = "GradientBoosting"
    elif "histgradientboosting" in text:
        selections["model"] = "HistGradientBoosting"
    elif "randomforest" in text or "rf(" in text:
        selections["model"] = "RandomForest"
    elif "svm" in text or "svc" in text:
        selections["model"] = "SVM"
    elif "knn" in text or "kneighbors" in text:
        selections["model"] = "KNN"

    # n_estimators
    m = re.search(r"n_estimators\s*=\s*(\d+)", prompt_text)
    if m:
        selections["n_estimators"] = int(m.group(1))

    # max_depth
    m = re.search(r"max_depth\s*=\s*(\d+)", prompt_text)
    if m:
        selections["max_depth"] = int(m.group(1))

    # learning_rate
    m = re.search(r"learning_rate\s*=\s*([\d.]+)", prompt_text)
    if m:
        selections["learning_rate"] = float(m.group(1))

    # preprocessing
    if "standardscaler" in text:
        selections["preprocessing"] = "StandardScaler"

    # class_weight
    if "class_weight" in text:
        if "balanced" in text:
            selections["class_weight"] = "balanced"
        elif "{" in prompt_text and "class_weight" in prompt_text:
            selections["class_weight"] = "custom"

    # Feature engineering — check for interactions, target encoding, etc.
    has_interactions = any(x in text for x in ["age*hours", "col0*col7", "interactions", "age²", "col0**2"])
    has_target_encoding = "target-encode" in text or "target_encod" in text
    has_polynomial = "poly" in text

    fe_list = []
    if has_interactions:
        fe_list.append("interactions")
    if has_target_encoding:
        fe_list.append("target_encoding")
    if has_polynomial:
        fe_list.append("polynomial")
    if fe_list:
        selections["feature_engineering"] = fe_list

    # Threshold tuning
    m = re.search(r"threshold\s*([\d.]+)", text)
    if not m:
        m = re.search(r">=\s*([\d.]+?)(?:\s|$|[,;)])", text)
    if m:
        try:
            thresh = float(m.group(1).rstrip("."))
            if thresh != 0.5:  # non-default threshold
                selections["threshold"] = thresh
        except ValueError:
            pass

    # min_samples_leaf
    m = re.search(r"min_samples_leaf\s*=\s*(\d+)", prompt_text)
    if m:
        selections["min_samples_leaf"] = int(m.group(1))

    # subsample
    m = re.search(r"subsample\s*=\s*([\d.]+)", prompt_text)
    if m:
        selections["subsample"] = float(m.group(1))

    return selections


def auto_encode_gene_expr(prompt_text: str) -> dict:
    """Parse a gene-expr compressed prompt into codebook menu selections."""
    text = prompt_text.lower()
    selections = {}

    # Model family
    if "logisticregression" in text or "logistic regression" in text:
        selections["model"] = "LogisticRegression"
    elif "randomforest" in text or "random forest" in text:
        selections["model"] = "RandomForest"
    elif "xgboost" in text or "xgb" in text:
        selections["model"] = "XGBoost"
    elif "svm" in text or "svc" in text or "linearsvc" in text:
        selections["model"] = "SVM"

    # Penalty/regularization type
    if "penalty='l1'" in text or 'penalty="l1"' in text or "l1" in text:
        selections["penalty"] = "l1"
    elif "penalty='l2'" in text or "l2 penalty" in text or "l2 regulariz" in text:
        selections["penalty"] = "l2"
    elif "elastic" in text:
        selections["penalty"] = "elasticnet"

    # C (regularization strength)
    m = re.search(r"\bC\s*=\s*([\d.]+)", prompt_text)
    if m:
        selections["C"] = float(m.group(1))

    # Feature selection method
    if "selectkbest" in text or "f_classif" in text or "anova" in text:
        selections["feature_selection"] = "anova_f_classif"
        # k (number of features selected)
        m = re.search(r"(?:k\s*=\s*|top[- ]?)(\d+)", text)
        if m:
            selections["n_features_selected"] = int(m.group(1))
    elif "l1" in text and ("select" in text or "nonzero" in text or "sparsity" in text):
        selections["feature_selection"] = "l1_sparsity"
    elif "specific feature" in text or "feature indices" in text or re.search(r"\[\d+,\s*\d+", prompt_text):
        selections["feature_selection"] = "specific_indices"
        # Count specified features
        indices = re.findall(r"\d{3,5}", prompt_text)
        if len(indices) >= 3:  # likely feature indices
            selections["n_features_selected"] = len(set(indices))

    # Preprocessing
    if "standardscaler" in text:
        selections["preprocessing"] = "StandardScaler"

    # max_iter
    m = re.search(r"max_iter\s*=\s*(\d+)", prompt_text)
    if m:
        selections["max_iter"] = int(m.group(1))

    # solver
    m = re.search(r"solver\s*=\s*['\"](\w+)['\"]", prompt_text)
    if m:
        selections["solver"] = m.group(1)

    return selections


# ---------------------------------------------------------------------------
# Compute codelength with escape penalties for off-menu items
# ---------------------------------------------------------------------------

def compute_codelength_with_escapes(codebook: dict, selections: dict,
                                     escape_bits: float = 12.0) -> tuple[float, dict]:
    """Compute codelength, handling off-menu values with escape penalty.

    Uses compute_sparse_codelength for on-menu items, adds escape penalty
    for items not in the codebook.
    """
    menu_names = {m["name"] for m in codebook["menus"]}

    on_menu = {k: v for k, v in selections.items() if k in menu_names}
    off_menu = {k: v for k, v in selections.items() if k not in menu_names}

    # Compute on-menu codelength
    bits, breakdown = compute_sparse_codelength(codebook, on_menu)

    # Add escape penalties for off-menu items
    for k, v in off_menu.items():
        penalty = escape_bits
        breakdown[k] = {"escape": True, "total": penalty}
        bits += penalty

    return bits, breakdown


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_end_to_end(experiment: str, codebook_path: str, n_val: int,
                    delta: float, sweep_conditions: list[str]):
    """Run the end-to-end test for an experiment."""
    exp_dir = PROJECT_ROOT / experiment
    codebook_dir = PROJECT_ROOT / "codebook"

    # Load codebook
    with open(codebook_path) as f:
        codebook = json.load(f)

    print(f"{'='*90}")
    print(f"End-to-End MDL Bound Test: {experiment}")
    print(f"Codebook: {codebook_path} (beta={codebook.get('beta', '?')})")
    print(f"n_val={n_val}, delta={delta}")
    print(f"{'='*90}\n")

    all_rows = []

    for condition in sweep_conditions:
        sweep_file = exp_dir / f"sweep_results-{condition}.csv"
        if not sweep_file.exists():
            print(f"  Sweep file not found: {sweep_file}")
            continue

        # Determine which compress directories to look in
        # full-analyst reuses full-aggressive compress dirs
        compress_condition = condition
        if "analyst" in condition:
            compress_condition = condition.replace("analyst", "aggressive")

        print(f"\n--- Condition: {condition} ---\n")

        with open(sweep_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ckpt = row["checkpoint"]
                explorer_val = float(row["explorer_val"]) if row["explorer_val"] else None
                reproducer_val = float(row["reproducer_val"]) if row["reproducer_val"] else None
                reproducer_holdout = float(row["reproducer_holdout"]) if row["reproducer_holdout"] else None
                gap_pp = float(row["gap_pp"]) if row["gap_pp"] else None
                cert_status = row["status"]

                if reproducer_val is None or reproducer_holdout is None:
                    continue

                # Read compressed prompt
                compress_dir = exp_dir / f"compress-{compress_condition}-{ckpt}"
                prompt_file = compress_dir / "compressed_prompt.md"
                if not prompt_file.exists():
                    print(f"  {ckpt}: compressed_prompt.md not found, skipping")
                    continue

                prompt_text = prompt_file.read_text()

                # Auto-encode
                if experiment == "folktables":
                    selections = auto_encode_folktables(prompt_text)
                elif experiment == "gene-expr":
                    selections = auto_encode_gene_expr(prompt_text)
                else:
                    print(f"  {ckpt}: no auto-encoder for {experiment}, skipping")
                    continue

                # Compute codelength
                bits, breakdown = compute_codelength_with_escapes(
                    codebook, selections, codebook.get("escape_penalty_bits", 12.0))

                # MDL bound
                bound = mdl_bound(bits, n_val, delta)

                # Confidence interval on true accuracy
                ci_lower = reproducer_val - bound
                ci_upper = reproducer_val + bound

                # Does holdout fall within CI?
                holdout_in_ci = ci_lower <= reproducer_holdout <= ci_upper

                # Explorer overfit gap
                if explorer_val is not None:
                    overfit_gap = explorer_val - reproducer_holdout
                else:
                    overfit_gap = None

                result = {
                    "condition": condition,
                    "checkpoint": ckpt,
                    "explorer_val": explorer_val,
                    "reproducer_val": reproducer_val,
                    "reproducer_holdout": reproducer_holdout,
                    "cert_status": cert_status,
                    "k_bits": round(bits, 1),
                    "mdl_bound": round(bound, 4),
                    "ci_lower": round(ci_lower, 4),
                    "ci_upper": round(ci_upper, 4),
                    "holdout_in_ci": holdout_in_ci,
                    "overfit_gap": round(overfit_gap, 4) if overfit_gap is not None else None,
                    "selections": selections,
                }
                all_rows.append(result)

    # Print results table
    print(f"\n{'='*90}")
    print(f"{'Ckpt':<9s} {'Expl':>6s} {'Repr':>6s} {'Hold':>6s} "
          f"{'Cert':>5s} {'k(bits)':>8s} {'Bound':>7s} "
          f"{'CI':>18s} {'InCI?':>5s} {'Overfit':>7s}")
    print(f"{'-'*90}")

    for r in all_rows:
        ci_str = f"[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]"
        overfit_str = f"{r['overfit_gap']:.3f}" if r['overfit_gap'] is not None else "—"
        in_ci_str = "YES" if r['holdout_in_ci'] else "NO"

        print(f"{r['checkpoint']:<9s} {r['explorer_val']:>6.3f} {r['reproducer_val']:>6.3f} "
              f"{r['reproducer_holdout']:>6.3f} {r['cert_status']:>5s} {r['k_bits']:>7.1f}  "
              f"{r['mdl_bound']:>6.4f} {ci_str:>18s} {in_ci_str:>5s} {overfit_str:>7s}")

    # Summary statistics
    print(f"\n{'='*90}")
    n_total = len(all_rows)
    n_in_ci = sum(1 for r in all_rows if r["holdout_in_ci"])
    n_informative = sum(1 for r in all_rows if r["mdl_bound"] < 0.5)
    avg_bits = sum(r["k_bits"] for r in all_rows) / n_total if n_total else 0
    avg_bound = sum(r["mdl_bound"] for r in all_rows) / n_total if n_total else 0

    print(f"Total checkpoints: {n_total}")
    print(f"Holdout within CI: {n_in_ci}/{n_total} ({100*n_in_ci/n_total:.0f}%)")
    print(f"Informative bounds (<50%): {n_informative}/{n_total}")
    print(f"Avg codelength: {avg_bits:.1f} bits")
    print(f"Avg bound: ±{avg_bound:.4f} ({avg_bound*100:.1f}pp)")

    # Separation analysis
    honest = [r for r in all_rows if r["cert_status"] == "PASS"]
    exploit = [r for r in all_rows if r["cert_status"] == "FAIL"]
    if honest and exploit:
        avg_honest_bits = sum(r["k_bits"] for r in honest) / len(honest)
        avg_exploit_bits = sum(r["k_bits"] for r in exploit) / len(exploit)
        print(f"\nHonest strategies (PASS): {len(honest)}, avg {avg_honest_bits:.1f} bits")
        print(f"Exploit strategies (FAIL): {len(exploit)}, avg {avg_exploit_bits:.1f} bits")
        print(f"Separation ratio: {avg_exploit_bits/avg_honest_bits:.1f}x")

    # Theoretical projections at different n_val
    print(f"\n--- Theoretical bound at different n_val ---")
    for n in [200, 500, 1000, 5000, 10000]:
        bounds = [mdl_bound(r["k_bits"], n, delta) for r in all_rows]
        avg_b = sum(bounds) / len(bounds)
        min_b = min(bounds)
        max_b = max(bounds)
        vac = "VACUOUS" if avg_b > 0.5 else "informative"
        print(f"  n={n:>6d}: avg ±{avg_b:.4f} ({avg_b*100:.1f}pp), "
              f"range [{min_b:.4f}, {max_b:.4f}]  [{vac}]")

    # Save results
    out_path = codebook_dir / f"e2e_results_{experiment}.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save CSV for easy analysis
    csv_path = codebook_dir / f"e2e_results_{experiment}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "checkpoint", "explorer_val", "reproducer_val",
            "reproducer_holdout", "cert_status", "k_bits", "mdl_bound",
            "ci_lower", "ci_upper", "holdout_in_ci", "overfit_gap",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV saved to {csv_path}")

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="End-to-end MDL bound test")
    parser.add_argument("experiment", help="Experiment directory name")
    parser.add_argument("--codebook", required=True, help="Codebook JSON path")
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--conditions", nargs="+",
                        default=["full-analyst", "scoreonly-aggressive"],
                        help="Sweep conditions to test")
    args = parser.parse_args()

    run_end_to_end(args.experiment, args.codebook, args.n_val, args.delta,
                   args.conditions)


if __name__ == "__main__":
    main()
