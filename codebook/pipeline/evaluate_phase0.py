#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Evaluate the best Phase 0 config on val + holdout.

Reads configs.jsonl from Phase 0 build dir, trains the best config on full
training data, evaluates on val and holdout (secretly), and saves results
to the experiment's config.json. Also writes baseline_train.py — a working
training script that the Phase 1 explorer can start from.

Usage:
    python codebook/pipeline/evaluate_phase0.py codebook/experiments/folktables-5k--full-analyst/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
CODEBOOK_ROOT = PIPELINE_DIR.parent
PROJECT_ROOT = CODEBOOK_ROOT.parent


def load_configs(configs_path: Path) -> list[dict]:
    configs = []
    with open(configs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                configs.append(json.loads(line))
    return configs


def build_train_script(config: dict, experiment_conf: dict) -> str:
    """Generate a train.py that implements the Phase 0 config."""
    menus = config["specified_menus"]
    model_name = menus.get("model", "LogisticRegression")

    # Build the script based on model type
    imports = ["from prepare import load_train, evaluate_val, save_model"]
    lines = ["X_train, y_train = load_train()"]

    if model_name in ("GradientBoosting", "GradientBoostingClassifier"):
        imports.append("from sklearn.ensemble import GradientBoostingClassifier")
        params = {}
        if "max_depth" in menus:
            params["max_depth"] = menus["max_depth"]
        if "n_estimators" in menus:
            params["n_estimators"] = menus["n_estimators"]
        if "learning_rate" in menus:
            params["learning_rate"] = menus["learning_rate"]
        if "subsample" in menus:
            params["subsample"] = menus["subsample"]
        params["random_state"] = 42
        param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        lines.append(f"model = GradientBoostingClassifier({param_str})")

    elif model_name in ("HistGradientBoosting", "HistGradientBoostingClassifier"):
        imports.append("from sklearn.ensemble import HistGradientBoostingClassifier")
        params = {}
        if "max_depth" in menus:
            params["max_depth"] = menus["max_depth"]
        if "max_iter" in menus:
            params["max_iter"] = menus["max_iter"]
        if "learning_rate" in menus:
            params["learning_rate"] = menus["learning_rate"]
        if "l2_regularization" in menus:
            params["l2_regularization"] = menus["l2_regularization"]
        params["random_state"] = 42
        param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        lines.append(f"model = HistGradientBoostingClassifier({param_str})")

    elif model_name in ("RandomForest", "RandomForestClassifier"):
        imports.append("from sklearn.ensemble import RandomForestClassifier")
        params = {}
        if "n_estimators" in menus:
            params["n_estimators"] = menus["n_estimators"]
        if "max_depth" in menus:
            params["max_depth"] = menus["max_depth"]
        params["random_state"] = 42
        param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        lines.append(f"model = RandomForestClassifier({param_str})")

    elif model_name == "LogisticRegression":
        imports.append("from sklearn.linear_model import LogisticRegression")
        imports.append("from sklearn.preprocessing import StandardScaler")
        imports.append("from sklearn.pipeline import Pipeline")
        lines.append("model = Pipeline([")
        lines.append('    ("scaler", StandardScaler()),')
        params = {"max_iter": 1000, "random_state": 42}
        if "C" in menus:
            params["C"] = menus["C"]
        if "penalty" in menus:
            params["penalty"] = menus["penalty"]
        param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        lines.append(f'    ("clf", LogisticRegression({param_str})),')
        lines.append("])")

    else:
        # Fallback: generic
        lines.append(f"# Model: {model_name}")
        lines.append(f"# Menus: {json.dumps(menus)}")
        lines.append("raise NotImplementedError(f'Unknown model: {model_name}')")

    # Add preprocessing if specified and not already in pipeline
    if menus.get("preprocessing") == "StandardScaler" and model_name != "LogisticRegression":
        imports.append("from sklearn.preprocessing import StandardScaler")
        imports.append("from sklearn.pipeline import Pipeline")
        # Wrap in pipeline
        model_line = [l for l in lines if l.startswith("model = ")][0]
        lines = [l for l in lines if not l.startswith("model = ")]
        inner_model = model_line.replace("model = ", "")
        lines.append("model = Pipeline([")
        lines.append('    ("scaler", StandardScaler()),')
        lines.append(f'    ("clf", {inner_model}),')
        lines.append("])")

    lines.append("model.fit(X_train, y_train)")
    lines.append("evaluate_val(model)")
    lines.append("save_model(model)")

    script = '"""Phase 0 baseline: best config from train-only CV exploration."""\n\n'
    script += "\n".join(imports) + "\n\n"
    script += "\n".join(lines) + "\n"
    return script


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 0 baseline")
    parser.add_argument("experiment_dir", help="Codebook experiment dir")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    experiment = config["experiment"]
    dataset_dir = PROJECT_ROOT / experiment

    # Load Phase 0 configs
    build_dir = exp_dir / "build"
    configs_path = build_dir / "configs.jsonl"
    if not configs_path.exists():
        print(f"ERROR: {configs_path} not found. Run Phase 0 first.")
        sys.exit(1)

    configs = load_configs(configs_path)
    print(f"Loaded {len(configs)} Phase 0 configs")

    # Find best config
    best = max(configs, key=lambda c: c["cv_accuracy"])
    print(f"Best Phase 0 config: iter {best.get('iter', '?')}, "
          f"CV={best['cv_accuracy']:.4f}")
    print(f"  {best.get('description', '')}")
    print(f"  Menus: {json.dumps(best['specified_menus'])}")

    # Read experiment.conf
    conf = {}
    conf_path = dataset_dir / "experiment.conf"
    if conf_path.exists():
        with open(conf_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    conf[key.strip()] = val.strip().strip('"')

    # Generate baseline train.py
    train_script = build_train_script(best, conf)
    baseline_path = exp_dir / "baseline_train.py"
    with open(baseline_path, "w") as f:
        f.write(train_script)
    print(f"\nBaseline train.py written to {baseline_path}")
    print(f"--- begin ---")
    print(train_script.strip())
    print(f"--- end ---\n")

    # Evaluate on val + holdout
    # Run in a temp dir with the experiment's prepare.py and venv
    import tempfile
    import shutil
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy(dataset_dir / "prepare.py", tmpdir / "prepare.py")
        shutil.copy(dataset_dir / "pyproject.toml", tmpdir / "pyproject.toml")

        # Symlink venv
        for agent_dir in dataset_dir.glob("explore-agent-*"):
            venv = agent_dir / ".venv"
            if venv.exists():
                os.symlink(venv, tmpdir / ".venv")
                uv_lock = agent_dir / "uv.lock"
                if uv_lock.exists():
                    shutil.copy(uv_lock, tmpdir / "uv.lock")
                break

        shutil.copy(baseline_path, tmpdir / "train.py")

        print("Running baseline evaluation...")
        result = subprocess.run(
            ["uv", "run", "python", "train.py"],
            cwd=tmpdir, capture_output=True, text=True, timeout=120,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            sys.exit(1)

        # Read holdout results
        holdout_path = tmpdir / "holdout_results.tsv"
        if holdout_path.exists():
            with open(holdout_path) as f:
                lines = f.readlines()
            if lines:
                last = lines[-1].strip().split("\t")
                # Format: timestamp \t hash \t val \t holdout
                val_acc = float(last[2]) if len(last) > 2 else float(last[0])
                holdout_acc = float(last[3]) if len(last) > 3 else float(last[1])
            else:
                print("ERROR: holdout_results.tsv is empty")
                sys.exit(1)
        else:
            print("ERROR: holdout_results.tsv not generated")
            sys.exit(1)

    print(f"\nPhase 0 Baseline Results:")
    print(f"  CV accuracy (train):  {best['cv_accuracy']:.4f}")
    print(f"  Val accuracy:         {val_acc:.4f}")
    print(f"  Holdout accuracy:     {holdout_acc:.4f}")
    print(f"  Val-holdout gap:      {(val_acc - holdout_acc)*100:.1f}pp")

    # Union bound
    import math
    n_val = config["n_val"]
    delta = config.get("delta", 0.05)
    n_configs = len(configs)
    union_bound = math.sqrt(math.log(2 * n_configs / delta) / (2 * n_val))
    print(f"  Union bound (N={n_configs}, n={n_val}): ±{union_bound*100:.1f}pp")

    # Update config.json
    config["phase0_baseline"] = {
        "description": best.get("description", ""),
        "config": best["specified_menus"],
        "cv_accuracy": best["cv_accuracy"],
        "val_accuracy": val_acc,
        "holdout_accuracy": holdout_acc,
        "n_configs": n_configs,
        "union_bound": round(union_bound, 6),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nUpdated {exp_dir / 'config.json'}")


if __name__ == "__main__":
    main()
