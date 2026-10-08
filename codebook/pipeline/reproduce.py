#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Codebook reproduce: implement a strategy from codebook selections.

A reproducer agent receives structured codebook selections (not free text)
and implements the strategy. The agent has NO access to validation data.

Usage:
    python codebook/pipeline/reproduce.py codebook/experiments/folktables-5k--full-aggressive/ \
        --checkpoints 4e9a8b4 17857bf ...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
CODEBOOK_ROOT = PIPELINE_DIR.parent
PROJECT_ROOT = CODEBOOK_ROOT.parent
sys.path.insert(0, str(CODEBOOK_ROOT))

from schema import compute_sparse_codelength, mdl_bound


def build_reproduce_prompt(codebook: dict, selections: dict,
                           codelength_bits: float, experiment_conf: dict) -> str:
    """Build the reproduce prompt from codebook selections."""

    task_desc = experiment_conf.get("TASK_DESC", "Binary classification task.")
    packages = experiment_conf.get("PACKAGES", "scikit-learn numpy")
    reproduce_imports = experiment_conf.get("REPRODUCE_IMPORTS",
        "from prepare_reproduce import load_train, save_model, get_feature_names")
    reproduce_functions = experiment_conf.get("REPRODUCE_FUNCTIONS", "")
    banned = experiment_conf.get("BANNED_VAL_FUNCTIONS", "load_val, evaluate_val")
    run_cmd = experiment_conf.get("RUN_CMD", "uv run python train.py")

    # Format selections as a readable specification
    on_menu = {k: v for k, v in selections.items() if not k.startswith("off_menu_")}
    off_menu = {k: v for k, v in selections.items() if k.startswith("off_menu_")}

    spec_lines = []
    for k, v in on_menu.items():
        if isinstance(v, str):
            spec_lines.append(f"- **{k}**: {v}")
        else:
            spec_lines.append(f"- **{k}**: {v}")

    for k, v in off_menu.items():
        clean_name = k.replace("off_menu_", "")
        spec_lines.append(f"- **{clean_name}**: {v}")

    spec_text = "\n".join(spec_lines)

    return f"""# Reproduction Agent

You are an ML agent. Implement the strategy specified below in `train.py` and run it.

## CRITICAL: No validation data access

There is NO `prepare.py` in this directory. Only `prepare_reproduce.py` exists.
You have NO access to validation data — no {banned}.
You will NEVER see any validation accuracy. Your job is to train a model and save it.

## Your interface

Your `train.py` must import from `prepare_reproduce` (NOT `prepare`):
```python
{reproduce_imports}
```

## Available functions (from prepare_reproduce.py, read-only)

{reproduce_functions}

You have: {packages}.

## Task

{task_desc}

## Strategy specification

Implement a model with these exact settings:

{spec_text}

This specification was encoded in {codelength_bits:.1f} bits under a structured codebook.
Implement it faithfully — use the exact values specified. Do not add adaptive search,
seed sweeps, or hyperparameter tuning beyond what is specified.

If a parameter is not specified above, use library defaults.
If the specification mentions techniques requiring val data, implement the best
train-only approximation.

## Instructions

1. Read `prepare_reproduce.py` to understand what's available.
2. Implement the strategy above in `train.py`. Import ONLY from `prepare_reproduce`.
3. At the end of training, call `save_model(model)` to save your trained model.
4. Run `{run_cmd}` and verify it completes without errors.
5. You may fix bugs and re-run, but do NOT add any adaptive search.
6. Stop after getting a working result. Do not run more than 3 attempts.
"""


def run_reproduce(cb_exp_dir: Path, checkpoint: str, config: dict):
    """Run codebook reproduction for one checkpoint."""
    experiment = config["experiment"]
    dataset_dir = PROJECT_ROOT / experiment
    short = checkpoint[:7]

    # Load codebook
    with open(cb_exp_dir / "codebook.json") as f:
        codebook = json.load(f)

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

    # Load codebook selections from compress step
    ckpt_dir = cb_exp_dir / "checkpoints" / short
    compress_dir = ckpt_dir / "compress"
    selections_path = compress_dir / "codebook_selections.json"
    result_path = compress_dir / "codebook_result.json"

    if not selections_path.exists():
        print(f"ERROR: {selections_path} not found. Run codebook compress first.")
        return None

    with open(selections_path) as f:
        selections = json.load(f)

    codelength = 0.0
    if result_path.exists():
        with open(result_path) as f:
            compress_result = json.load(f)
        codelength = compress_result.get("codelength_bits", 0)

    # Create reproduce directory
    reproduce_dir = ckpt_dir / "reproduce"
    if reproduce_dir.exists():
        shutil.rmtree(reproduce_dir)
    reproduce_dir.mkdir(parents=True)

    # Copy reproduce files (NOT prepare.py — isolation!)
    reproduce_py = dataset_dir / "prepare_reproduce.py"
    if reproduce_py.exists():
        shutil.copy(reproduce_py, reproduce_dir / "prepare_reproduce.py")

    pyproject = dataset_dir / "pyproject.toml"
    if pyproject.exists():
        shutil.copy(pyproject, reproduce_dir / "pyproject.toml")

    # Symlink .venv if available
    for agent_dir in dataset_dir.glob("explore-agent-*"):
        venv = agent_dir / ".venv"
        if venv.exists():
            os.symlink(venv, reproduce_dir / ".venv")
            uv_lock = agent_dir / "uv.lock"
            if uv_lock.exists():
                shutil.copy(uv_lock, reproduce_dir / "uv.lock")
            break

    # Write blank train.py
    with open(reproduce_dir / "train.py", "w") as f:
        f.write("# Implement the strategy from the codebook specification\n")

    # Build and save reproduce prompt
    prompt = build_reproduce_prompt(codebook, selections, codelength, conf)
    with open(reproduce_dir / "reproduce_prompt.md", "w") as f:
        f.write(prompt)

    # Save codebook selections for reference
    with open(reproduce_dir / "codebook_selections.json", "w") as f:
        json.dump(selections, f, indent=2)

    # Init git repo
    subprocess.run(["git", "init"], cwd=reproduce_dir, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=reproduce_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"],
                   cwd=reproduce_dir, capture_output=True)

    condition = config["condition"]
    print(f"Reproducing {experiment} checkpoint {short} (condition: {condition})")
    print(f"  Codelength: {codelength:.1f} bits")
    print(f"  Output: {reproduce_dir}")

    # Run the reproduce agent
    subprocess.run(
        [
            "claude", "--dangerously-skip-permissions",
            "--max-turns", "100",
            "--model", "opus",
            "-p", prompt,
        ],
        cwd=reproduce_dir,
    )

    # Post-agent evaluation
    print("\n=== POST-AGENT EVALUATION ===")
    model_path = reproduce_dir / "model.pkl"
    if model_path.exists():
        subprocess.run(
            ["uv", "run", "python", "prepare_reproduce.py"],
            cwd=reproduce_dir,
        )
        holdout_path = reproduce_dir / "holdout_results.tsv"
        if holdout_path.exists():
            with open(holdout_path) as f:
                lines = f.readlines()
            if lines:
                last = lines[-1].strip().split("\t")
                val_acc = float(last[0]) if len(last) > 0 else None
                holdout_acc = float(last[1]) if len(last) > 1 else None
                print(f"  Reproducer val: {val_acc}")
                print(f"  Reproducer holdout: {holdout_acc}")
                return {
                    "checkpoint": short,
                    "condition": condition,
                    "codelength_bits": codelength,
                    "reproducer_val": val_acc,
                    "reproducer_holdout": holdout_acc,
                }
    else:
        print("  WARNING: No model.pkl found")

    return {"checkpoint": short, "condition": condition,
            "codelength_bits": codelength}


def main():
    parser = argparse.ArgumentParser(description="Codebook reproduce")
    parser.add_argument("experiment_dir", help="Experiment dir (e.g., codebook/experiments/folktables-5k--full-aggressive/)")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    results = []
    for ckpt in args.checkpoints:
        r = run_reproduce(exp_dir, ckpt, config)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"Reproduced {len(results)} checkpoints")
        for r in results:
            val = r.get("reproducer_val", "?")
            hold = r.get("reproducer_holdout", "?")
            print(f"  {r['checkpoint']}: {r['codelength_bits']:.1f} bits, "
                  f"val={val}, holdout={hold}")


if __name__ == "__main__":
    main()
