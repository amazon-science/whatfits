#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Launch explorer starting from Phase 0 baseline.

Sets up the explorer agent directory with:
- Phase 0's best train.py as the starting point
- An explore prompt that includes Phase 0 results and tells the agent to beat it
- The standard auto-restart loop

Usage:
    python codebook/pipeline/run_explore_from_phase0.py codebook/experiments/folktables-5k--full-analyst/
    python codebook/pipeline/run_explore_from_phase0.py codebook/experiments/folktables-5k--full-analyst/ --max-turns 2000
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


def build_phase0_context(configs: list[dict], phase0_baseline: dict) -> str:
    """Build a summary of Phase 0 results to inject into the explore prompt."""
    lines = ["## Phase 0 Results (train-only cross-validation, no val access)"]
    lines.append("")
    lines.append("A prior exploration using only training data (5-fold CV) found these results:")
    lines.append("")
    lines.append("| Iter | CV Accuracy | Config |")
    lines.append("|---|---|---|")
    for c in configs:
        menus = c["specified_menus"]
        # Remove random_seed from display
        display_menus = {k: v for k, v in menus.items() if k != "random_seed"}
        config_str = ", ".join(f"{k}={v}" for k, v in display_menus.items())
        lines.append(f"| {c.get('iter', '?')} | {c['cv_accuracy']:.4f} | {config_str} |")
    lines.append("")

    best_cv = phase0_baseline["cv_accuracy"]
    best_val = phase0_baseline["val_accuracy"]
    best_desc = phase0_baseline.get("description", "")
    lines.append(f"**Best Phase 0 result**: CV={best_cv:.4f}, val={best_val:.4f} ({best_desc})")
    lines.append("")
    lines.append("Your starting `train.py` implements this best config. Your first `evaluate_val()` "
                 "call should confirm ~{:.3f} val accuracy.".format(best_val))
    lines.append("")
    lines.append("**Your goal**: improve beyond this baseline using val data access. "
                 "The baseline was found WITHOUT any val access — just train CV. "
                 "Can val data help you do better?")
    lines.append("")
    lines.append("Do NOT spend iterations on configs that perform worse than the baseline. "
                 "Start from the provided train.py and try to improve it.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Launch explorer from Phase 0 baseline")
    parser.add_argument("experiment_dir", help="Codebook experiment dir")
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--max-restarts", type=int, default=10)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    experiment = config["experiment"]
    condition = config["condition"]
    dataset_dir = PROJECT_ROOT / experiment
    phase0_baseline = config.get("phase0_baseline")

    if not phase0_baseline:
        print("ERROR: No phase0_baseline in config.json. Run evaluate_phase0.py first.")
        sys.exit(1)

    # Load Phase 0 configs
    configs_path = exp_dir / "build" / "configs.jsonl"
    configs = []
    with open(configs_path) as f:
        for line in f:
            if line.strip():
                configs.append(json.loads(line))

    # Check for baseline train.py
    baseline_train = exp_dir / "baseline_train.py"
    if not baseline_train.exists():
        print("ERROR: baseline_train.py not found. Run evaluate_phase0.py first.")
        sys.exit(1)

    # Determine explorer dir name
    explorer_dir = dataset_dir / f"explore-agent-{condition}"

    # Render the explore prompt using the main pipeline's template system
    # First set env vars, then call render_prompt.sh
    env = os.environ.copy()
    val_access = "full"
    framing = condition.split("-")[-1]  # "analyst" or "aggressive"
    env["VAL_ACCESS_LEVEL"] = val_access
    env["EXPLORER_FRAMING"] = framing

    # Generate Phase 0 context
    phase0_context = build_phase0_context(configs, phase0_baseline)

    if explorer_dir.exists():
        print(f"Explorer dir already exists: {explorer_dir}")
        print("To re-run, delete it first.")
        # Just launch the restart loop
    else:
        print(f"Creating explorer dir: {explorer_dir}")
        explorer_dir.mkdir(parents=True)

        # Copy experiment files
        for fname in ["prepare.py", "pyproject.toml"]:
            src = dataset_dir / fname
            if src.exists():
                shutil.copy(src, explorer_dir / fname)

        # Copy Phase 0 baseline as starting train.py
        shutil.copy(baseline_train, explorer_dir / "train.py")

        # Render explore prompt
        render_result = subprocess.run(
            ["bash", str(PROJECT_ROOT / "pipeline" / "render_prompt.sh"),
             str(dataset_dir), "explore"],
            capture_output=True, text=True, env=env,
        )
        base_prompt = render_result.stdout

        # Inject Phase 0 context after the first section
        full_prompt = base_prompt + "\n\n" + phase0_context

        with open(explorer_dir / "explore_prompt.md", "w") as f:
            f.write(full_prompt)

        (explorer_dir / "iterations").mkdir(exist_ok=True)

        # Init git
        subprocess.run(["git", "init"], cwd=explorer_dir, capture_output=True)
        subprocess.run(["git", "add", "prepare.py", "pyproject.toml",
                        "train.py", "explore_prompt.md"],
                       cwd=explorer_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "baseline (Phase 0)"],
                       cwd=explorer_dir, capture_output=True)

    # Launch the explorer with the restart loop
    print(f"Launching explorer in {explorer_dir}")
    print(f"Max turns: {args.max_turns}, Max restarts: {args.max_restarts}")

    # Use the main pipeline's restart loop
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "pipeline" / "run_explore.sh"),
         str(dataset_dir), str(args.max_turns), str(args.max_restarts)],
        env=env,
    )


if __name__ == "__main__":
    main()
