#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run an agent to build a codebook prior via sequential exploration on training data.

The agent iteratively improves a model using only cross-validation on training
data (no val access). Each iteration builds on the previous, logging configs
to configs.jsonl. The codebook is then built from these configs by a separate
script (build_codebook_from_configs.py) using performance-weighted priors.

Usage:
    python codebook/build_codebook_agent_v2.py folktables
    python codebook/build_codebook_agent_v2.py folktables --max-iters 15

Then build the codebook:
    python codebook/build_codebook_from_configs.py codebook/build-<experiment>/configs.jsonl \
        --beta 0 10 50 --output codebook/<experiment>-perf.json
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
CODEBOOK_ROOT = PIPELINE_DIR.parent
PROJECT_ROOT = CODEBOOK_ROOT.parent


def build_prompt(experiment: str, experiment_dir: Path, max_iters: int) -> str:
    """Build the sequential exploration prompt."""

    # Read prepare.py to understand available functions
    prepare_path = experiment_dir / "prepare.py"
    with open(prepare_path) as f:
        prepare_code = f.read()

    # Read experiment.conf for task description
    conf_path = experiment_dir / "experiment.conf"
    task_desc = ""
    packages = ""
    if conf_path.exists():
        with open(conf_path) as f:
            for line in f:
                if line.startswith("TASK_DESC="):
                    task_desc = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("PACKAGES="):
                    packages = line.split("=", 1)[1].strip().strip('"')

    return f"""# Sequential Exploration: Build a Training-Data Prior for {experiment}

You are exploring the strategy space for an ML task using ONLY training data.
Your goal is to iteratively improve a model through sequential experimentation,
logging every configuration you try. This log will be used to build a prior
distribution over ML strategies.

## The task

{task_desc}

## CRITICAL RULES

1. **You may ONLY use training data.** Call `load_train()` to get (X_train, y_train).
   Do NOT call `load_val()`, `evaluate_val()`, or any validation function.
   The exploration must use only training data — no val data at any point.

2. **Use cross-validation** (e.g., 5-fold CV) on training data to evaluate each config.

3. **Work sequentially.** Each iteration should build on what you learned from previous ones.
   Start simple, then refine based on what works and what doesn't.

4. **Run exactly {max_iters} iterations.** Each iteration = one complete configuration.

5. **Your output is one file**: `configs.jsonl` — one JSON line per iteration.

## Available packages

{packages}

## Available functions from prepare.py

- `load_train()` → (X_train, y_train) — numpy arrays
- `get_feature_names()` → list of feature names

That's ALL you can use. No val functions exist in your environment.

## How to work

Write a Python script called `explore.py` that does the following:

For each iteration (1 to {max_iters}):
1. Define a complete training configuration (model, hyperparameters, preprocessing, etc.)
2. Evaluate it using 5-fold cross-validation on training data
3. Print the CV accuracy and what you learned
4. Log the config to `configs.jsonl`
5. Based on the result, decide what to try next

The script should run all {max_iters} iterations in sequence, printing progress.

### Iteration flow

- **Iteration 1**: Start with a simple baseline (e.g., LogisticRegression with defaults)
- **Iteration 2**: Based on iteration 1's result, try an improvement (different model, add preprocessing, etc.)
- **Iteration 3+**: Continue refining. Try the most promising direction from previous results.

You should explore broadly at first (try different model families), then narrow down
and refine the best approaches. A good trajectory might look like:

1. LogisticRegression baseline → 0.78
2. Add StandardScaler → 0.79
3. Try RandomForest → 0.80
4. Try GradientBoosting → 0.81
5. Tune GBT depth → 0.811
6. Tune GBT n_estimators → 0.812
7. Try HistGradientBoosting → 0.813
8. Try feature interactions → 0.808 (didn't help)
9. Try different learning rate → 0.814
10. ...

### configs.jsonl format

Each line is a JSON object:

```json
{{"iter": 1, "cv_accuracy": 0.7889, "description": "LogisticRegression baseline with default params", "specified_menus": {{"model": "LogisticRegression"}}}}
```

The `specified_menus` dict contains ONLY the parameters you explicitly set.
Parameters left at library defaults should be OMITTED. This distinction matters —
it tells us which choices are deliberate.

The `description` field should briefly explain what this iteration tried and why
(e.g., "Switched to GBT based on RF result" or "Tuned learning rate from 0.1 to 0.05").

### What to include in specified_menus

Include any parameter you explicitly chose:
- `model`: the model class name (always include this)
- `n_estimators`, `max_depth`, `learning_rate`: if you set them explicitly
- `preprocessing`: if you added scaling, encoding, etc.
- `feature_engineering`: if you added interactions, polynomial features, etc.
- `class_weight`: if you set balanced or custom weights
- `random_seed`: if you set a specific seed
- Any other hyperparameter you explicitly chose

Do NOT include parameters you left at defaults. For example, if you use
`GradientBoostingClassifier()` with no arguments, specified_menus should just
be `{{"model": "GradientBoosting"}}`.

## Important

- Log EVERY iteration to configs.jsonl, including ones that performed worse
- The script should be self-contained — run with `uv run python explore.py`
- Print a summary at the end showing all iterations and their CV accuracies
- Focus on finding what works, not on finding the absolute optimum
- Try at least 3 different model families across your {max_iters} iterations
"""


def main():
    parser = argparse.ArgumentParser(description="Sequential codebook exploration")
    parser.add_argument("experiment", help="Experiment directory name (e.g., folktables)")
    parser.add_argument("--max-iters", type=int, default=12,
                        help="Number of sequential iterations (default: 12)")
    parser.add_argument("--max-turns", type=int, default=50,
                        help="Max agent turns (default: 50)")
    args = parser.parse_args()

    experiment_dir = PROJECT_ROOT / args.experiment
    if not experiment_dir.exists():
        print(f"ERROR: {experiment_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Create working directory
    work_dir = CODEBOOK_ROOT / f"build-{args.experiment}"
    if work_dir.exists():
        # Back up existing configs if any
        old_configs = work_dir / "configs.jsonl"
        if old_configs.exists():
            backup = work_dir / "configs.jsonl.bak"
            shutil.copy(old_configs, backup)
            print(f"Backed up existing configs to {backup}")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # Copy only prepare.py (for train data) and pyproject.toml
    shutil.copy(experiment_dir / "prepare.py", work_dir / "prepare.py")
    shutil.copy(experiment_dir / "pyproject.toml", work_dir / "pyproject.toml")

    # Symlink .venv if available
    for agent_dir in experiment_dir.glob("explore-agent-*"):
        venv = agent_dir / ".venv"
        if venv.exists():
            os.symlink(venv, work_dir / ".venv")
            uv_lock = agent_dir / "uv.lock"
            if uv_lock.exists():
                shutil.copy(uv_lock, work_dir / "uv.lock")
            break

    # Build the prompt
    prompt = build_prompt(args.experiment, experiment_dir, args.max_iters)
    prompt_path = work_dir / "build_prompt.md"
    with open(prompt_path, "w") as f:
        f.write(prompt)

    print(f"Working directory: {work_dir}")
    print(f"Sequential exploration: {args.max_iters} iterations")
    print(f"Launching agent (max {args.max_turns} turns)...")

    # Run the agent
    subprocess.run(
        [
            "claude", "--dangerously-skip-permissions",
            "--max-turns", str(args.max_turns),
            "--model", "sonnet",
            "-p", prompt,
        ],
        cwd=work_dir,
    )

    # Check for output
    configs_path = work_dir / "configs.jsonl"
    if configs_path.exists():
        import json
        configs = []
        with open(configs_path) as f:
            for line in f:
                if line.strip():
                    configs.append(json.loads(line))

        print(f"\n{'='*60}")
        print(f"Exploration complete: {len(configs)} configs logged")
        print(f"{'='*60}")
        for c in configs:
            desc = c.get("description", "")[:50]
            print(f"  Iter {c.get('iter', '?'):>2d}: CV={c['cv_accuracy']:.4f}  {desc}")

        accs = [c["cv_accuracy"] for c in configs]
        print(f"\n  Range: {min(accs):.4f} – {max(accs):.4f}")
        print(f"  Best:  {max(accs):.4f}")

        print(f"\nNext step: build codebook with performance-weighted priors:")
        print(f"  python codebook/build_codebook_from_configs.py {configs_path} \\")
        print(f"      --beta 0 10 50 --output codebook/{args.experiment}-perf.json")
    else:
        print("\nWARNING: configs.jsonl not found. The agent may not have completed.")


if __name__ == "__main__":
    main()
