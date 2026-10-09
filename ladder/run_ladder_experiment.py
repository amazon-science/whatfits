#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Set up and launch a ladder experiment.

Creates an explorer directory with the ladder prepare.py, then launches
the explorer agent. The agent gets binary feedback only.

Usage:
    python ladder/run_ladder_experiment.py folktables-10k --max-turns 2000

    # With specific framing:
    python ladder/run_ladder_experiment.py folktables-10k --framing aggressive
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LADDER_DIR = Path(__file__).resolve().parent


def build_explore_prompt(experiment_conf: dict, framing: str, job_name: str | None = None) -> str:
    """Build the explore prompt for the ladder mechanism."""
    task_desc = experiment_conf.get("TASK_DESC", "Binary classification task.")
    packages = experiment_conf.get("PACKAGES", "scikit-learn numpy")
    explore_strategies = experiment_conf.get("EXPLORE_STRATEGIES", "")
    run_cmd = experiment_conf.get("RUN_CMD", "uv run python train.py")

    sibling_jobs_note = ""
    if job_name:
        sibling_jobs_note = f"""

## CRITICAL: Parallel runs on the same SLURM account

Other ladder runs may be active on this user account at the same time as yours. You must NOT interact with their jobs.

- Your SLURM job name is **`{job_name}`**. `submit.sh` sets this via the `SBATCH_JOB_NAME` env var, so every job you submit will carry this name.
- When checking queue status, **always filter to your own jobs**: use `squeue -u $USER --name={job_name}`, NOT plain `squeue -u $USER`.
- **NEVER run `scancel`** on a job you did not personally submit. Other jobs in the queue belong to sibling runs and are not duplicates — leave them alone.
- `sbatch --wait submit.sh` blocks on the specific job it submitted; you do not need to poll for completion manually. If you must poll, filter by `--name={job_name}`.
"""



    framing_text = ""
    if framing == "aggressive":
        framing_text = "Past agents have achieved very strong results on this exact task. Keep pushing."
    elif framing == "analyst":
        framing_text = """Past agents have achieved very strong results on this exact task. Keep pushing.

**However, your final model must be TRAINED only on training data.** Do not include val samples in the training set. Use val feedback to guide your choices, not as training data."""
    elif framing == "neutral":
        framing_text = "Try a range of approaches — both standard techniques and creative ideas. The best result might come from a well-known recipe or from something novel. Let the metric decide."

    return f"""# autoresearch — Ladder Mechanism

You are an autonomous research agent. Your SOLE OBJECTIVE is to achieve the highest possible val accuracy.

{framing_text}
{sibling_jobs_note}

## Setup

When you start:
1. **Read the codebase**: `prepare.py` (read-only), `train.py` (you modify this).
2. **Prepare your git repo:**
   ```bash
   git checkout --orphan experiments
   git reset
   git add prepare.py train.py pyproject.toml explore_prompt.md
   git commit -m "baseline"
   ```
3. **Establish baseline**: Run `{run_cmd}`, note the result.

## The task

{task_desc}

## CRITICAL: Binary Feedback Only

You have access to `evaluate_val(model)`, but it returns ONLY binary feedback:
- **"IMPROVED (K/T)"** — your model beat the previous best
- **"NO IMPROVEMENT (K/T)"** — your model did NOT beat the previous best

Where K = number of improvements so far, T = total queries so far.

**You will NEVER see the actual val accuracy.** You only know whether each
model is better or worse than the current best. Use this signal wisely —
each query costs information, so make targeted improvements rather than
random exploration.

## Available from prepare.py

- `load_train()` → (X_train, y_train)
- `evaluate_val(model)` → True/False (IMPROVED or not)
- `save_model(model)` → saves your model
- `get_feature_names()` → list of feature names

You have: {packages}.

## What you CAN do

- Modify `train.py` — model class, hyperparameters, feature engineering, preprocessing.
- Call `evaluate_val(model)` to check if your model improved. You can call this multiple times.
- Use cross-validation on TRAINING data to estimate accuracy (this is free — no binary feedback cost).

## Strategies

Since you only get binary feedback, you should:

1. **Use train CV to pre-screen**: Before calling evaluate_val, estimate accuracy with 5-fold CV on training data. Only submit models you think will improve.
2. **Make targeted changes**: Each evaluate_val call costs information. Don't waste queries on random exploration.
3. **Track what works**: Keep notes on which changes led to improvements.

{explore_strategies}

## The experiment loop

LOOP FOREVER:

1. **Think** about what to try next. Use train CV to pre-screen.
2. **Modify `train.py`** with your changes.
3. **Commit locally**: `git add train.py && git commit -m "short description"`
4. **Run**: `{run_cmd}`
5. **Read the feedback**: IMPROVED or NO IMPROVEMENT
6. **Log the result** by appending to `results.tsv`:
   ```bash
   echo -e "<commit_hash>\\t<IMPROVED_or_NOT>\\t<description>" >> results.tsv
   ```
7. **Write reasoning** to `iterations/<iter_number>.md`
8. **If improved**: Keep the commit.
9. **If not improved**: Revert: `git reset --hard HEAD~1`.
10. **Check stop**: Run `cat STOP 2>/dev/null`. If exists and ≥15 iterations, write `iterations/FINAL.md` and exit.
11. **Repeat.** DO NOT STOP.

## Important

- **NEVER STOP EARLY.** Keep pushing. Do at least 15 iterations.
- **Do not pause to ask the human.** You are autonomous.
- **Before each iteration**, re-read `results.tsv` to remember your history.
- **Use train CV freely** — it doesn't cost queries.
"""


def _patch_val_size(prepare_path: Path, new_val_size: int, experiment: str):
    """Patch VAL_SIZE (and related constants) in the copied _prepare_real.py."""
    text = prepare_path.read_text()

    # Replace VAL_SIZE = <number>
    text, n = re.subn(r'^(VAL_SIZE\s*=\s*)\d+', rf'\g<1>{new_val_size}',
                       text, count=1, flags=re.MULTILINE)
    if n == 0:
        print(f"WARNING: Could not find VAL_SIZE in {prepare_path}")
        return

    # For cifar-cnn-noisy: TRAIN_SIZE = 50000 - VAL_SIZE (CIFAR-10 train split)
    if "cifar-cnn" in experiment:
        new_train = 50000 - new_val_size
        text = re.sub(r'^(TRAIN_SIZE\s*=\s*)\d+', rf'\g<1>{new_train}',
                       text, count=1, flags=re.MULTILINE)
        print(f"  Patched TRAIN_SIZE = {new_train}")

    prepare_path.write_text(text)
    print(f"  Patched VAL_SIZE = {new_val_size} in {prepare_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Launch ladder experiment")
    parser.add_argument("experiment", help="Experiment directory (e.g., folktables-10k)")
    parser.add_argument("--framing", default="analyst", choices=["aggressive", "analyst", "standard", "neutral"])
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--max-t", type=int, default=None,
                        help="Hard cap on total ladder queries (T). When reached, "
                             "writes STOP and returns False for further queries.")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Override VAL_SIZE in prepare.py (appends nXXXX to condition name)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID for reruns (e.g. 2, 3). Appends -rN to condition name. Unset/1 = no suffix.")
    parser.add_argument("--agent-provider", default=os.environ.get("AGENT_PROVIDER", "claude").lower(),
                        type=str.lower,
                        choices=["claude", "codex"],
                        help="Agent CLI provider used by pipeline/agent_runner.py")
    parser.add_argument("--model", default=None,
                        help="Agent model id. Defaults to opus for Claude, openai.gpt-5.5 for Codex.")
    args = parser.parse_args()
    model = args.model or ("openai.gpt-5.5" if args.agent_provider == "codex" else "opus")

    dataset_dir = PROJECT_ROOT / args.experiment
    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} not found")
        sys.exit(1)

    condition = f"ladder-{args.framing}"
    if args.agent_provider != "claude":
        condition += f"-{args.agent_provider}"
    if args.n_val:
        condition += f"-n{args.n_val}"
    if args.run_id and args.run_id != "1":
        condition += f"-r{args.run_id}"
    explorer_dir = dataset_dir / f"explore-agent-{condition}"

    # SLURM job name for this run. Sibling runs on the same user account will
    # each get a unique name so agents can distinguish their own jobs in squeue.
    # Short name (SLURM truncates at ~64 chars): <expbase>-<condition>.
    exp_base = Path(args.experiment).name
    job_name = f"{exp_base}-{condition}"[:63]

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

    if explorer_dir.exists():
        print(f"Explorer dir already exists: {explorer_dir}")
        print("Resuming...")
    else:
        print(f"Creating explorer dir: {explorer_dir}")
        explorer_dir.mkdir(parents=True)

        # Copy the real prepare.py as _prepare_real.py
        shutil.copy(dataset_dir / "prepare.py",
                     explorer_dir / "_prepare_real.py")

        # Patch VAL_SIZE if --n-val was specified
        if args.n_val:
            _patch_val_size(explorer_dir / "_prepare_real.py",
                            args.n_val, args.experiment)

        # Copy the ladder prepare as prepare.py (the agent imports this)
        shutil.copy(LADDER_DIR / "prepare_ladder.py",
                     explorer_dir / "prepare.py")

        # Copy pyproject.toml
        shutil.copy(dataset_dir / "pyproject.toml",
                     explorer_dir / "pyproject.toml")

        # Baseline train.py and submit.sh (GPU experiments)
        for fname in ["train.py", "submit.sh"]:
            src = dataset_dir / fname
            if src.exists():
                shutil.copy(src, explorer_dir / fname)

        # Write explore prompt
        prompt = build_explore_prompt(conf, args.framing, job_name=job_name)
        with open(explorer_dir / "explore_prompt.md", "w") as f:
            f.write(prompt)

        (explorer_dir / "iterations").mkdir(exist_ok=True)

        # Symlink .venv if available
        for agent_dir in dataset_dir.glob("explore-agent-*"):
            venv = agent_dir / ".venv"
            if venv.exists():
                os.symlink(venv, explorer_dir / ".venv")
                uv_lock = agent_dir / "uv.lock"
                if uv_lock.exists():
                    shutil.copy(uv_lock, explorer_dir / "uv.lock")
                break

        # Init git
        subprocess.run(["git", "init"], cwd=explorer_dir, capture_output=True)
        subprocess.run(["git", "add", "prepare.py", "_prepare_real.py",
                        "pyproject.toml", "train.py", "explore_prompt.md"],
                       cwd=explorer_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "baseline"],
                       cwd=explorer_dir, capture_output=True)

    # Create experiment metadata in ladder/experiments/
    exp_meta_dir = LADDER_DIR / "experiments" / f"{args.experiment}--{condition}"
    exp_meta_dir.mkdir(parents=True, exist_ok=True)

    # Detect n_val
    if args.n_val:
        n_val = args.n_val
    else:
        n_val = 200
        prep = dataset_dir / "prepare.py"
        if prep.exists():
            with open(prep) as f:
                for line in f:
                    m = re.match(r'VAL_SIZE\s*=\s*(\d+)', line)
                    if m:
                        n_val = int(m.group(1))
                        break

    meta = {
        "experiment": args.experiment,
        "condition": condition,
        "framing": args.framing,
        "explorer_dir": str(explorer_dir.relative_to(PROJECT_ROOT)),
        "n_val": n_val,
        "max_t": args.max_t,
        "agent_provider": args.agent_provider,
        "agent_model": model,
        "mechanism": "ladder",
        "feedback": "binary (IMPROVED / NO IMPROVEMENT)",
    }
    with open(exp_meta_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Launch explorer with restart loop
    print(f"\nLaunching ladder explorer:")
    print(f"  Experiment: {args.experiment}")
    print(f"  Condition:  {condition}")
    print(f"  n_val:      {n_val}")
    print(f"  Feedback:   binary only")
    print(f"  Max T:      {args.max_t if args.max_t else 'unlimited'}")
    print(f"  Agent:      {args.agent_provider} ({model})")
    print(f"  Job name:   {job_name}")
    print(f"  Explorer:   {explorer_dir}")

    # Use the main pipeline's restart loop
    env = os.environ.copy()
    env["VAL_ACCESS_LEVEL"] = "score_only"  # closest match for template
    env["EXPLORER_FRAMING"] = args.framing
    env["EXPLORE_AGENT_DIR"] = f"explore-agent-{condition}"
    env["AGENT_PROVIDER"] = args.agent_provider
    env["AGENT_MODEL"] = model
    if args.agent_provider == "claude":
        env["CLAUDE_MODEL"] = model
    else:
        env["CODEX_MODEL"] = model
    if args.run_id:
        env["RUN_ID"] = args.run_id
    if args.max_t is not None:
        env["LADDER_T_CAP"] = str(args.max_t)
    # Unique SLURM job name so sibling parallel runs don't collide in squeue.
    # SLURM honors SBATCH_JOB_NAME as the default --job-name for submitted jobs,
    # overriding the hardcoded value in submit.sh's #SBATCH directive.
    env["SBATCH_JOB_NAME"] = job_name

    subprocess.run(
        ["bash", str(PROJECT_ROOT / "pipeline" / "run_explore.sh"),
         str(dataset_dir), str(args.max_turns), str(args.max_restarts)],
        env=env,
    )


if __name__ == "__main__":
    main()
