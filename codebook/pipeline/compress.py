#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Codebook compress: encode an explorer strategy into codebook selections.

Reads the explorer's best train.py at a given checkpoint and uses an LM agent
to encode it as structured codebook selections. The codelength in bits is
computed from the selections under the performance-weighted prior.

Usage:
    python codebook/run_codebook_compress.py folktables-5k <checkpoint> \
        --codebook codebook/folktables-perf.json

    # Batch: all improvement checkpoints
    python codebook/run_codebook_compress.py folktables-5k \
        --checkpoints abc1234 def5678 ... \
        --codebook codebook/folktables-perf.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codebook"))

from schema import compute_sparse_codelength, mdl_bound


def build_compress_prompt(codebook: dict, train_py: str, results_tsv: str,
                          iteration_logs: str, experiment_conf: dict) -> str:
    """Build the prompt for the codebook compression agent."""

    # Format codebook menus for the prompt
    menu_descriptions = []
    for m in codebook["menus"]:
        pi = m["selection_probability"]
        if m["type"] == "categorical":
            opts = ", ".join(f'"{k}" ({v:.0%})' for k, v in m["options"].items())
            menu_descriptions.append(
                f'  - **{m["name"]}** (categorical, π={pi:.3f}): {opts}')
        elif m["type"] == "integer":
            menu_descriptions.append(
                f'  - **{m["name"]}** (integer [{m["min"]}, {m["max"]}], π={pi:.3f})')
        elif m["type"] == "continuous":
            menu_descriptions.append(
                f'  - **{m["name"]}** (continuous [{m["min"]}, {m["max"]}], π={pi:.3f})')

    menus_text = "\n".join(menu_descriptions)

    return f"""# Codebook Compression Agent

You are encoding an ML strategy into a structured codebook. Read the explorer's
training script and results, then output a JSON file mapping codebook menu names
to the values used in the strategy.

## The task

{experiment_conf.get("TASK_DESC", "Binary classification task.")}

## Codebook menus

The codebook defines these menus (each with a selection probability π):

{menus_text}

Each menu is optional. Only specify menus that the strategy EXPLICITLY sets.
Leave menus unspecified if the strategy uses default values for that parameter.

For menus not in the codebook (off-menu choices), use the special format:
  "off_menu_<name>": "<description>"

## Explorer's best training script

```python
{train_py}
```

## Results history

```
{results_tsv}
```

## Iteration logs (summary)

{iteration_logs[:3000]}

## Your output

Write a single JSON file called `codebook_selections.json` with this format:

```json
{{
    "menu_name_1": "value",
    "menu_name_2": 42,
    "off_menu_threshold": "custom threshold 0.72 based on val optimization",
    "off_menu_feature_engineering": "target encoding on cols 1,3,4,5,6,9"
}}
```

Rules:
1. Only include menus that the strategy explicitly sets (not defaults)
2. For categorical menus, use the exact option name from the codebook
3. For integer/continuous menus, use the exact numeric value
4. For off-menu choices (things not in any codebook menu), prefix with "off_menu_"
5. Be precise — include ALL specific values (seeds, C, depth, etc.)
6. Do NOT include random seeds (they don't affect expected generalization)

Write ONLY `codebook_selections.json`. Nothing else.
"""


def run_compress(experiment: str, checkpoint: str, codebook_path: str,
                 condition: str = None):
    """Run codebook compression for one checkpoint."""
    exp_dir = PROJECT_ROOT / experiment
    codebook_dir = PROJECT_ROOT / "codebook"

    with open(codebook_path) as f:
        codebook = json.load(f)

    # Read experiment.conf
    conf = {}
    conf_path = exp_dir / "experiment.conf"
    if conf_path.exists():
        with open(conf_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    conf[key.strip()] = val.strip().strip('"')

    # Determine condition and explore dir
    if condition is None:
        val_access = conf.get("VAL_ACCESS_LEVEL", "full")
        framing = conf.get("EXPLORER_FRAMING", "aggressive")
        access_str = "scoreonly" if val_access == "score_only" else val_access
        condition = f"{access_str}-{framing}"

    explore_dir = exp_dir / f"explore-agent-{condition}"
    if not explore_dir.exists():
        print(f"ERROR: Explorer dir not found: {explore_dir}")
        return None

    # Resolve checkpoint to commit hash
    result = subprocess.run(
        ["git", "rev-parse", checkpoint],
        cwd=explore_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: Cannot resolve checkpoint '{checkpoint}' in {explore_dir}")
        return None
    commit = result.stdout.strip()
    short = commit[:7]

    # Extract train.py at checkpoint
    result = subprocess.run(
        ["git", "show", f"{commit}:train.py"],
        cwd=explore_dir, capture_output=True, text=True
    )
    train_py = result.stdout if result.returncode == 0 else "(could not extract)"

    # Get results.tsv up to this checkpoint
    results_tsv = ""
    results_path = explore_dir / "results.tsv"
    if results_path.exists():
        # Get all ancestor commits
        ancestor_result = subprocess.run(
            ["git", "log", "--reverse", "--format=%H", commit],
            cwd=explore_dir, capture_output=True, text=True
        )
        ancestors = set(h[:7] for h in ancestor_result.stdout.strip().split("\n") if h)
        with open(results_path) as f:
            for line in f:
                line_hash = line.split("\t")[0][:7]
                if line_hash in ancestors:
                    results_tsv += line

    # Get iteration logs (summarized)
    iteration_logs = ""
    iters_dir = explore_dir / "iterations"
    if iters_dir.exists():
        for f in sorted(iters_dir.glob("*.md")):
            if f.name == "FINAL.md":
                continue
            with open(f) as fh:
                content = fh.read()
            # Extract key lines
            for line in content.split("\n"):
                if any(k in line.lower() for k in ["strategy", "result", "lesson", "accuracy", "val_"]):
                    iteration_logs += line + "\n"

    # Create compress output directory
    compress_dir = codebook_dir / f"compress-{experiment}-{condition}-{short}"
    compress_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt
    prompt = build_compress_prompt(codebook, train_py, results_tsv,
                                    iteration_logs, conf)

    prompt_path = compress_dir / "compress_prompt.md"
    with open(prompt_path, "w") as f:
        f.write(prompt)

    # Save train.py for reference
    with open(compress_dir / "train.py", "w") as f:
        f.write(train_py)

    print(f"Compressing {experiment} checkpoint {short} (condition: {condition})")
    print(f"  Output: {compress_dir}")

    # Run the compress agent
    subprocess.run(
        [
            "claude", "--dangerously-skip-permissions",
            "--max-turns", "10",
            "--model", "sonnet",
            "-p", prompt,
        ],
        cwd=compress_dir,
    )

    # Check output
    selections_path = compress_dir / "codebook_selections.json"
    if not selections_path.exists():
        print(f"  WARNING: codebook_selections.json not found")
        return None

    with open(selections_path) as f:
        selections = json.load(f)

    # Separate on-menu and off-menu selections
    menu_names = {m["name"] for m in codebook["menus"]}
    on_menu = {k: v for k, v in selections.items()
               if k in menu_names and not k.startswith("off_menu_")}
    off_menu = {k: v for k, v in selections.items()
                if k not in menu_names or k.startswith("off_menu_")}

    # Compute codelength
    escape_bits = codebook.get("escape_penalty_bits", 12.0)
    bits, breakdown = compute_sparse_codelength(codebook, on_menu)

    # Add escape penalties for off-menu items
    for k, v in off_menu.items():
        bits += escape_bits
        breakdown[k] = {"escape": True, "total": escape_bits}

    # Save results
    result = {
        "experiment": experiment,
        "condition": condition,
        "checkpoint": short,
        "commit": commit,
        "selections": selections,
        "on_menu": on_menu,
        "off_menu": off_menu,
        "codelength_bits": round(bits, 2),
        "breakdown": breakdown,
    }

    with open(compress_dir / "codebook_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Codelength: {bits:.1f} bits")
    print(f"  On-menu: {len(on_menu)}, Off-menu: {len(off_menu)}")
    for k, v in on_menu.items():
        print(f"    {k}: {v}")
    for k, v in off_menu.items():
        print(f"    {k}: {v} [ESCAPE +{escape_bits} bits]")

    return result


def main():
    parser = argparse.ArgumentParser(description="Codebook compress")
    parser.add_argument("experiment", help="Experiment directory name")
    parser.add_argument("checkpoint", nargs="?", help="Single checkpoint hash")
    parser.add_argument("--checkpoints", nargs="+", help="Multiple checkpoint hashes")
    parser.add_argument("--codebook", required=True, help="Codebook JSON path")
    parser.add_argument("--condition", help="Override condition (e.g., full-aggressive)")
    args = parser.parse_args()

    checkpoints = args.checkpoints or ([args.checkpoint] if args.checkpoint else [])
    if not checkpoints:
        parser.error("Provide a checkpoint or --checkpoints")

    results = []
    for ckpt in checkpoints:
        r = run_compress(args.experiment, ckpt, args.codebook, args.condition)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"Compressed {len(results)} checkpoints")
        for r in results:
            print(f"  {r['checkpoint']}: {r['codelength_bits']:.1f} bits "
                  f"({len(r['on_menu'])} on-menu, {len(r['off_menu'])} off-menu)")


if __name__ == "__main__":
    main()
