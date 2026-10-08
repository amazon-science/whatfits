#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Iterative codebook compression with bit budget enforcement.

Multi-turn process:
1. Agent reads explorer's train.py and outputs codebook selections JSON
2. We compute codelength. If over budget, show the cost breakdown and ask to drop items.
3. Repeat up to MAX_TURNS times. On final turn, greedily drop most expensive items.

Usage:
    python codebook/pipeline/compress_iterative.py codebook/experiments/folktables-5k--full-aggressive/ \
        --checkpoints 4e9a8b4 17857bf ...

    # Config (n_val, beta, bit_budget, codebook path, explorer dir) read from config.json
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
CODEBOOK_ROOT = PIPELINE_DIR.parent
PROJECT_ROOT = CODEBOOK_ROOT.parent
sys.path.insert(0, str(CODEBOOK_ROOT))

from schema import compute_sparse_codelength, mdl_bound

MAX_TURNS = 8  # more turns than token compression since bits are harder to estimate


def build_initial_prompt(codebook: dict, train_py: str, results_tsv: str,
                         experiment_conf: dict, bit_budget: float) -> str:
    """Build the initial compression prompt."""
    menu_lines = []
    for m in codebook["menus"]:
        pi = m["selection_probability"]
        select_cost = -math.log2(max(pi, 1e-6))
        skip_cost = -math.log2(max(1 - pi, 1e-6))

        if m["type"] == "categorical":
            opts = ", ".join(f'"{k}"' for k in m["options"])
            menu_lines.append(
                f'  - **{m["name"]}** (categorical, π={pi:.3f}, '
                f'select costs {select_cost:.1f} bits, skip costs {skip_cost:.1f} bits): {opts}')
        elif m["type"] == "integer":
            n_vals = m["max"] - m["min"] + 1
            val_cost = math.log2(n_vals) if n_vals > 1 else 0
            menu_lines.append(
                f'  - **{m["name"]}** (integer [{m["min"]}, {m["max"]}], π={pi:.3f}, '
                f'select costs {select_cost:.1f}+{val_cost:.1f} bits, skip costs {skip_cost:.1f} bits)')
        elif m["type"] == "continuous":
            menu_lines.append(
                f'  - **{m["name"]}** (continuous [{m["min"]}, {m["max"]}], π={pi:.3f}, '
                f'select costs {select_cost:.1f}+value bits, skip costs {skip_cost:.1f} bits)')

    menus_text = "\n".join(menu_lines)

    # Compute skip-all baseline cost
    skip_all_bits = 0
    for m in codebook["menus"]:
        pi = m["selection_probability"]
        pi = max(1e-6, min(1 - 1e-6, pi))
        skip_all_bits += -math.log2(1 - pi)

    return f"""# Codebook Compression Agent (Bit-Budgeted)

You must encode the explorer's strategy into codebook selections within a strict
**{bit_budget:.0f}-bit budget**.

## Budget mechanics

The codebook uses a spike-and-slab prior. Every menu has a cost:
- **Selecting** menu i costs: -log2(π_i) + value_bits
- **Skipping** menu i costs: -log2(1 - π_i)

Skipping ALL menus costs {skip_all_bits:.1f} bits (the baseline). Each menu you
specify adds its selection cost minus the skip cost you avoid.

Off-menu items (not in the codebook) cost 12 bits each — very expensive!

Your total codelength must be ≤ **{bit_budget:.0f} bits**.

## Codebook menus (with costs)

{menus_text}

Off-menu items: 12 bits each (escape penalty).

## Explorer's training script

```python
{train_py}
```

## Results history

```
{results_tsv}
```

## Task

{experiment_conf.get("TASK_DESC", "Binary classification task.")}

## Your output

Write `codebook_selections.json` with ONLY the most important selections that
fit within {bit_budget:.0f} bits. Format:

```json
{{
    "model": "GradientBoosting",
    "max_depth": 5
}}
```

**Strategy for staying within budget:**
1. Start with the highest-impact, lowest-cost selections (model family is cheap)
2. Add selections in order of importance until approaching the budget
3. Do NOT include off-menu items unless absolutely essential (12 bits each!)
4. Leave out minor hyperparameters — the reproducer will use defaults

You MUST write the file `codebook_selections.json` with your selections.
Use the Write tool or create it with a bash command. Do not explain — just write the JSON file.
"""


def build_revision_prompt(selections: dict, bits: float, breakdown: dict,
                          bit_budget: float, codebook: dict) -> str:
    """Build a revision prompt showing current cost and asking for cuts."""
    cost_lines = []
    for name, info in sorted(breakdown.items(), key=lambda x: -x[1].get("total", 0)):
        total = info.get("total", 0)
        if "escape" in info:
            cost_lines.append(f"  {name}: {total:.1f} bits [OFF-MENU ESCAPE]")
        elif "skip" in info:
            cost_lines.append(f"  {name}: {total:.1f} bits (skipped)")
        else:
            sel = info.get("select", 0)
            val = info.get("value", 0)
            cost_lines.append(f"  {name}: {total:.1f} bits (select={sel:.1f} + value={val:.1f})")

    cost_text = "\n".join(cost_lines)
    over_by = bits - bit_budget

    return f"""Your codebook selections use **{bits:.1f} bits**, which is **{over_by:.1f} bits over** the {bit_budget:.0f}-bit budget.

Current cost breakdown:
{cost_text}

**Total: {bits:.1f} bits (budget: {bit_budget:.0f} bits)**

You need to cut at least {over_by:.1f} bits. Options:
- Remove expensive selections (the reproducer will use defaults for those)
- Remove off-menu items (12 bits each — removing one saves a lot)
- Keep only the selections that matter most for accuracy

Rewrite `codebook_selections.json` to fit within {bit_budget:.0f} bits.
Write ONLY the JSON file. Nothing else."""


def greedy_truncate(selections: dict, breakdown: dict, codebook: dict,
                    bit_budget: float) -> dict:
    """Greedily drop most expensive items until within budget."""
    menu_names = {m["name"] for m in codebook["menus"]}

    # Compute current total
    items = []
    for name, info in breakdown.items():
        if "skip" in info:
            continue  # skipped menus don't contribute to selections
        items.append((name, info.get("total", 0)))

    # Sort by cost descending, drop until within budget
    items.sort(key=lambda x: -x[1])

    current_bits = sum(cost for _, cost in items)
    # Add skip costs for all menus
    skip_bits = 0
    for m in codebook["menus"]:
        pi = m["selection_probability"]
        pi = max(1e-6, min(1 - 1e-6, pi))
        if m["name"] not in selections:
            skip_bits += -math.log2(1 - pi)

    kept = dict(selections)
    for name, cost in items:
        if current_bits + skip_bits <= bit_budget:
            break
        if name in kept:
            del kept[name]
            current_bits -= cost
            # Add back the skip cost for this menu
            for m in codebook["menus"]:
                if m["name"] == name:
                    pi = m["selection_probability"]
                    pi = max(1e-6, min(1 - 1e-6, pi))
                    skip_bits += -math.log2(1 - pi)
                    break

    return kept


def run_compress_one(exp_dir: Path, checkpoint: str, codebook: dict,
                     bit_budget: float, config: dict):
    """Run iterative codebook compression for one checkpoint."""
    explore_dir = PROJECT_ROOT / config["explorer_dir"]
    short = checkpoint[:7]

    # Read experiment.conf for task description
    conf = {}
    dataset_dir = PROJECT_ROOT / config["experiment"]
    conf_path = dataset_dir / "experiment.conf"
    if conf_path.exists():
        with open(conf_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    conf[key.strip()] = val.strip().strip('"')

    # Resolve commit
    result = subprocess.run(
        ["git", "rev-parse", checkpoint],
        cwd=explore_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: Cannot resolve '{checkpoint}' in {explore_dir}")
        return None
    commit = result.stdout.strip()

    # Extract train.py
    result = subprocess.run(
        ["git", "show", f"{commit}:train.py"],
        cwd=explore_dir, capture_output=True, text=True
    )
    train_py = result.stdout if result.returncode == 0 else "(not found)"

    # Get results.tsv up to checkpoint
    results_tsv = ""
    results_path = explore_dir / "results.tsv"
    if results_path.exists():
        ancestor_result = subprocess.run(
            ["git", "log", "--reverse", "--format=%H", commit],
            cwd=explore_dir, capture_output=True, text=True
        )
        ancestors = set(h[:7] for h in ancestor_result.stdout.strip().split("\n") if h)
        with open(results_path) as f:
            for line in f:
                if line.split("\t")[0][:7] in ancestors:
                    results_tsv += line

    # Create compress directory under experiment checkpoints
    ckpt_dir = exp_dir / "checkpoints" / short
    compress_dir = ckpt_dir / "compress"
    if compress_dir.exists():
        shutil.rmtree(compress_dir)
    compress_dir.mkdir(parents=True)

    # Save train.py for reference
    with open(compress_dir / "train.py", "w") as f:
        f.write(train_py)

    log = {"budget_bits": bit_budget, "checkpoint": short, "turns": []}
    selections = {}
    bits = 0.0
    breakdown = {}

    print(f"\n{'='*60}")
    print(f"Compressing {short} (budget: {bit_budget} bits)")
    print(f"{'='*60}")

    for turn in range(1, MAX_TURNS + 1):
        print(f"  Turn {turn}/{MAX_TURNS}:", end=" ")

        if turn == 1 or not selections:
            prompt = build_initial_prompt(codebook, train_py, results_tsv, conf, bit_budget)
        else:
            prompt = build_revision_prompt(selections, bits, breakdown, bit_budget, codebook)

        # Save prompt
        with open(compress_dir / f"prompt_turn{turn}.md", "w") as f:
            f.write(prompt)

        # Run agent
        t0 = time.time()
        subprocess.run(
            ["claude", "--dangerously-skip-permissions",
             "--max-turns", "10", "--model", "sonnet",
             "-p", prompt],
            cwd=compress_dir,
            capture_output=True, timeout=120,
        )
        elapsed = time.time() - t0

        # Read selections
        sel_path = compress_dir / "codebook_selections.json"
        if not sel_path.exists():
            print(f"no output (retrying)")
            log["turns"].append({"turn": turn, "status": "no_output", "elapsed": round(elapsed, 1)})
            continue

        with open(sel_path) as f:
            try:
                selections = json.load(f)
            except json.JSONDecodeError:
                print(f"invalid JSON (retrying)")
                log["turns"].append({"turn": turn, "status": "bad_json", "elapsed": round(elapsed, 1)})
                continue

        # Filter out null/None values and random seeds
        selections = {k: v for k, v in selections.items()
                      if v is not None and k != "random_seed"}

        # Compute codelength
        menu_names = {m["name"] for m in codebook["menus"]}
        on_menu = {k: v for k, v in selections.items()
                   if k in menu_names and not k.startswith("off_menu_")}
        off_menu = {k: v for k, v in selections.items()
                    if k not in menu_names or k.startswith("off_menu_")}

        bits, breakdown = compute_sparse_codelength(codebook, on_menu)
        escape_bits = codebook.get("escape_penalty_bits", 12.0)
        for k in off_menu:
            bits += escape_bits
            breakdown[k] = {"escape": True, "total": escape_bits}

        log["turns"].append({
            "turn": turn, "bits": round(bits, 2), "selections": selections,
            "status": "pass" if bits <= bit_budget else "over",
            "elapsed": round(elapsed, 1),
        })

        if bits <= bit_budget:
            print(f"{bits:.1f}/{bit_budget:.0f} bits ✓")
            break
        else:
            print(f"{bits:.1f}/{bit_budget:.0f} bits ✗ (over by {bits - bit_budget:.1f})")

            if turn == MAX_TURNS:
                # Greedy truncation
                print(f"  Max turns reached. Greedy truncation...")
                selections = greedy_truncate(selections, breakdown, codebook, bit_budget)
                # Recompute
                on_menu = {k: v for k, v in selections.items() if k in menu_names}
                bits, breakdown = compute_sparse_codelength(codebook, on_menu)
                print(f"  After truncation: {bits:.1f} bits")
                log["turns"].append({
                    "turn": "truncated", "bits": round(bits, 2),
                    "selections": selections, "status": "truncated",
                })
                # Write truncated selections
                with open(sel_path, "w") as f:
                    json.dump(selections, f, indent=2)

    # Save final result
    final_selections = selections
    final_on_menu = {k: v for k, v in final_selections.items()
                     if k in menu_names and not k.startswith("off_menu_")}
    final_bits, final_breakdown = compute_sparse_codelength(codebook, final_on_menu)
    # Count any remaining off-menu
    for k in final_selections:
        if k not in menu_names or k.startswith("off_menu_"):
            final_bits += escape_bits

    result = {
        "experiment": config["experiment"],
        "condition": config["condition"],
        "checkpoint": short,
        "commit": commit,
        "bit_budget": bit_budget,
        "codelength_bits": round(final_bits, 2),
        "selections": final_selections,
        "breakdown": final_breakdown,
    }
    with open(compress_dir / "codebook_result.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(compress_dir / "compress_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"  Final: {final_bits:.1f} bits, selections: {final_selections}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Iterative codebook compression")
    parser.add_argument("experiment_dir", help="Experiment dir (e.g., codebook/experiments/folktables-5k--full-aggressive/)")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--bit-budget", type=float, default=None,
                        help="Override bit budget from config.json")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    with open(exp_dir / "config.json") as f:
        config = json.load(f)
    with open(exp_dir / "codebook.json") as f:
        codebook = json.load(f)

    bit_budget = args.bit_budget or config.get("bit_budget", 16)

    results = []
    for ckpt in args.checkpoints:
        r = run_compress_one(exp_dir, ckpt, codebook, bit_budget, config)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} checkpoints at {bit_budget}-bit budget")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['checkpoint']}: {r['codelength_bits']:.1f} bits — {r['selections']}")


if __name__ == "__main__":
    main()
