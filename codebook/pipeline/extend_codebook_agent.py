#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Phase 0b: Extend the codebook with menus the agent didn't try.

After Phase 0a (sequential exploration), an agent reviews the configs it tried
and proposes additional codebook dimensions it *could have* explored — techniques
that are reasonable for the task but weren't tested due to budget constraints.

These get lower prior probabilities than empirically-tested menus, reflecting
that they're hypothetical rather than validated.

Usage:
    python codebook/pipeline/extend_codebook_agent.py codebook/experiments/folktables-10k--full-analyst/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
CODEBOOK_ROOT = PIPELINE_DIR.parent
PROJECT_ROOT = CODEBOOK_ROOT.parent


def build_prompt(configs: list[dict], codebook: dict, experiment_conf: dict) -> str:
    """Build prompt for the codebook extension agent."""

    task_desc = experiment_conf.get("TASK_DESC", "Binary classification task.")
    packages = experiment_conf.get("PACKAGES", "scikit-learn numpy")

    # Summarize what Phase 0 explored
    explored_menus = set()
    for c in configs:
        for k in c["specified_menus"]:
            if k != "random_seed":
                explored_menus.add(k)

    explored_summary = []
    for c in configs:
        menus = {k: v for k, v in c["specified_menus"].items() if k != "random_seed"}
        explored_summary.append(f"  iter {c.get('iter','?')}: CV={c['cv_accuracy']:.4f} — {json.dumps(menus)}")

    existing_menus = []
    for m in codebook["menus"]:
        existing_menus.append(f"  - {m['name']} ({m['type']}, π={m['selection_probability']:.3f})")

    return f"""# Extend Codebook with Unexplored Dimensions

You are reviewing a codebook built from a Phase 0 exploration on a machine learning task.
The codebook currently covers only what was explicitly tried during 12 iterations of
train-only cross-validation. Your job is to propose **additional codebook menus** for
techniques that a competent ML practitioner might try on this task but that weren't
explored in Phase 0.

## Task

{task_desc}

Available packages: {packages}

## What Phase 0 explored

{chr(10).join(explored_summary)}

## Current codebook menus

{chr(10).join(existing_menus)}

## Your job

Propose additional menus for techniques NOT currently in the codebook. Think about:

1. **Feature engineering**: target encoding, frequency encoding, binning, log transforms,
   interaction features, polynomial features — things that could help but weren't tried
2. **Ensemble methods**: voting, stacking, blending — combining multiple models
3. **Threshold tuning**: optimizing the classification threshold instead of using 0.5
4. **Class weighting**: sample weights, class_weight parameter
5. **Additional model families**: XGBoost, LightGBM, MLP, etc.
6. **Regularization variants**: different L1/L2 settings, early stopping criteria
7. **Any other technique** reasonable for tabular binary classification

For each proposed menu, you must assign:
- **selection_probability (π)**: how likely a random competent practitioner would use this
  technique on this task. Use LOWER values than the empirical menus (since these are
  hypothetical, not validated). Typical range: 0.05-0.30.
- **type and options**: same format as existing menus

## Output

Write `extended_menus.json` — a JSON array of new menu objects. Example:

```json
[
    {{
        "name": "target_encoding",
        "type": "categorical",
        "selection_probability": 0.15,
        "options": {{"OCCP": 0.4, "OCCP+COW": 0.3, "all_categorical": 0.2, "none": 0.1}}
    }},
    {{
        "name": "threshold",
        "type": "continuous",
        "selection_probability": 0.10,
        "min": 0.4,
        "max": 0.6,
        "scale": "linear"
    }},
    {{
        "name": "ensemble_method",
        "type": "categorical",
        "selection_probability": 0.08,
        "options": {{"voting_2models": 0.4, "voting_3plus": 0.3, "stacking": 0.2, "single": 0.1}}
    }}
]
```

Rules:
- Only propose menus for techniques NOT already in the codebook
- Assign π values that reflect genuine prior belief, not optimism
- Include 5-10 new menus covering the major unexplored dimensions
- Each menu should be self-contained and clearly defined

Write ONLY `extended_menus.json`. Nothing else.
"""


def main():
    parser = argparse.ArgumentParser(description="Extend codebook with unexplored dimensions")
    parser.add_argument("experiment_dir", help="Codebook experiment directory")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    experiment = config["experiment"]
    dataset_dir = PROJECT_ROOT / experiment

    # Load existing codebook
    with open(exp_dir / "codebook.json") as f:
        codebook = json.load(f)

    # Load Phase 0 configs
    configs = []
    configs_path = exp_dir / "build" / "configs.jsonl"
    with open(configs_path) as f:
        for line in f:
            if line.strip():
                configs.append(json.loads(line))

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

    prompt = build_prompt(configs, codebook, conf)

    # Run in the build dir
    build_dir = exp_dir / "build"
    with open(build_dir / "extend_prompt.md", "w") as f:
        f.write(prompt)

    print(f"Running codebook extension agent...")
    subprocess.run(
        ["claude", "--dangerously-skip-permissions",
         "--max-turns", "10", "--model", "sonnet",
         "-p", prompt],
        cwd=build_dir,
    )

    # Read extended menus
    extended_path = build_dir / "extended_menus.json"
    if not extended_path.exists():
        print("ERROR: extended_menus.json not written")
        sys.exit(1)

    with open(extended_path) as f:
        new_menus = json.load(f)

    print(f"\nProposed {len(new_menus)} new menus:")
    for m in new_menus:
        print(f"  - {m['name']} ({m['type']}, π={m.get('selection_probability', '?')})")

    # Merge into codebook
    existing_names = {m["name"] for m in codebook["menus"]}
    added = 0
    for m in new_menus:
        if m["name"] not in existing_names:
            codebook["menus"].append(m)
            existing_names.add(m["name"])
            added += 1
        else:
            print(f"  SKIP: {m['name']} already exists")

    # Sort by selection_probability descending
    codebook["menus"].sort(key=lambda m: -m.get("selection_probability", 0))
    codebook["description"] = codebook.get("description", "") + " + extended menus"

    # Save
    with open(exp_dir / "codebook.json", "w") as f:
        json.dump(codebook, f, indent=2)

    print(f"\nAdded {added} new menus to codebook ({len(codebook['menus'])} total)")
    print(f"Saved to {exp_dir / 'codebook.json'}")


if __name__ == "__main__":
    main()
