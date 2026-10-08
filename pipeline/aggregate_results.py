#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Aggregate compression certificate results across experiments.

Usage:
    python aggregate_results.py <experiment_dir> [experiment_dir2 ...]
    python aggregate_results.py --all <parent_dir>

Output: CSV to stdout with columns:
    experiment, checkpoint, explorer_val, reproducer_val, holdout, audit, gap
"""

import argparse
import os
import re
import sys
from pathlib import Path


def parse_results_tsv(path):
    """Parse results.tsv, return list of (commit, val, desc)."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().strip().splitlines():
        parts = re.split(r'\t|\\t', line.strip())
        if len(parts) >= 2:
            try:
                val = float(parts[1])
                desc = parts[2] if len(parts) >= 3 else ""
                rows.append((parts[0], val, desc))
            except ValueError:
                continue
    return rows


def parse_holdout_tsv(path):
    """Parse holdout_results.tsv, return (val, holdout) from last line."""
    if not path.exists():
        return None, None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None, None
    parts = re.split(r'\t|\\t', lines[-1].strip())
    if len(parts) >= 3:
        try:
            return float(parts[1]), float(parts[2])
        except ValueError:
            pass
    return None, None


def run_audit(pipeline_dir, reproduce_dir):
    """Run audit.py and return verdict."""
    import subprocess
    audit_script = pipeline_dir / "audit.py"
    if not audit_script.exists():
        return "N/A"
    result = subprocess.run(
        [sys.executable, str(audit_script), str(reproduce_dir)],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    if "CLEAN" in output:
        return "CLEAN"
    elif "FAIL" in output:
        return "FAIL"
    return output or "ERROR"


def process_experiment(exp_dir, pipeline_dir):
    """Process one experiment directory, return list of result dicts."""
    exp_dir = Path(exp_dir)
    results = []

    # Get experiment name from config
    conf = exp_dir / "experiment.conf"
    exp_name = exp_dir.name
    if conf.exists():
        for line in conf.read_text().splitlines():
            if line.startswith("EXPERIMENT_NAME="):
                exp_name = line.split("=", 1)[1].strip().strip('"').strip("'")

    # Find all reproduce-* directories (new: reproduce-{condition}-{label}, old: reproduce-{label})
    reproduce_dirs = sorted(d for d in exp_dir.iterdir()
                            if d.is_dir() and d.name.startswith("reproduce-"))
    if not reproduce_dirs:
        return results

    # Get explorer results from compress dirs
    explore_results = {}
    for compress_dir in sorted(d for d in exp_dir.iterdir()
                               if d.is_dir() and d.name.startswith("compress-")):
        # Extract the full suffix after "compress-"
        suffix = compress_dir.name[len("compress-"):]
        tsv = compress_dir / "results.tsv"
        rows = parse_results_tsv(tsv)
        if rows:
            explore_results[suffix] = rows[-1][1]

    for rdir in reproduce_dirs:
        # Extract the full suffix after "reproduce-"
        suffix = rdir.name[len("reproduce-"):]

        # Reproducer val from results.tsv
        reproduce_rows = parse_results_tsv(rdir / "results.tsv")
        if not reproduce_rows:
            continue
        reproduce_val = reproduce_rows[-1][1]
        reproduce_desc = reproduce_rows[-1][2]

        # Holdout
        _, holdout = parse_holdout_tsv(rdir / "holdout_results.tsv")

        # Explorer val for this checkpoint
        explorer_val = explore_results.get(suffix)

        # Audit
        audit = run_audit(pipeline_dir, rdir)

        # Gap
        gap = None
        if explorer_val is not None and reproduce_val is not None:
            gap = explorer_val - reproduce_val

        results.append({
            "experiment": exp_name,
            "checkpoint": suffix,
            "explorer_val": explorer_val,
            "reproducer_val": reproduce_val,
            "holdout": holdout,
            "audit": audit,
            "gap": gap,
            "description": reproduce_desc,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate compression certificate results")
    parser.add_argument("dirs", nargs="*", help="Experiment directories")
    parser.add_argument("--all", metavar="PARENT", help="Process all experiment dirs under PARENT")
    parser.add_argument("--format", choices=["csv", "markdown"], default="csv")
    args = parser.parse_args()

    if args.all:
        parent = Path(args.all)
        exp_dirs = [d for d in sorted(parent.iterdir())
                    if d.is_dir() and (d / "experiment.conf").exists()]
    elif args.dirs:
        exp_dirs = [Path(d) for d in args.dirs]
    else:
        parser.error("Provide experiment directories or --all <parent>")

    # Find pipeline dir
    script_dir = Path(__file__).parent
    pipeline_dir = script_dir

    all_results = []
    for exp_dir in exp_dirs:
        all_results.extend(process_experiment(exp_dir, pipeline_dir))

    if not all_results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Output
    cols = ["experiment", "checkpoint", "explorer_val", "reproducer_val", "holdout", "audit", "gap"]

    if args.format == "csv":
        print(",".join(cols))
        for r in all_results:
            vals = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            print(",".join(vals))
    else:
        # Markdown table
        print("| " + " | ".join(cols) + " |")
        print("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in all_results:
            vals = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    vals.append("-")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            print("| " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
