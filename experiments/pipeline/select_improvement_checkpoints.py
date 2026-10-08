#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Select checkpoints where the explorer improved on the best-so-far metric.

Usage: select_improvement_checkpoints.py <results.tsv> <direction> [min_rel_delta]
  direction: "higher" (accuracy) or "lower" (BPB/loss)
  min_rel_delta: minimum relative improvement to include (default 0.01 = 1%)
                 First checkpoint is always included regardless.

Outputs commit hashes (short, 7-char) of improvement checkpoints, one per line.

Logic:
  - Parse results.tsv: columns are commit\tmetric\tdescription
  - Handle literal \t (some agents write escaped tabs)
  - For each commit, take the best metric value (some commits have multiple entries)
  - Walk commits in order, keep only those that improve on the running best by >= min_rel_delta
"""
import sys
import re

def main():
    if len(sys.argv) < 3:
        print("Usage: select_improvement_checkpoints.py <results.tsv> <direction> [min_rel_delta]", file=sys.stderr)
        sys.exit(1)

    results_file = sys.argv[1]
    direction = sys.argv[2]  # "higher" or "lower"
    min_rel_delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    # Parse results.tsv
    entries = []  # list of (commit_short, metric_value)
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\t|\\t', line)
            if len(parts) < 2:
                continue
            commit = parts[0].strip()[:7]
            try:
                metric = float(parts[1].strip())
            except ValueError:
                continue
            if commit == "na" or commit == "":
                continue
            entries.append((commit, metric))

    if not entries:
        sys.exit(0)

    # Group by commit (preserve order of first appearance)
    from collections import OrderedDict
    commit_best = OrderedDict()
    for commit, metric in entries:
        if commit not in commit_best:
            commit_best[commit] = metric
        else:
            if direction == "higher":
                commit_best[commit] = max(commit_best[commit], metric)
            else:
                commit_best[commit] = min(commit_best[commit], metric)

    # Walk in order, keep improvements that exceed min_rel_delta
    best_so_far = None
    last_selected = None
    improvement_commits = []
    for commit, metric in commit_best.items():
        if best_so_far is None:
            # Always include the first checkpoint
            best_so_far = metric
            last_selected = metric
            improvement_commits.append(commit)
            print(f"{commit}", file=sys.stdout)
            print(f"  {commit}: {metric}", file=sys.stderr)
            continue

        # Check if this is an improvement at all
        if direction == "higher":
            is_improvement = metric > best_so_far + 1e-8
        else:
            is_improvement = metric < best_so_far - 1e-8

        if not is_improvement:
            continue

        best_so_far = metric

        # Check if relative improvement from last selected checkpoint is >= threshold
        if last_selected != 0:
            rel_delta = abs(metric - last_selected) / abs(last_selected)
        else:
            rel_delta = float('inf')

        if rel_delta >= min_rel_delta:
            last_selected = metric
            improvement_commits.append(commit)
            print(f"{commit}", file=sys.stdout)
            print(f"  {commit}: {metric} (rel_delta={rel_delta:.3f})", file=sys.stderr)

    # Always include the final best if not already included
    if improvement_commits and improvement_commits[-1] != list(commit_best.keys())[list(commit_best.values()).index(best_so_far)]:
        final_commit = [k for k, v in commit_best.items() if v == best_so_far][-1]
        if final_commit not in improvement_commits:
            improvement_commits.append(final_commit)
            print(f"{final_commit}", file=sys.stdout)
            print(f"  {final_commit}: {best_so_far} (final best)", file=sys.stderr)

    print(f"\n{len(improvement_commits)} checkpoints selected (min_rel_delta={min_rel_delta}) out of {len(commit_best)} total", file=sys.stderr)

if __name__ == "__main__":
    main()
