#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: monitor_runs.sh [experiments_root]
#
# One-line status per explorer directory under each experiment.
# Use this to sanity-check a wave of runs without opening 12 tmux panes.

set -u
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

printf "=== explorers under %s (%s) ===\n" "$ROOT" "$(date -u +%H:%M:%SZ)"
printf "%-36s %-26s %6s %8s %8s %s\n" "DATASET" "CONDITION" "ITERS" "LASTIT" "PROCESS" "NOTES"

shopt -s nullglob
for dataset_dir in "$ROOT"/*/; do
    dataset=$(basename "$dataset_dir")
    case "$dataset" in pipeline|plots|__pycache__) continue;; esac
    [ -f "$dataset_dir/experiment.conf" ] || continue

    for agent_dir in "$dataset_dir"explore-agent-*/; do
        [ -d "$agent_dir" ] || continue
        agent_name=$(basename "$agent_dir")
        condition="${agent_name#explore-agent-}"

        iters=0
        [ -f "$agent_dir/results.tsv" ] && iters=$(wc -l < "$agent_dir/results.tsv" 2>/dev/null || echo 0)

        lastit="-"
        if [ -f "$agent_dir/results.tsv" ] && [ "$iters" -gt 0 ]; then
            lastit=$(stat -c %Y "$agent_dir/results.tsv" 2>/dev/null)
            now=$(date +%s)
            age=$(( now - lastit ))
            if [ "$age" -lt 60 ]; then lastit="${age}s"
            elif [ "$age" -lt 3600 ]; then lastit="$((age/60))m"
            else lastit="$((age/3600))h"
            fi
        fi

        # Is there a claude process anchored to this dir? Check agent.log mtime.
        proc="idle"
        if [ -f "$agent_dir/agent.log" ]; then
            log_age=$(( $(date +%s) - $(stat -c %Y "$agent_dir/agent.log" 2>/dev/null || echo 0) ))
            [ "$log_age" -lt 300 ] && proc="active(${log_age}s)"
        fi

        notes=""
        [ -f "$agent_dir/STOP" ] && notes="STOP "
        [ -f "$agent_dir/iterations/FINAL.md" ] && notes="${notes}FINAL "
        [ -f "$agent_dir/run_meta.json" ] || notes="${notes}no-meta "

        printf "%-36s %-26s %6s %8s %8s %s\n" \
            "$dataset" "$condition" "$iters" "$lastit" "$proc" "$notes"
    done
done

echo ""
echo "=== squeue -u $USER (summary) ==="
jobs=$(squeue -h -u "$USER" 2>/dev/null | wc -l)
echo "  total jobs in queue: $jobs"
if [ "$jobs" -gt 0 ]; then
    squeue -h -u "$USER" -o "%j" | sort | uniq -c | sort -rn | head -10
fi

echo ""
echo "=== orphan explore/reproduce processes ==="
ps -o pid,etime,cmd -u "$USER" 2>/dev/null \
    | grep -E 'run_explore\.sh|run_reproduce\.sh|run_all_checkpoints\.sh|run_ladder_experiment' \
    | grep -v grep \
    | head -20 || echo "  (none)"
