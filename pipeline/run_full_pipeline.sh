#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: run_full_pipeline.sh <experiment_dir> [MAX_TURNS] [CHECKPOINT ...]
#
# Runs the documented score-only compression arm:
#   explore -> select improvement checkpoints -> compress/reproduce/audit -> summary
#
# Environment knobs:
#   MAX_RESTARTS    explorer restart count, default 10
#   MIN_REL_DELTA   relative improvement threshold for checkpoint selection, default 0.005
#   FORCE_EXPLORE   set to 1 to rerun/continue exploration even if results.tsv exists

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"
shift 2>/dev/null || true

MAX_TURNS="${1:-2000}"
if [ $# -gt 0 ]; then
    shift
fi

MAX_RESTARTS="${MAX_RESTARTS:-10}"
MIN_REL_DELTA="${MIN_REL_DELTA:-0.005}"
EXPLORE_DIR="$(get_explore_agent_dir)"

echo "=========================================="
echo "  ${EXPERIMENT_NAME} - FULL COMPRESSION ARM"
echo "  Experiment: $EXPERIMENT_DIR"
echo "  Explore dir: $EXPLORE_DIR"
echo "  Max turns: $MAX_TURNS"
echo "=========================================="

if [ "${FORCE_EXPLORE:-0}" = "1" ] || [ ! -f "$EXPLORE_DIR/results.tsv" ]; then
    echo ""
    echo "=== EXPLORE ==="
    bash "$PIPELINE_DIR/run_explore.sh" "$EXPERIMENT_DIR" "$MAX_TURNS" "$MAX_RESTARTS"
else
    echo ""
    echo "=== EXPLORE ==="
    echo "Skipping: existing results found at $EXPLORE_DIR/results.tsv"
    echo "Set FORCE_EXPLORE=1 to continue/relaunch the explorer."
fi

CHECKPOINTS=("$@")
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo ""
    echo "=== SELECT CHECKPOINTS ==="
    if [ ! -f "$EXPLORE_DIR/results.tsv" ]; then
        echo "ERROR: no results.tsv found after exploration: $EXPLORE_DIR/results.tsv" >&2
        exit 1
    fi
    CHECKPOINT_TEXT=$(
        python "$PIPELINE_DIR/select_improvement_checkpoints.py" \
            "$EXPLORE_DIR/results.tsv" "${METRIC_DIRECTION:-higher}" "$MIN_REL_DELTA" \
            2>/dev/null || true
    )
    if [ -n "$CHECKPOINT_TEXT" ]; then
        read -r -a CHECKPOINTS <<< "$CHECKPOINT_TEXT"
    else
        CHECKPOINTS=("latest")
    fi
fi

echo "Checkpoints: ${CHECKPOINTS[*]}"

echo ""
echo "=== COMPRESS / REPRODUCE / AUDIT ==="
bash "$PIPELINE_DIR/run_all_checkpoints.sh" "$EXPERIMENT_DIR" "${CHECKPOINTS[@]}"

echo ""
echo "=== AGGREGATED RESULTS ==="
python "$PIPELINE_DIR/aggregate_results.py" "$EXPERIMENT_DIR" --format markdown || true

echo ""
echo "Full compression arm complete."
