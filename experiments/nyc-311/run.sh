#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Convenience wrapper — copy or symlink this into an experiment directory as run.sh
#
# Usage:
#   bash run.sh explore              # run explorer
#   bash run.sh compress [CKPT]      # compress at checkpoint
#   bash run.sh reproduce [CKPT]     # reproduce from checkpoint
#   bash run.sh sweep [CKPT_LIST]    # run all checkpoints
#   bash run.sh compare [CKPT]       # compare results
#   bash run.sh audit [CKPT]         # audit reproducer
#   bash run.sh full [TURNS] [CKPT]  # full pipeline
#   bash run.sh stop                 # stop explorer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/../../pipeline" ]; then
    PIPELINE_DIR="$(cd "$SCRIPT_DIR/../../pipeline" && pwd)"
elif [ -d "$SCRIPT_DIR/../pipeline" ]; then
    PIPELINE_DIR="$(cd "$SCRIPT_DIR/../pipeline" && pwd)"
else
    echo "ERROR: Cannot find pipeline/ directory"
    exit 2
fi

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
    explore)    bash "$PIPELINE_DIR/run_explore.sh" "$SCRIPT_DIR" "$@" ;;
    compress)   bash "$PIPELINE_DIR/run_compress.sh" "$SCRIPT_DIR" "$@" ;;
    reproduce)  bash "$PIPELINE_DIR/run_reproduce.sh" "$SCRIPT_DIR" "$@" ;;
    sweep)      bash "$PIPELINE_DIR/run_all_checkpoints.sh" "$SCRIPT_DIR" "$@" ;;
    compare)    bash "$PIPELINE_DIR/compare_results.sh" "$SCRIPT_DIR" "$@" ;;
    audit)
        LABEL="${1:-latest}"
        python "$PIPELINE_DIR/audit.py" "$SCRIPT_DIR/reproduce-${LABEL}" ;;
    audit-all)
        python "$PIPELINE_DIR/audit.py" --all "$SCRIPT_DIR" ;;
    results)    python "$PIPELINE_DIR/aggregate_results.py" "$SCRIPT_DIR" --format "${1:-markdown}" ;;
    results-all)
        python "$PIPELINE_DIR/aggregate_results.py" --all "$(dirname "$SCRIPT_DIR")" --format "${1:-markdown}" ;;
    full)       bash "$PIPELINE_DIR/run_full_pipeline.sh" "$SCRIPT_DIR" "$@" ;;
    stop)       bash "$PIPELINE_DIR/stop_explorer.sh" "$SCRIPT_DIR" ;;
    help|*)
        echo "Usage: bash run.sh <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  explore              Run explorer agent"
        echo "  compress [CKPT]      Compress strategy at checkpoint"
        echo "  reproduce [CKPT]     Reproduce from compressed prompt"
        echo "  sweep [CKPT_LIST]    Run compress+reproduce at multiple checkpoints"
        echo "  compare [CKPT]       Compare explorer vs reproducer"
        echo "  audit [CKPT]         Audit reproducer code for val leakage"
        echo "  audit-all            Audit all reproduce-iter* directories"
        echo "  results [FORMAT]     Aggregate results for this experiment (csv|markdown)"
        echo "  results-all [FORMAT] Aggregate results across all experiments"
        echo "  full [TURNS] [CKPT]  Run full pipeline (explore→compress→reproduce→audit)"
        echo "  stop                 Send STOP signal to explorer"
        ;;
esac
