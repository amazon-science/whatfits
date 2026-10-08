#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Full codebook pipeline:
#   Phase 0: build codebook + evaluate baseline
#   Phase 1: explorer from Phase 0 baseline
#   Phase 2: codebook compress (bit-budgeted)
#   Phase 3: codebook reproduce
#   Evaluate + plot
#
# Usage:
#   # Full pipeline from scratch:
#   bash codebook/pipeline/run_pipeline.sh folktables-5k full-analyst
#
#   # Individual phases:
#   bash codebook/pipeline/run_pipeline.sh folktables-5k full-analyst --phase 0
#   bash codebook/pipeline/run_pipeline.sh folktables-5k full-analyst --phase 1
#   bash codebook/pipeline/run_pipeline.sh folktables-5k full-analyst --phase 2-3
#   bash codebook/pipeline/run_pipeline.sh folktables-5k full-analyst --phase plot

set -e

EXPERIMENT="${1:?Usage: $0 <experiment> <condition> [--phase N] [--beta N] [--bit-budget N]}"
CONDITION="${2:?Usage: $0 <experiment> <condition>}"
shift 2

BETA=20
BIT_BUDGET=16
MAX_ITERS=12
PHASE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --beta) BETA="$2"; shift 2;;
        --bit-budget) BIT_BUDGET="$2"; shift 2;;
        --max-iters) MAX_ITERS="$2"; shift 2;;
        --phase) PHASE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CODEBOOK_DIR="$PROJECT_ROOT/codebook"
PIPELINE="$CODEBOOK_DIR/pipeline"
EXP_DIR="$CODEBOOK_DIR/experiments/${EXPERIMENT}--${CONDITION}"
DATASET_DIR="$PROJECT_ROOT/$EXPERIMENT"

# Detect n_val from prepare.py
N_VAL=$(grep "VAL_SIZE" "$DATASET_DIR/prepare.py" 2>/dev/null | head -1 | grep -o '[0-9]\+' || echo 200)

echo "=========================================="
echo "  Codebook Pipeline"
echo "  Experiment: $EXPERIMENT"
echo "  Condition:  $CONDITION"
echo "  n_val:      $N_VAL"
echo "  Beta:       $BETA"
echo "  Bit budget: $BIT_BUDGET bits"
echo "  Phase:      $PHASE"
echo "=========================================="

# --- Setup experiment dir ---
if [ ! -d "$EXP_DIR" ]; then
    echo ""
    echo "=== SETUP: Creating experiment directory ==="
    mkdir -p "$EXP_DIR/checkpoints"
    cat > "$EXP_DIR/config.json" << EOF
{
    "experiment": "$EXPERIMENT",
    "condition": "$CONDITION",
    "explorer_dir": "$EXPERIMENT/explore-agent-${CONDITION}",
    "n_val": $N_VAL,
    "n_holdout": 5000,
    "metric_direction": "higher",
    "metric_name": "val_accuracy",
    "beta": $BETA,
    "bit_budget": $BIT_BUDGET,
    "delta": 0.05
}
EOF
fi

# --- Phase 0: Build codebook + evaluate baseline ---
if [[ "$PHASE" == "all" || "$PHASE" == "0" ]]; then
    echo ""
    echo "=== PHASE 0: Build codebook + evaluate baseline ==="

    CONFIGS_FILE="$EXP_DIR/build/configs.jsonl"
    if [ ! -f "$CONFIGS_FILE" ]; then
        python "$PIPELINE/build_codebook_agent.py" "$EXPERIMENT" --max-iters "$MAX_ITERS"
        # Move build dir to experiment
        BUILD_SRC="$CODEBOOK_DIR/build-${EXPERIMENT}"
        if [ -d "$BUILD_SRC" ]; then
            mv "$BUILD_SRC" "$EXP_DIR/build"
        fi
    else
        echo "Phase 0 configs already exist: $CONFIGS_FILE"
    fi

    # Build performance-weighted codebook
    python "$PIPELINE/build_codebook_from_configs.py" "$CONFIGS_FILE" \
        --beta "$BETA" --n_val "$N_VAL" --name "$EXPERIMENT" \
        --output "$EXP_DIR/codebook.json"

    # Evaluate baseline on val + holdout
    python "$PIPELINE/evaluate_phase0.py" "$EXP_DIR"
fi

# --- Phase 1: Explorer from Phase 0 baseline ---
if [[ "$PHASE" == "all" || "$PHASE" == "1" ]]; then
    echo ""
    echo "=== PHASE 1: Explorer from Phase 0 baseline ==="

    EXPLORE_DIR="$DATASET_DIR/explore-agent-${CONDITION}"
    if [ -d "$EXPLORE_DIR" ] && [ -f "$EXPLORE_DIR/results.tsv" ]; then
        N_ITERS=$(wc -l < "$EXPLORE_DIR/results.tsv")
        echo "Explorer already has $N_ITERS iterations in $EXPLORE_DIR"
    else
        python "$PIPELINE/run_explore_from_phase0.py" "$EXP_DIR" \
            --max-turns 2000 --max-restarts 10
    fi
fi

# --- Phase 2+3: Compress + Reproduce ---
if [[ "$PHASE" == "all" || "$PHASE" == "2-3" || "$PHASE" == "2" || "$PHASE" == "3" ]]; then
    echo ""
    echo "=== SELECT CHECKPOINTS ==="

    EXPLORE_DIR="$DATASET_DIR/explore-agent-${CONDITION}"
    DIRECTION=$(python3 -c "import json; print(json.load(open('$EXP_DIR/config.json'))['metric_direction'])")
    CHECKPOINTS=$(python "$PROJECT_ROOT/pipeline/select_improvement_checkpoints.py" \
        "$EXPLORE_DIR/results.tsv" "$DIRECTION" 0.01 2>/dev/null | tail -1)

    echo "Selected: $CHECKPOINTS"
    CKPT_ARRAY=($CHECKPOINTS)

    if [[ "$PHASE" != "3" ]]; then
        echo ""
        echo "=== PHASE 2: Codebook compress (${BIT_BUDGET}-bit budget) ==="
        python "$PIPELINE/compress_iterative.py" "$EXP_DIR" \
            --checkpoints ${CKPT_ARRAY[@]} --bit-budget "$BIT_BUDGET"
    fi

    if [[ "$PHASE" != "2" ]]; then
        echo ""
        echo "=== PHASE 3: Codebook reproduce ==="
        python "$PIPELINE/reproduce.py" "$EXP_DIR" \
            --checkpoints ${CKPT_ARRAY[@]}
    fi
fi

# --- Plot ---
if [[ "$PHASE" == "all" || "$PHASE" == "plot" ]]; then
    echo ""
    echo "=== PLOT ==="
    python "$PIPELINE/plot_timeline.py" "$EXP_DIR"
fi

echo ""
echo "Pipeline complete: $EXP_DIR"
