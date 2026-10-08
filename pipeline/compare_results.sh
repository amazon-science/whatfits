#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: compare_results.sh <experiment_dir> [CHECKPOINT]

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"
LABEL="${2:-latest}"

EXPLORE_DIR="$(get_explore_agent_dir)"
CONDITION="$(get_condition_suffix)"
COMPRESS_DIR="${EXPERIMENT_DIR}/compress-${CONDITION}-${LABEL}"
REPRODUCE_DIR="${EXPERIMENT_DIR}/reproduce-${CONDITION}-${LABEL}"

echo "=== COMPRESSION CERTIFICATE: ${EXPERIMENT_NAME} checkpoint=${LABEL} ==="
echo ""

echo "--- Explorer Agent (at checkpoint ${LABEL}) ---"
if [ -f "$COMPRESS_DIR/results.tsv" ]; then
    echo "Val results up to this checkpoint:"
    cat "$COMPRESS_DIR/results.tsv"
    echo ""
    BEST_VAL=$(cut -f2 "$COMPRESS_DIR/results.tsv" | sort $(get_sort_best_flag) | head -1)
    echo "Best val_accuracy: $BEST_VAL"
else
    echo "(no results.tsv)"
fi
echo ""

if [ -f "$EXPLORE_DIR/holdout_results.tsv" ]; then
    echo "Holdout (secret) for explorer:"
    echo "  Best val (with holdout):"
    sort -t$'\t' -k2 $(get_sort_best_flag) "$EXPLORE_DIR/holdout_results.tsv" | head -1 | \
        awk -F'\t' '{printf "    val=%.4f  holdout=%.4f  gap=%.1fpp\n", $2, $3, ($2-$3)*100}'
    echo "  Best holdout:"
    sort -t$'\t' -k3 $(get_sort_best_flag) "$EXPLORE_DIR/holdout_results.tsv" | head -1 | \
        awk -F'\t' '{printf "    val=%.4f  holdout=%.4f  gap=%.1fpp\n", $2, $3, ($2-$3)*100}'
fi
echo ""

echo "--- Reproducer Agent (checkpoint ${LABEL}) ---"
if [ -f "$REPRODUCE_DIR/results.tsv" ]; then
    echo "Val results:"
    cat "$REPRODUCE_DIR/results.tsv"
else
    echo "(no results.tsv)"
fi
echo ""

if [ -f "$REPRODUCE_DIR/holdout_results.tsv" ]; then
    echo "Holdout (secret) for reproducer:"
    echo "  Best val (with holdout):"
    sort -t$'\t' -k2 $(get_sort_best_flag) "$REPRODUCE_DIR/holdout_results.tsv" | head -1 | \
        awk -F'\t' '{printf "    val=%.4f  holdout=%.4f  gap=%.1fpp\n", $2, $3, ($2-$3)*100}'
    echo "  Best holdout:"
    sort -t$'\t' -k3 $(get_sort_best_flag) "$REPRODUCE_DIR/holdout_results.tsv" | head -1 | \
        awk -F'\t' '{printf "    val=%.4f  holdout=%.4f  gap=%.1fpp\n", $2, $3, ($2-$3)*100}'
fi
echo ""

echo "--- Compressed Prompt (${LABEL}) ---"
if [ -f "$COMPRESS_DIR/compressed_prompt.md" ]; then
    cat "$COMPRESS_DIR/compressed_prompt.md"
    echo ""
    echo "Word count: $(wc -w < "$COMPRESS_DIR/compressed_prompt.md") words"
fi
echo ""

echo "--- Verdict ---"
if [ -f "$REPRODUCE_DIR/holdout_results.tsv" ] && [ -f "$EXPLORE_DIR/holdout_results.tsv" ]; then
    EXPLORE_BEST_VAL=$(sort -t$'\t' -k2 $(get_sort_best_flag) "$EXPLORE_DIR/holdout_results.tsv" | head -1 | cut -f2)
    REPRODUCE_BEST_VAL=$(sort -t$'\t' -k2 $(get_sort_best_flag) "$REPRODUCE_DIR/holdout_results.tsv" | head -1 | cut -f2)
    REPRODUCE_HOLDOUT=$(sort -t$'\t' -k2 $(get_sort_best_flag) "$REPRODUCE_DIR/holdout_results.tsv" | head -1 | cut -f3)
    echo "Explorer best val:   $EXPLORE_BEST_VAL"
    echo "Reproducer best val: $REPRODUCE_BEST_VAL"
    echo "Reproducer holdout:  $REPRODUCE_HOLDOUT"
fi
