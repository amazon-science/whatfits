#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: run_all_checkpoints.sh <experiment_dir> [CHECKPOINT_LIST]
# Default checkpoints: 2 5 8 10 12 15

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"
shift 2>/dev/null || true

if [ $# -gt 0 ]; then
    CHECKPOINTS=("$@")
else
    CHECKPOINTS=(2 5 8 10 12 15)
fi

echo "=========================================="
echo "  ${EXPERIMENT_NAME} — COMPRESSION SWEEP"
echo "  Checkpoints: ${CHECKPOINTS[*]}"
echo "=========================================="

CONDITION="$(get_condition_suffix)"

for ITER in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "############################################"
    echo "  CHECKPOINT: $ITER"
    echo "############################################"

    # Determine the label that run_compress.sh will use
    # Short all-digit strings (≤4 chars) are iteration numbers; longer or mixed are commit hashes
    if [[ "$ITER" =~ ^[0-9]+$ ]] && [ ${#ITER} -le 4 ]; then
        LABEL="iter${ITER}"
    else
        # Commit hash — run_compress.sh uses first 7 chars
        LABEL="$(echo "$ITER" | head -c 7)"
    fi

    rm -rf "${EXPERIMENT_DIR}/compress-${CONDITION}-${LABEL}" "${EXPERIMENT_DIR}/reproduce-${CONDITION}-${LABEL}"

    echo "--- Compressing at $ITER ---"
    bash "$PIPELINE_DIR/run_compress.sh" "$EXPERIMENT_DIR" "$ITER" || { echo "WARNING: compress failed for $ITER"; continue; }

    echo ""
    echo "--- Reproducing from $ITER ---"
    bash "$PIPELINE_DIR/run_reproduce.sh" "$EXPERIMENT_DIR" "$LABEL" || echo "WARNING: reproduce had errors for $ITER (exit $?)"

    echo ""
    echo "--- Auditing reproducer for $ITER ---"
    python "$PIPELINE_DIR/audit.py" "${EXPERIMENT_DIR}/reproduce-${CONDITION}-${LABEL}" || true

    echo ""
    echo "--- Results for $ITER ---"
    bash "$PIPELINE_DIR/compare_results.sh" "$EXPERIMENT_DIR" "$LABEL" || true
    echo ""
done

# Summary table
echo ""
echo "=========================================="
echo "  SUMMARY TABLE"
echo "=========================================="
printf "%-6s  %-10s  %-10s  %-12s  %-12s  %-8s  %-8s\n" \
    "Iter" "Expl.Val" "Repr.Val" "Repr.Holdout" "Gap(pp)" "Status" "Audit"
echo "------  ----------  ----------  ------------  ------------  --------  --------"

for ITER in "${CHECKPOINTS[@]}"; do
    if [[ "$ITER" =~ ^[0-9]+$ ]] && [ ${#ITER} -le 4 ]; then
        LABEL="iter${ITER}"
    else
        LABEL="$(echo "$ITER" | head -c 7)"
    fi
    COMPRESS_DIR="${EXPERIMENT_DIR}/compress-${CONDITION}-${LABEL}"
    REPRODUCE_DIR="${EXPERIMENT_DIR}/reproduce-${CONDITION}-${LABEL}"

    EXPL_VAL=""
    if [ -f "$COMPRESS_DIR/results.tsv" ]; then
        EXPL_VAL=$(awk -F'\t' '{print $2}' "$COMPRESS_DIR/results.tsv" | sort $(get_sort_best_flag) | head -1)
    fi

    REPR_VAL=""
    REPR_HOLDOUT=""
    REPR_GAP=""
    if [ -f "$REPRODUCE_DIR/holdout_results.tsv" ]; then
        BEST_LINE=$(sort -t$'\t' -k2 $(get_sort_best_flag) "$REPRODUCE_DIR/holdout_results.tsv" | head -1)
        REPR_VAL=$(echo "$BEST_LINE" | cut -f2)
        REPR_HOLDOUT=$(echo "$BEST_LINE" | cut -f3)
        if [ -n "$REPR_VAL" ] && [ -n "$REPR_HOLDOUT" ]; then
            REPR_GAP=$(echo "$REPR_VAL $REPR_HOLDOUT" | awk '{printf "%.1f", ($1-$2)*100}')
        fi
    fi

    STATUS="?"
    if [ -n "$EXPL_VAL" ] && [ -n "$REPR_VAL" ]; then
        DIFF=$(echo "$EXPL_VAL $REPR_VAL" | awk '{printf "%.4f", $1-$2}')
        # higher-is-better: PASS if explorer-reproducer < threshold (reproducer matched or beat)
        # lower-is-better:  PASS if reproducer-explorer < threshold (i.e. -DIFF < threshold)
        if [ "${METRIC_DIRECTION:-higher}" = "lower" ]; then
            IS_CLOSE=$(echo "$DIFF $CERT_THRESHOLD" | awk '{print (-$1 < $2) ? 1 : 0}')
        else
            IS_CLOSE=$(echo "$DIFF $CERT_THRESHOLD" | awk '{print ($1 < $2) ? 1 : 0}')
        fi
        if [ "$IS_CLOSE" = "1" ]; then
            STATUS="PASS"
        else
            STATUS="FAIL"
        fi
    fi

    AUDIT="?"
    if [ -d "${REPRODUCE_DIR}" ]; then
        if python "$PIPELINE_DIR/audit.py" "${REPRODUCE_DIR}" > /dev/null 2>&1; then
            AUDIT="CLEAN"
        else
            AUDIT="TAINTED"
        fi
    fi

    printf "%-6s  %-10s  %-10s  %-12s  %-12s  %-8s  %-8s\n" \
        "$ITER" "$EXPL_VAL" "$REPR_VAL" "$REPR_HOLDOUT" "$REPR_GAP" "$STATUS" "$AUDIT"
done

echo ""
echo "PASS = reproducer within ${CERT_THRESHOLD} of explorer val (strategy compresses)"
echo "FAIL = reproducer >${CERT_THRESHOLD} from explorer val (strategy incompressible)"

# Persist results to CSV
RESULTS_FILE="${EXPERIMENT_DIR}/sweep_results-${CONDITION}.csv"
echo "checkpoint,explorer_val,reproducer_val,reproducer_holdout,gap_pp,status,audit" > "$RESULTS_FILE"
for ITER in "${CHECKPOINTS[@]}"; do
    if [[ "$ITER" =~ ^[0-9]+$ ]] && [ ${#ITER} -le 4 ]; then
        LABEL="iter${ITER}"
    else
        LABEL="$(echo "$ITER" | head -c 7)"
    fi
    COMPRESS_DIR="${EXPERIMENT_DIR}/compress-${CONDITION}-${LABEL}"
    REPRODUCE_DIR="${EXPERIMENT_DIR}/reproduce-${CONDITION}-${LABEL}"
    E=""; R=""; H=""; G=""; S="?"; A="?"
    [ -f "$COMPRESS_DIR/results.tsv" ] && E=$(awk -F'\t' '{print $2}' "$COMPRESS_DIR/results.tsv" | sort $(get_sort_best_flag) | head -1)
    if [ -f "$REPRODUCE_DIR/holdout_results.tsv" ]; then
        BEST_LINE=$(sort -t$'\t' -k2 $(get_sort_best_flag) "$REPRODUCE_DIR/holdout_results.tsv" | head -1)
        R=$(echo "$BEST_LINE" | cut -f2); H=$(echo "$BEST_LINE" | cut -f3)
        [ -n "$R" ] && [ -n "$H" ] && G=$(echo "$R $H" | awk '{printf "%.1f", ($1-$2)*100}')
    fi
    if [ -n "$E" ] && [ -n "$R" ]; then
        DIFF=$(echo "$E $R" | awk '{printf "%.4f", $1-$2}')
        if [ "${METRIC_DIRECTION:-higher}" = "lower" ]; then
            S=$(echo "$DIFF $CERT_THRESHOLD" | awk '{print (-$1 < $2) ? "PASS" : "FAIL"}')
        else
            S=$(echo "$DIFF $CERT_THRESHOLD" | awk '{print ($1 < $2) ? "PASS" : "FAIL"}')
        fi
    fi
    [ -d "$REPRODUCE_DIR" ] && { python "$PIPELINE_DIR/audit.py" "$REPRODUCE_DIR" > /dev/null 2>&1 && A="CLEAN" || A="TAINTED"; }
    echo "$ITER,$E,$R,$H,$G,$S,$A" >> "$RESULTS_FILE"
done
echo ""
echo "Results saved to: $RESULTS_FILE"

# Auto-append to research log
bash "$PIPELINE_DIR/append_results_to_log.sh" "$EXPERIMENT_DIR"
