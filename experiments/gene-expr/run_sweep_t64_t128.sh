#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Gene-expr compression sweep: t64 (primary) then t128 (secondary)
# This script waits for the explorer to finish, selects checkpoints, and runs both budgets.

set -e

EXPDIR="/fsx/gstevenw/testing_alignment_algos/compression-cert/experiments/gene-expr"
PIPELINE="/fsx/gstevenw/testing_alignment_algos/compression-cert/pipeline"

cd "$EXPDIR"

echo "=== Gene-expr sweep: waiting for explorer to finish ==="
echo "$(date): Starting monitor"

# Wait for explorer to have 15+ result lines
while true; do
    if [ -f explore-agent-scoreonly-neutral/results.tsv ]; then
        COUNT=$(wc -l < explore-agent-scoreonly-neutral/results.tsv)
        echo "$(date): results.tsv has $COUNT lines"
        if [ "$COUNT" -ge 15 ]; then
            echo "Reached 15+ entries. Explorer done (or close enough)."
            break
        fi
    else
        echo "$(date): No results.tsv yet"
    fi
    sleep 120
done

# Give agent a moment to write final results
sleep 30

echo ""
echo "=== Selecting improvement checkpoints ==="
CHECKPOINTS=$(python "$PIPELINE/select_improvement_checkpoints.py" \
    explore-agent-scoreonly-neutral/results.tsv higher 0.01)

if [ -z "$CHECKPOINTS" ]; then
    echo "ERROR: No checkpoints selected. Exiting."
    exit 1
fi

echo "Selected checkpoints: $CHECKPOINTS"
CKPT_ARRAY=($CHECKPOINTS)

# Export shared env vars
export VAL_ACCESS_LEVEL=score_only
export EXPLORER_FRAMING=neutral
export EXPLORE_AGENT_DIR=explore-agent-scoreonly-neutral

echo ""
echo "=========================================="
echo "  PRIMARY SWEEP: t64 (COMPRESS_TOKENS=64)"
echo "=========================================="
COMPRESS_TOKENS=64 bash "$PIPELINE/run_all_checkpoints.sh" "$EXPDIR" "${CKPT_ARRAY[@]}"

echo ""
echo "=========================================="
echo "  SECONDARY SWEEP: t128 (COMPRESS_TOKENS=128)"
echo "=========================================="
COMPRESS_TOKENS=128 bash "$PIPELINE/run_all_checkpoints.sh" "$EXPDIR" "${CKPT_ARRAY[@]}"

echo ""
echo "=========================================="
echo "  ALL DONE — Gene-expr sweeps complete"
echo "  Primary: t64 results in sweep_results-scoreonly-neutral-t64.csv"
echo "  Secondary: t128 results in sweep_results-scoreonly-neutral.csv"
echo "=========================================="
