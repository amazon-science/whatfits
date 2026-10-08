#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Append sweep results summary to research_log.md
# Usage: append_results_to_log.sh <experiment_dir>
# Called automatically at the end of run_all_checkpoints.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"

CONDITION="$(get_condition_suffix)"
RESULTS_FILE="${EXPERIMENT_DIR}/sweep_results-${CONDITION}.csv"
LOG_FILE="$(dirname "$EXPERIMENT_DIR")/research_log.md"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "No sweep results found: $RESULTS_FILE"
    exit 0
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "No research log found: $LOG_FILE"
    exit 0
fi

# Check if this sweep was already appended (avoid duplicates)
MARKER="<!-- sweep:${EXPERIMENT_NAME}:${CONDITION} -->"
if grep -qF "$MARKER" "$LOG_FILE" 2>/dev/null; then
    # Remove old entry and re-append with fresh data
    # Find the marker and delete until the next marker or section
    python3 -c "
import re, sys
log = open('$LOG_FILE').read()
# Remove old auto-appended block for this experiment+condition
pattern = re.escape('$MARKER') + r'.*?' + re.escape('<!-- /sweep -->')
log = re.sub(pattern, '', log, flags=re.DOTALL)
# Clean up extra blank lines
log = re.sub(r'\n{3,}', '\n\n', log)
open('$LOG_FILE', 'w').write(log)
"
fi

# Get explorer best val and holdout from explore-agent dir
EXPLORE_DIR="$(get_explore_agent_dir)"
EXPLORE_BEST_VAL=""
EXPLORE_BEST_HOLDOUT=""
EXPLORE_ITERS=""
if [ -f "$EXPLORE_DIR/results.tsv" ]; then
    EXPLORE_ITERS=$(wc -l < "$EXPLORE_DIR/results.tsv" | tr -d ' ')
    EXPLORE_BEST_VAL=$(awk -F'\t' '{print $2}' "$EXPLORE_DIR/results.tsv" | sort $(get_sort_best_flag) | head -1)
fi
if [ -f "$EXPLORE_DIR/holdout_results.tsv" ]; then
    EXPLORE_BEST_HOLDOUT=$(sort -t$'\t' -k3 $(get_sort_best_flag) "$EXPLORE_DIR/holdout_results.tsv" | head -1 | cut -f3)
fi

# Build the append block
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

{
    echo ""
    echo "$MARKER"
    echo "### Auto-appended: ${EXPERIMENT_NAME} (${CONDITION}) — ${TIMESTAMP}"
    echo ""
    echo "Explorer: ${EXPLORE_ITERS:-?} iterations, best val=${EXPLORE_BEST_VAL:-?}, best holdout=${EXPLORE_BEST_HOLDOUT:-?}"
    echo ""
    echo "| Checkpoint | Explorer Val | Reproducer Val | Reproducer Holdout | Gap(pp) | Status | Audit |"
    echo "|-----------|-------------|---------------|-------------------|---------|--------|-------|"

    # Skip header line in CSV
    tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r ckpt eval rval rhold gap status audit; do
        printf "| %s | %s | %s | %s | %s | %s | %s |\n" \
            "$ckpt" "$eval" "$rval" "$rhold" "$gap" "$status" "$audit"
    done

    echo ""
    echo "<!-- /sweep -->"
} >> "$LOG_FILE"

echo "Appended sweep results to $LOG_FILE"
