#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: run_compress.sh <experiment_dir> [CHECKPOINT]
# CHECKPOINT: iteration number, commit hash, or "latest" (default)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"
CHECKPOINT="${2:-latest}"

EXPLORE_DIR="$(get_explore_agent_dir)"

if [ ! -d "$EXPLORE_DIR" ]; then
    echo "ERROR: Explorer agent directory not found: $EXPLORE_DIR"
    echo "Run run_explore.sh first."
    exit 1
fi

# Resolve checkpoint to a commit hash
cd "$EXPLORE_DIR"
if [ "$CHECKPOINT" = "latest" ]; then
    COMMIT=$(git rev-parse HEAD)
    LABEL="latest"
elif [[ "$CHECKPOINT" =~ ^[0-9]+$ ]] && [ ${#CHECKPOINT} -le 4 ]; then
    COMMIT=$(git log --reverse --format='%H' | sed -n "$((CHECKPOINT + 1))p")
    if [ -z "$COMMIT" ]; then
        echo "ERROR: Iteration $CHECKPOINT not found. Available commits:"
        git log --oneline --reverse
        exit 1
    fi
    LABEL="iter${CHECKPOINT}"
else
    COMMIT=$(git rev-parse "$CHECKPOINT" 2>/dev/null)
    if [ -z "$COMMIT" ]; then
        echo "ERROR: Cannot resolve checkpoint '$CHECKPOINT'"
        exit 1
    fi
    LABEL="$(echo "$COMMIT" | head -c 7)"
fi

SHORT_COMMIT=$(echo "$COMMIT" | head -c 7)
echo "Checkpoint: $LABEL (commit $SHORT_COMMIT)"

CONDITION="$(get_condition_suffix)"
COMPRESS_DIR="${EXPERIMENT_DIR}/compress-${CONDITION}-${LABEL}"

if [ -d "$COMPRESS_DIR" ]; then
    echo "Removing existing compress directory for checkpoint $LABEL"
    rm -rf "$COMPRESS_DIR"
fi

echo "Creating compressor directory: $COMPRESS_DIR"
mkdir -p "$COMPRESS_DIR"

# Extract train.py at the specified commit
cd "$EXPLORE_DIR"
git show "${COMMIT}:train.py" > "$COMPRESS_DIR/train.py"
cp "$EXPLORE_DIR/prepare.py" "$COMPRESS_DIR/"

# Get results.tsv up to this commit
SORT_FLAG=$(get_sort_best_flag)
if [ -f "$EXPLORE_DIR/results.tsv" ]; then
    ANCESTOR_COMMITS=$(git log --reverse --format='%H' "$COMMIT")
    SHORT_ANCESTORS=$(echo "$ANCESTOR_COMMITS" | while read c; do echo "$c" | head -c 7; done)

    > "$COMPRESS_DIR/results.tsv"
    while IFS= read -r line; do
        line_hash=$(echo "$line" | cut -f1 | head -c 7)
        if echo "$SHORT_ANCESTORS" | grep -q "$line_hash"; then
            echo "$line" >> "$COMPRESS_DIR/results.tsv"
        fi
    done < "$EXPLORE_DIR/results.tsv"
fi

# Find the best-performing commit and extract its train.py as best_train.py
BEST_COMMIT=""
BEST_METRIC=""
if [ -f "$COMPRESS_DIR/results.tsv" ]; then
    BEST_LINE=$(awk -F'\t' '$1 != "na" && $2+0 == $2 {print $0}' "$COMPRESS_DIR/results.tsv" | sort -t'	' -k2 $SORT_FLAG | head -1)
    BEST_COMMIT=$(echo "$BEST_LINE" | cut -f1)
    BEST_METRIC=$(echo "$BEST_LINE" | cut -f2)
fi

if [ -n "$BEST_COMMIT" ]; then
    # Try to extract train.py from the best commit
    BEST_FULL=$(git rev-parse "$BEST_COMMIT" 2>/dev/null)
    if [ -n "$BEST_FULL" ]; then
        git show "${BEST_FULL}:train.py" > "$COMPRESS_DIR/best_train.py"
        BEST_SHORT=$(echo "$BEST_FULL" | head -c 7)
        echo "Best result: ${METRIC_NAME:-val_accuracy}=${BEST_METRIC} at commit ${BEST_SHORT}"
        export BEST_METRIC_SUMMARY="${METRIC_NAME:-val_accuracy}=${BEST_METRIC} at commit ${BEST_SHORT}"
    else
        # Best commit might have been reverted; fall back to checkpoint train.py
        cp "$COMPRESS_DIR/train.py" "$COMPRESS_DIR/best_train.py"
        export BEST_METRIC_SUMMARY="best commit was reverted — using checkpoint train.py"
    fi
else
    cp "$COMPRESS_DIR/train.py" "$COMPRESS_DIR/best_train.py"
    export BEST_METRIC_SUMMARY="no valid results found"
fi

# Copy iteration logs up to this checkpoint
if [ -d "$EXPLORE_DIR/iterations" ]; then
    mkdir -p "$COMPRESS_DIR/iterations"
    N_COMMITS=$(git log --reverse --format='%H' "$COMMIT" | wc -l)
    MAX_ITER=$((N_COMMITS - 1))
    for f in "$EXPLORE_DIR/iterations/"*.md; do
        [ -f "$f" ] || continue
        basename_f=$(basename "$f" .md)
        iter_num=$(echo "$basename_f" | grep -o '[0-9]\+' | head -1)
        if [ -n "$iter_num" ] && [ "$iter_num" -le "$MAX_ITER" ]; then
            cp "$f" "$COMPRESS_DIR/iterations/"
        fi
    done
fi

# Render the compress prompt from template
"$PIPELINE_DIR/render_prompt.sh" "$EXPERIMENT_DIR" compress > "$COMPRESS_DIR/compress_prompt.md"

# Save metadata
echo "checkpoint=$LABEL" > "$COMPRESS_DIR/checkpoint_info.txt"
echo "commit=$COMMIT" >> "$COMPRESS_DIR/checkpoint_info.txt"
echo "short_commit=$SHORT_COMMIT" >> "$COMPRESS_DIR/checkpoint_info.txt"
echo "best_commit=$BEST_COMMIT" >> "$COMPRESS_DIR/checkpoint_info.txt"
echo "best_metric=$BEST_METRIC" >> "$COMPRESS_DIR/checkpoint_info.txt"

cd "$COMPRESS_DIR"

echo "Starting compressor in $COMPRESS_DIR"
echo "Results up to this checkpoint:"
cat results.tsv 2>/dev/null || echo "(none)"
echo ""

TOKEN_BUDGET="${COMPRESS_TOKENS:-128}"

# Iterative compression: multi-turn with budget enforcement
# Agent writes, sees truncation feedback, revises (up to 5 turns), hard truncate as fallback
python3 "$PIPELINE_DIR/compress_iterative.py" "$COMPRESS_DIR" "$TOKEN_BUDGET" 2>&1 | tee agent.log
