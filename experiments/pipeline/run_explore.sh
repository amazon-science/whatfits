#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: run_explore.sh <experiment_dir> [MAX_TURNS] [MAX_RESTARTS]
#
# Runs the explorer agent with automatic restarts. When the agent dies
# (context limits, rate limits), it relaunches with iteration history
# injected into the prompt so the agent can continue effectively.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"
MAX_TURNS="${2:-2000}"
MAX_RESTARTS="${3:-10}"

AGENT_DIR="$(get_explore_agent_dir)"

if [ ! -d "$AGENT_DIR" ]; then
    echo "Creating explorer agent directory: $AGENT_DIR"
    mkdir -p "$AGENT_DIR"

    # Copy experiment files
    for f in $(get_explore_files); do
        if [ -f "$EXPERIMENT_DIR/$f" ]; then
            cp "$EXPERIMENT_DIR/$f" "$AGENT_DIR/"
        fi
    done
    # Render explore prompt (template or override)
    bash "$PIPELINE_DIR/render_prompt.sh" "$EXPERIMENT_DIR" explore > "$AGENT_DIR/explore_prompt.md"
    mkdir -p "$AGENT_DIR/iterations"

    # Write run metadata for reproducibility
    cat > "$AGENT_DIR/run_meta.json" <<META
{
  "run_id": "${RUN_ID:-1}",
  "claude_model": "${CLAUDE_MODEL:-opus}",
  "val_access_level": "${VAL_ACCESS_LEVEL:-score_only}",
  "explorer_framing": "${EXPLORER_FRAMING:-standard}",
  "compress_tokens": "${COMPRESS_TOKENS:-128}",
  "condition_suffix": "$(get_condition_suffix)",
  "launched": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)"
}
META

    cd "$AGENT_DIR"

    # Run optional setup command (e.g., embedding extraction)
    if [ -n "$SETUP_CMD" ]; then
        echo "Running setup: $SETUP_CMD"
        eval "$SETUP_CMD"
    fi

    git init
    git add $(get_explore_files)
    git commit -m "baseline"
else
    echo "Explorer agent directory already exists: $AGENT_DIR"
    cd "$AGENT_DIR"
fi

echo "Starting explorer agent in $AGENT_DIR"
echo "Max turns: $MAX_TURNS, Max restarts: $MAX_RESTARTS"
echo "Logging to agent.log, restart tracking in restart.log"

# ---------------------------------------------------------------------------
# Restart loop with context injection
# ---------------------------------------------------------------------------

for attempt in $(seq 1 "$MAX_RESTARTS"); do
    echo "=== Launch $attempt at $(date) ===" >> restart.log
    ITERS_BEFORE=$(wc -l < results.tsv 2>/dev/null || echo 0)

    # Build context from previous iteration logs
    CONTEXT=""
    if [ -d iterations ] && ls iterations/*.md >/dev/null 2>&1; then
        CONTEXT="

## Previous session history

You are resuming from a previous session. Here are the key lessons from prior iterations:

$(for f in $(ls iterations/*.md 2>/dev/null | sort -t/ -k2 -V); do
    iter_name=$(basename "$f" .md)
    # Skip FINAL.md
    [ "$iter_name" = "FINAL" ] && continue
    echo "### Iteration $iter_name"
    # Extract strategy, result, and lesson lines (keep compact)
    grep -E "^- \*\*(Strategy|Result|Lesson|Next)\*\*" "$f" 2>/dev/null
    echo ""
done)

Resume from iteration $((ITERS_BEFORE + 1)). Do NOT repeat strategies that already failed.
"
    fi

    FULL_PROMPT="$(cat explore_prompt.md)${CONTEXT}"

    claude --dangerously-skip-permissions --max-turns "$MAX_TURNS" \
        --model "${CLAUDE_MODEL:-opus}" \
        -p "$FULL_PROMPT" >> agent.log 2>&1

    ITERS_AFTER=$(wc -l < results.tsv 2>/dev/null || echo 0)
    echo "  Ended at $(date), iters: $ITERS_BEFORE -> $ITERS_AFTER" >> restart.log

    # Check if STOP file exists
    if [ -f STOP ]; then
        echo "  STOP file found, exiting loop" >> restart.log
        break
    fi

    # Check if FINAL.md was written (graceful stop)
    if [ -f iterations/FINAL.md ]; then
        echo "  FINAL.md found, agent finished gracefully" >> restart.log
        break
    fi

    # If no progress was made, wait before retry (may be rate limited)
    if [ "$ITERS_AFTER" -eq "$ITERS_BEFORE" ]; then
        echo "  No progress, waiting 60s before retry" >> restart.log
        sleep 60
    else
        sleep 5
    fi
done

echo "=== Loop finished at $(date), total iters: $(wc -l < results.tsv 2>/dev/null || echo 0) ===" >> restart.log
