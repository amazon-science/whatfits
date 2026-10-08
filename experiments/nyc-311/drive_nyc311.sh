#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# End-to-end driver for the NYC 311 (Jan 2026) compression-certificate experiment.
#
# Runs BOTH arms on Opus 4.6 (the paper's model), unattended, with watchdogs so
# nothing runs forever:
#   ARM A (compression / score-only): explore -> select checkpoints ->
#          compress + reproduce + audit at each -> summary
#   ARM B (ladder / 1-bit feedback):  explore with binary feedback -> bound
#
# Usage: bash drive_nyc311.sh 2>&1 | tee drive_nyc311.log
set -u

# Repo-relative: resolve from this script's location (experiments/nyc-311/).
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$EXP_DIR/../.." && pwd)"
PIPE="$ROOT/experiments/pipeline"
MODEL="us.anthropic.claude-opus-4-6-v1"

# --- Budget knobs (override via env) ---
EXPLORE_TARGET_ITERS="${EXPLORE_TARGET_ITERS:-20}"   # stop explorer after this many iters (>=15 for graceful exit)
EXPLORE_MAX_SECONDS="${EXPLORE_MAX_SECONDS:-5400}"   # hard wall-clock cap per explore arm (90 min)
EXPLORE_MAX_TURNS="${EXPLORE_MAX_TURNS:-600}"        # per claude launch
LADDER_TARGET_ITERS="${LADDER_TARGET_ITERS:-20}"
LADDER_MAX_SECONDS="${LADDER_MAX_SECONDS:-5400}"

export CLAUDE_MODEL="$MODEL"
export VAL_ACCESS_LEVEL=score_only
export EXPLORER_FRAMING=neutral
export COMPRESS_TOKENS=128

banner() { echo; echo "################################################################"; echo "#  $* — $(date)"; echo "################################################################"; }

# Watchdog: when results.tsv in $1 reaches $2 lines, write STOP so the agent
# exits gracefully (agent checks STOP each iteration once >=15 iters done).
# Also enforces a hard time cap.
watchdog() {
    local agent_dir="$1" target="$2" max_secs="$3" start
    start=$(date +%s)
    while true; do
        sleep 30
        [ -d "$agent_dir" ] || continue
        local n now
        n=$(wc -l < "$agent_dir/results.tsv" 2>/dev/null || echo 0)
        now=$(date +%s)
        if [ "$n" -ge "$target" ] && [ ! -f "$agent_dir/STOP" ]; then
            echo "[watchdog] $agent_dir reached $n iters (target $target) -> STOP"
            echo STOP > "$agent_dir/STOP"
        fi
        if [ $((now - start)) -ge "$max_secs" ]; then
            echo "[watchdog] $agent_dir hit time cap ${max_secs}s (iters=$n) -> STOP"
            [ -f "$agent_dir/STOP" ] || echo STOP > "$agent_dir/STOP"
            return 0
        fi
        # Exit once the explore loop has finished (restart.log says finished) or STOP consumed
        if [ -f "$agent_dir/iterations/FINAL.md" ]; then
            echo "[watchdog] $agent_dir wrote FINAL.md -> done"
            return 0
        fi
    done
}

# =====================================================================
banner "ARM A — COMPRESSION (score-only) EXPLORE"
# =====================================================================
EXPLORE_A="$EXP_DIR/explore-agent-scoreonly-neutral"
if [ -f "$EXPLORE_A/results.tsv" ] && [ "$(wc -l < "$EXPLORE_A/results.tsv")" -ge 15 ]; then
    echo "SKIP explore A: already has $(wc -l < "$EXPLORE_A/results.tsv") iters"
else
    watchdog "$EXPLORE_A" "$EXPLORE_TARGET_ITERS" "$EXPLORE_MAX_SECONDS" &
    WD=$!
    bash "$PIPE/run_explore.sh" "$EXP_DIR" "$EXPLORE_MAX_TURNS" 10
    kill "$WD" 2>/dev/null || true
fi
echo "--- explore A results.tsv ---"; cat "$EXPLORE_A/results.tsv" 2>/dev/null || echo "(none)"

# =====================================================================
banner "ARM A — SELECT IMPROVEMENT CHECKPOINTS"
# =====================================================================
CKPTS=$(python3 "$PIPE/select_improvement_checkpoints.py" "$EXPLORE_A/results.tsv" higher 0.005 2>/dev/null)
echo "Improvement checkpoints (commits): $CKPTS"
# Fall back to iteration indices if selection is empty
if [ -z "$CKPTS" ]; then
    echo "No improvement checkpoints selected; falling back to latest."
    CKPTS="latest"
fi

# =====================================================================
banner "ARM A — COMPRESS + REPRODUCE + AUDIT PER CHECKPOINT"
# =====================================================================
COND=scoreonly-neutral
for COMMIT in $CKPTS; do
    SHORT=$(echo "$COMMIT" | head -c 7)
    banner "checkpoint $SHORT"
    rm -rf "$EXP_DIR/compress-${COND}-${SHORT}" "$EXP_DIR/reproduce-${COND}-${SHORT}"
    echo "--- compress $SHORT ---"
    bash "$PIPE/run_compress.sh" "$EXP_DIR" "$COMMIT" || { echo "WARN: compress failed $SHORT"; continue; }
    echo "--- reproduce $SHORT ---"
    bash "$PIPE/run_reproduce.sh" "$EXP_DIR" "$SHORT" || echo "WARN: reproduce errors $SHORT"
    echo "--- audit $SHORT ---"
    python3 "$PIPE/audit.py" "$EXP_DIR/reproduce-${COND}-${SHORT}" || true
done

# =====================================================================
banner "ARM B — LADDER (1-bit feedback) EXPLORE"
# =====================================================================
LADDER_COND="ladder-neutral"
EXPLORE_B="$EXP_DIR/explore-agent-${LADDER_COND}"
if [ -f "$EXPLORE_B/results.tsv" ] && [ "$(wc -l < "$EXPLORE_B/results.tsv")" -ge 15 ]; then
    echo "SKIP ladder: already has $(wc -l < "$EXPLORE_B/results.tsv") iters"
else
    watchdog "$EXPLORE_B" "$LADDER_TARGET_ITERS" "$LADDER_MAX_SECONDS" &
    WD=$!
    # Ladder runner resolves the experiment as PROJECT_ROOT/<arg>, so pass the
    # repo-relative path "experiments/nyc-311" (matches launch_wave.sh usage).
    ( cd "$ROOT" && python3 ladder/run_ladder_experiment.py experiments/nyc-311 \
        --framing neutral --model "$MODEL" --max-turns "$EXPLORE_MAX_TURNS" --max-restarts 10 )
    kill "$WD" 2>/dev/null || true
fi
echo "--- ladder state ---"; cat "$EXPLORE_B/ladder_state.json" 2>/dev/null || echo "(none)"

# =====================================================================
banner "ARM B — LADDER BOUND"
# =====================================================================
( cd "$ROOT" && python3 ladder/compute_bound.py experiments/nyc-311 "$LADDER_COND" ) 2>&1 || true

# =====================================================================
banner "DRIVER COMPLETE"
# =====================================================================
echo "Compression arm dirs:"; ls -d "$EXP_DIR"/compress-* "$EXP_DIR"/reproduce-* 2>/dev/null
echo "Explore A holdout (secret):"; cat "$EXPLORE_A/holdout_results.tsv" 2>/dev/null | tail -5
echo "Done at $(date)."
