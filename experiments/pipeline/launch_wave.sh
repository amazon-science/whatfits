#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Launch a named tmux session with one window per dataset/arm for a given run.
# Usage: launch_wave.sh <session_name> <run_id> <arm> <dataset> [dataset ...]
#   arm = scoreonly | ladder
#
# Example:
#   launch_wave.sh reruns-r2 2 scoreonly folktables gene-expr cifar-cnn
#
# Window names encode dataset+arm. Each window runs ONE explorer (foreground)
# so it stays visible and killable. Attach with: tmux attach -t <session_name>

set -eu

SESSION="$1"
RUN_ID="$2"
ARM="$3"
shift 3

REPO="/fsx/gstevenw/testing_alignment_algos/compression-cert"
CLAUDE_MODEL="us.anthropic.claude-opus-4-6-v1"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Adding windows to it."
    CREATE=0
else
    CREATE=1
fi

launch_one() {
    local dataset="$1"
    local winname="${dataset}-${ARM}"

    local cmd
    if [ "$ARM" = "scoreonly" ]; then
        cmd="cd $REPO && RUN_ID=$RUN_ID CLAUDE_MODEL=$CLAUDE_MODEL \
VAL_ACCESS_LEVEL=score_only EXPLORER_FRAMING=neutral \
bash experiments/pipeline/run_explore.sh experiments/$dataset"
    elif [ "$ARM" = "ladder" ]; then
        cmd="cd $REPO && python ladder/run_ladder_experiment.py experiments/$dataset \
--framing neutral --run-id $RUN_ID --model $CLAUDE_MODEL"
    else
        echo "Unknown arm: $ARM" >&2
        exit 1
    fi

    if [ "$CREATE" = "1" ]; then
        tmux new-session -d -s "$SESSION" -n "$winname" "$cmd"
        CREATE=0
    else
        tmux new-window -t "$SESSION" -n "$winname" "$cmd"
    fi
    echo "Launched window: $winname"
}

for dataset in "$@"; do
    launch_one "$dataset"
done

echo ""
echo "Session created. Attach with:"
echo "  tmux attach -t $SESSION"
echo ""
echo "List windows:"
tmux list-windows -t "$SESSION"
