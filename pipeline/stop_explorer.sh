#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: stop_explorer.sh <experiment_dir>

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "${1:-.}"

AGENT_DIR="$(get_explore_agent_dir)"
if [ ! -d "$AGENT_DIR" ]; then
    echo "No explorer agent directory found."
    exit 1
fi

echo "STOP" > "$AGENT_DIR/STOP"
echo "STOP signal sent to explorer in $AGENT_DIR"
