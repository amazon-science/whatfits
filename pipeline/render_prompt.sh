#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Usage: render_prompt.sh <experiment_dir> <template_type>
# template_type: explore | compress | reproduce
# Outputs rendered prompt to stdout.
#
# Replaces all {{VAR}} placeholders with the corresponding shell variable.
# Variables come from experiment.conf (sourced by lib.sh).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
resolve_experiment_dir "$1"
TEMPLATE_TYPE="$2"

# Check for experiment-specific override first
if [ -f "$EXPERIMENT_DIR/${TEMPLATE_TYPE}_prompt.md" ]; then
    cat "$EXPERIMENT_DIR/${TEMPLATE_TYPE}_prompt.md"
    exit 0
fi

TEMPLATE_FILE="$PIPELINE_DIR/templates/${TEMPLATE_TYPE}_prompt.md.tmpl"
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: Template not found: $TEMPLATE_FILE" >&2
    exit 1
fi

# Derive computed variables for explore template
if [ "$TEMPLATE_TYPE" = "explore" ]; then
    derive_explore_vars
fi

content=$(cat "$TEMPLATE_FILE")

# Generic replacement: find all {{VAR}} patterns and replace with shell variable values
# Extract unique variable names from template
vars=$(grep -Eo '\{\{[A-Z_]+\}\}' "$TEMPLATE_FILE" | sort -u | sed 's/{{//;s/}}//')

for var in $vars; do
    value="${!var-}"
    # Default COMPRESS_TOKENS to 128 if not set
    if [ "$var" = "COMPRESS_TOKENS" ] && [ -z "$value" ]; then
        value="128"
    fi
    content="${content//\{\{${var}\}\}/$value}"
done

# For reproduce template: insert compressed prompt (may be set by caller)
if [ -n "$COMPRESSED_PROMPT" ]; then
    content="${content//\{\{COMPRESSED_PROMPT\}\}/$COMPRESSED_PROMPT}"
fi

echo "$content"
