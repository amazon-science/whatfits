#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Source before running the pipeline on a shared/contended machine.
#
#   source codex_provider/env.sh
#
# Thread caps prevent CPU oversubscription: reproducer agents often write
# multi-model stacks that each spawn one thread per core, saturating a shared
# box (observed: load 50+ on 48 cores -> multi-CPU-hour runaways). Capping keeps
# per-fit time in the expected 1-3s range.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# --- Codex/Bedrock provider (for non-Claude runs via the shim) ---
# Requires AWS_BEARER_TOKEN_BEDROCK in the environment (AWS-native auth).
export AWS_REGION="${AWS_REGION:-us-east-2}"
# Put codex_provider/ first on PATH so `claude` resolves to the shim:
#   export PATH="$(pwd)/codex_provider:$PATH"   # then rename shim to `claude`
