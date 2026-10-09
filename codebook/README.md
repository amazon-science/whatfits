<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
-->

# Codebook Prototype

This directory contains an experimental structured-codebook prototype. It is not
part of the main paper reproduction path described in the top-level README,
which uses the free-text compression pipeline in `pipeline/` plus the one-bit
ladder mechanism in `ladder/`.

The scripts are kept for reference because they show an alternative way to
assign explicit bit lengths to structured strategy descriptions. They expect
additional generated artifacts, such as `codebook/experiments/...` directories
and learned `codebook.json` files, which are not shipped in this public release.

Use the top-level README commands for reproducing the accepted-paper
experiments.
