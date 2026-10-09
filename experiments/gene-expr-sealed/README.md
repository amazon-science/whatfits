<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
-->

# Sealed Gene-Expression Experiment

This is the same synthetic high-dimensional classification task as
`experiments/gene-expr`, but the agent-facing `prepare.py` contains no generator
constants, signal indices, random seeds, or label construction logic. It only
loads a pre-materialized cache.

## Stage The Cache

From this directory:

```bash
uv run python stage_data.py
uv run python prepare.py
```

The staging helper writes:

```text
~/.cache/autoresearch-geneexpr-sealed/data_gene_expr_sealed.pkl
```

The pipeline copies only `prepare.py`, `prepare_reproduce.py`, `train.py`, and
`pyproject.toml` into agent work directories, so this staging helper is not part
of the agent-visible task files.
