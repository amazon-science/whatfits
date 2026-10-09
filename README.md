<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
-->

# whatfits

Code repository for the paper **"What Fits (Into Few Tokens) Doesn't Overfit."**
📄 Paper: https://arxiv.org/html/2606.11045v1

Benchmark-driven machine learning reuses held-out data adaptively, which in
principle invites overfitting — yet in practice it overfits far less than
worst-case theory predicts. This repository tests the hypothesis that
**successful ML strategies are highly compressible**, using resettable LLM
research agents and two complementary information bottlenecks.

## What this code does

The pipeline runs autonomous ML-research agents through two bottlenecks that
narrow the validation-data channel:

- **Output compression** — an *explorer* agent adaptively searches for a
  high-performing model using a validation set; a *compressor* distills its
  strategy into an extremely short prompt (≤128 tokens); a fresh, blind
  *reproducer* must reproduce the performance from that prompt and the training
  data alone (no validation set, no explorer code).
- **Input compression (ladder)** — the explorer sees the validation set only
  through one-bit feedback (did this model improve on the running best?),
  yielding a description-length bound and simultaneous confidence intervals over
  its improvement checkpoints.

### Capabilities

- **Full agentic pipeline**: explore → compress → reproduce → audit, plus the
  one-bit ladder mechanism (`pipeline/`, `ladder/`). `pipeline/` is the
  canonical implementation; `experiments/pipeline/` is a compatibility copy for
  older launch scripts.
- **Configurable agent provider**: runs explorer, compressor, and reproducer loops
  through Claude Code by default, with an explicit Codex CLI option for
  cross-model-family checks.
- **Leakage auditing** (`experiments/pipeline/audit.py`) and a token-budget
  sweep for measuring reproduction quality vs. prompt length.
- **Datasets** live together under `experiments/`, each a self-contained module
  (`prepare.py`, `prepare_reproduce.py`, `experiment.conf`, `train.py`):
  - `gene-expr` — synthetic high-dimensional (p ≫ n) classification.
  - `gene-expr-sealed` — same data with the generator hidden, so the strategy
    must be learned rather than read from the preparation code.
  - `nyc-311` — real tabular classification on NYC 311 service requests.

## Requirements

- Python ≥ 3.11, with [`uv`](https://github.com/astral-sh/uv) (`uv run` is the
  runner). Core deps: scikit-learn, xgboost, numpy, scipy, pandas.
- Agent CLI:
  - Default: Claude Code CLI; `claude` must be installed, authenticated, and
    available on `PATH`.
  - Optional: Codex CLI; `codex` must be installed and configured for the target
    provider/model.

## How to run the experiments

Install the root helper environment once:

```bash
uv sync
```

The synthetic `gene-expr` task stages itself on first use. The two datasets that
need explicit staging are:

```bash
# Real post-cutoff control data from NYC Open Data
cd experiments/nyc-311
uv run python download_data.py
uv run python prepare.py

# Sealed synthetic variant: materialize cache, then verify the loader
cd ../gene-expr-sealed
uv run python stage_data.py
uv run python prepare.py
```

From the repository root, each dataset directory has a `run.sh` wrapper over the
pipeline:

```bash
cd experiments/gene-expr        # or use experiments/nyc-311, experiments/gene-expr-sealed

bash run.sh explore             # run the explorer agent (adaptive search)
bash run.sh compress [CKPT]     # compress a checkpoint's strategy to a short prompt
bash run.sh reproduce [CKPT]    # blind reproducer from the compressed prompt
bash run.sh sweep [CKPTS...]    # compress+reproduce across checkpoints
bash run.sh full                # explore → compress → reproduce → audit
```

**Ladder (one-bit) arm:**

Run these from the repository root:

```bash
uv run python ladder/run_ladder_experiment.py experiments/<dataset> --framing neutral
uv run python ladder/compute_bound.py experiments/<dataset> ladder-neutral
```

**End-to-end driver** (both arms, with watchdog/timeout guards) — see
`experiments/nyc-311/drive_nyc311.sh` for a full worked example:

```bash
bash experiments/nyc-311/drive_nyc311.sh
```

### Agent Provider

The pipeline uses `AGENT_PROVIDER=claude` by default and calls:

```bash
claude --dangerously-skip-permissions --max-turns N --model "$AGENT_MODEL" -p "$PROMPT"
```

To run the same rendered prompts through Codex instead, set:

```bash
export AGENT_PROVIDER=codex
export AGENT_MODEL=openai.gpt-5.5
export CODEX_MODEL_PROVIDER=amazon-bedrock
export CODEX_SANDBOX=workspace-write
```

This invokes Codex directly through `pipeline/agent_runner.py`; it does not shadow
`claude` on `PATH`. If the Codex agent needs unrestricted local git writes inside
an isolated checkout, use `CODEX_SANDBOX=danger-full-access`. Non-Claude runs add
the provider to generated condition directories, e.g. `explore-agent-scoreonly-neutral-codex`,
unless `EXPLORE_AGENT_DIR` is explicitly set.

### Paper Reproduction Map

- Output compression: `bash experiments/<dataset>/run.sh full`
- One-bit ladder: `uv run python ladder/run_ladder_experiment.py experiments/<dataset> --framing neutral`
- Gene-Expr sealed control: stage with `experiments/gene-expr-sealed/stage_data.py`, then use the same `run.sh` commands.
- NYC 311 post-cutoff control: stage with `experiments/nyc-311/download_data.py`, then use the same `run.sh` commands.

### Shared-machine note

Set thread caps before running on a contended machine; reproducer agents often
write multi-model stacks that otherwise spawn one thread per core:

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

The drivers also wrap training/agent calls in `timeout` guards.

## Notes

- Data files, model artifacts, and agent run directories are gitignored;
  regenerate them via the pipeline.
- Paper-run provenance: Claude `us.anthropic.claude-opus-4-6-v1` (Bedrock
  us-west-2). Codex/GPT reruns should record `AGENT_PROVIDER`, `AGENT_MODEL`,
  `CODEX_MODEL_PROVIDER`, and `CODEX_SANDBOX` in their run notes.
- `codebook/` is an experimental structured-description prototype and is not
  needed for the main paper reproduction path.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more
information.

## License

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

This project is licensed under the Creative Commons Attribution-NonCommercial
4.0 International Public License. See [LICENSE](LICENSE).

Third-party dependencies are declared in package manager configuration files and
are obtained from their original sources; this repository does not vendor
third-party source code.

SPDX-License-Identifier: CC-BY-NC-4.0
