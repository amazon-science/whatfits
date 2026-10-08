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
  one-bit ladder mechanism (`pipeline/`, `experiments/pipeline/`, `ladder/`).
- **Multi-provider agents**: runs on Claude (via the Claude Code CLI) and, via a
  drop-in shim (`codex_provider/`), on GPT-5.5 and GPT-OSS-120B through the Codex
  CLI on Amazon Bedrock — the pipeline scripts are unchanged across providers.
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
- An agent CLI + credentials:
  - **Claude**: the `claude` CLI, configured for your provider (e.g. Amazon
    Bedrock, `us.anthropic.claude-opus-4-6-v1`).
  - **GPT (optional)**: the `codex` CLI (≥ 0.145) — see `codex_provider/`.

## How to run the experiments

Each dataset directory has a `run.sh` wrapper over the pipeline:

```bash
cd experiments/gene-expr        # or nyc-311, gene-expr-sealed

bash run.sh explore             # run the explorer agent (adaptive search)
bash run.sh compress [CKPT]     # compress a checkpoint's strategy to a short prompt
bash run.sh reproduce [CKPT]    # blind reproducer from the compressed prompt
bash run.sh sweep [CKPTS...]    # compress+reproduce across checkpoints
bash run.sh full                # explore → compress → reproduce → audit
```

**Ladder (one-bit) arm:**

```bash
python ladder/run_ladder_experiment.py experiments/<dataset> --framing neutral
python ladder/compute_bound.py experiments/<dataset> ladder-neutral
```

**End-to-end driver** (both arms, with watchdog/timeout guards) — see
`experiments/nyc-311/drive_nyc311.sh` for a full worked example:

```bash
bash experiments/nyc-311/drive_nyc311.sh
```

### Running on a non-Claude model (GPT via Bedrock)

`codex_provider/` contains a drop-in `claude` shim that re-emits the pipeline's
`claude -p …` calls as `codex exec` against Amazon Bedrock, so the same scripts
run on GPT-5.5 / GPT-OSS-120B:

```bash
cp codex_provider/config.toml.example ~/.codex/config.toml   # edit model id
source codex_provider/env.sh                                 # thread caps + region
# put the shim first on PATH as `claude`, then run the pipeline as usual
```

Auth is AWS-native (`AWS_BEARER_TOKEN_BEDROCK`), not `OPENAI_API_KEY`.

### Shared-machine note

`codex_provider/env.sh` exports thread caps (`OMP/OPENBLAS/MKL/NUMEXPR=4`) to
prevent CPU oversubscription; the drivers wrap training/agent calls in `timeout`
guards. Source it before running on a contended box.

## Notes

- Data files, model artifacts, and agent run directories are gitignored;
  regenerate them via the pipeline.
- Provenance: Claude `us.anthropic.claude-opus-4-6-v1` (Bedrock us-west-2);
  GPT `openai.gpt-5.5` / `openai.gpt-oss-120b-1:0` (Bedrock, Codex CLI ≥ 0.145).

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
