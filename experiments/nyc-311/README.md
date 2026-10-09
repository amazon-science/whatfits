<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
-->

# NYC 311 January 2026 Experiment

This experiment predicts whether a New York City 311 service request is closed
within 72 hours using only request-creation metadata.

## Data Source

Source: NYC Open Data, "311 Service Requests from 2020 to Present", dataset ID
`erm2-nwe9`, maintained by the City of New York / NYC Office of Technology and
Innovation.

The staged extract filters:

```text
created_date >= 2026-01-01T00:00:00
created_date <  2026-02-01T00:00:00
```

Rows are ordered by `created_date, unique_key`, and only the creation-time fields
used by the experiment plus `closed_date` for label construction are downloaded.

## Stage The Raw CSV

From this directory:

```bash
uv run python download_data.py
uv run python prepare.py
```

The downloader writes:

```text
~/.cache/autoresearch-nyc311/raw_nyc_311_jan2026.csv
```

`prepare.py` then builds and caches the fixed train/validation/holdout split at:

```text
~/.cache/autoresearch-nyc311/data_nyc311.pkl
```

For offline reproduction from an already-downloaded CSV:

```bash
uv run python download_data.py --source-file /path/to/raw_nyc_311_jan2026.csv
uv run python prepare.py
```

The raw CSV and derived pickle are intentionally not committed to this
repository.
