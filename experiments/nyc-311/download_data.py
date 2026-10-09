#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Download or stage the January 2026 NYC 311 raw CSV.

The experiment's prepare.py expects the raw file at:
    ~/.cache/autoresearch-nyc311/raw_nyc_311_jan2026.csv

By default this script downloads the public NYC Open Data extract through the
Socrata SODA CSV API. For offline/local reuse, pass --source-file PATH to copy
an already-downloaded CSV to the cache location.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CACHE_DIR = Path.home() / ".cache" / "autoresearch-nyc311"
RAW_CSV = CACHE_DIR / "raw_nyc_311_jan2026.csv"

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
WHERE = "created_date >= '2026-01-01T00:00:00' and created_date < '2026-02-01T00:00:00'"
ORDER = "created_date,unique_key"
SELECT_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "city",
    "borough",
    "community_board",
    "council_district",
    "police_precinct",
    "open_data_channel_type",
    "latitude",
    "longitude",
]
USER_AGENT = "whatfits-nyc311-repro/1.0"


def socrata_count() -> int:
    url = BASE_URL.replace(".csv", ".json") + "?" + urlencode(
        {"$select": "count(*)", "$where": WHERE}
    )
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    return int(json.loads(body)[0]["count"])


def download(output: Path, page_size: int) -> None:
    total = socrata_count()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fout:
        first_page = True
        for offset in range(0, total, page_size):
            params = {
                "$select": ",".join(SELECT_COLUMNS),
                "$where": WHERE,
                "$order": ORDER,
                "$limit": page_size,
                "$offset": offset,
            }
            req = Request(BASE_URL + "?" + urlencode(params), headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=120) as resp:
                chunk = resp.read()
            if first_page:
                fout.write(chunk)
                first_page = False
            else:
                newline = chunk.find(b"\n")
                fout.write(chunk[newline + 1 :] if newline >= 0 else b"")
            print(f"{output}: fetched {min(offset + page_size, total)} / {total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the NYC 311 January 2026 CSV")
    parser.add_argument("--output", type=Path, default=RAW_CSV, help=f"Output CSV (default: {RAW_CSV})")
    parser.add_argument("--source-file", type=Path, help="Copy an existing raw CSV instead of downloading")
    parser.add_argument("--page-size", type=int, default=50_000)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Already staged: {args.output}")
        print("Use --force to overwrite.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.source_file:
        shutil.copyfile(args.source_file, args.output)
        print(f"Copied {args.source_file} -> {args.output}")
    else:
        download(args.output, args.page_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
