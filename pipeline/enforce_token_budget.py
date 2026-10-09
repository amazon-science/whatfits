#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Check whether a compressed prompt fits a token budget.

This intentionally uses a conservative tokenizer-free estimate so the public
pipeline has no extra runtime dependency. It is used as a guardrail and report,
not as the authoritative compressor; the compressor itself still tries to keep
the prompt below the requested budget.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Return a stable approximate token count for prompt text."""
    return len(TOKEN_RE.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a compressed prompt token budget")
    parser.add_argument("prompt", type=Path, help="Path to compressed_prompt.md")
    parser.add_argument("budget", type=int, help="Maximum token budget")
    args = parser.parse_args()

    if not args.prompt.exists():
        print(f"ERROR: prompt not found: {args.prompt}", file=sys.stderr)
        return 2

    n_tokens = estimate_tokens(args.prompt.read_text())
    status = "OK" if n_tokens <= args.budget else "OVER"
    print(f"Token budget check: {n_tokens} estimated tokens / {args.budget} ({status})")
    return 0 if n_tokens <= args.budget else 1


if __name__ == "__main__":
    raise SystemExit(main())
