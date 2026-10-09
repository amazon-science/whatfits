#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Shared codelength helpers for the experimental codebook pipeline."""

from __future__ import annotations

import math
from typing import Any


EPS = 1e-12


def _clamp_probability(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def _log2_inv(p: float) -> float:
    return -math.log2(_clamp_probability(p))


def _value_bits(menu: dict[str, Any], value: Any, escape_bits: float) -> tuple[float, dict[str, Any]]:
    kind = menu.get("type")

    if kind == "categorical":
        options = menu.get("options", {})
        key = value if isinstance(value, str) else str(value)
        if key in options:
            bits = _log2_inv(float(options[key]))
            return bits, {"value": bits, "option": key}
        bits = float(menu.get("unknown_option_bits", escape_bits))
        return bits, {"value": bits, "unknown_option": key}

    if kind == "integer":
        lo = int(menu.get("min", 0))
        hi = int(menu.get("max", lo))
        n_values = max(1, hi - lo + 1)
        bits = math.log2(n_values) if n_values > 1 else 0.0
        return bits, {"value": bits, "range": [lo, hi]}

    if kind == "continuous":
        bits = float(menu.get("precision_bits", menu.get("value_bits", 8.0)))
        return bits, {"value": bits, "precision_bits": bits}

    bits = float(menu.get("unknown_type_bits", escape_bits))
    return bits, {"value": bits, "unknown_type": kind}


def compute_sparse_codelength(
    codebook: dict[str, Any], selections: dict[str, Any]
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Compute spike-and-slab codelength for on-codebook selections.

    Every menu incurs either a skip cost, -log2(1 - pi), or a selected cost,
    -log2(pi) plus the value cost for that menu.
    """
    escape_bits = float(codebook.get("escape_penalty_bits", 12.0))
    menu_by_name = {m["name"]: m for m in codebook.get("menus", [])}
    total = 0.0
    breakdown: dict[str, dict[str, Any]] = {}

    for name, menu in menu_by_name.items():
        pi = _clamp_probability(float(menu.get("selection_probability", 0.5)))
        if name in selections:
            select_bits = _log2_inv(pi)
            val_bits, val_info = _value_bits(menu, selections[name], escape_bits)
            item_total = select_bits + val_bits
            info = {"select": select_bits, "total": item_total}
            info.update(val_info)
        else:
            item_total = _log2_inv(1.0 - pi)
            info = {"skip": item_total, "total": item_total}
        breakdown[name] = info
        total += item_total

    for name in selections:
        if name not in menu_by_name:
            breakdown[name] = {"escape": True, "total": escape_bits}
            total += escape_bits

    return total, breakdown


def mdl_bound(bits: float, n_val: int, delta: float = 0.05) -> float:
    """Return the bounded-differences MDL deviation radius."""
    if n_val <= 0:
        raise ValueError("n_val must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    bits = max(0.0, float(bits))
    return math.sqrt((bits * math.log(2.0) + math.log(1.0 / delta)) / (2.0 * n_val))
