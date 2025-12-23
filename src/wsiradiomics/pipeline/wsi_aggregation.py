# -*- coding: utf-8 -*-
# @Time    : 2025/12/23 10:13
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: wsi_aggregation.py
# @Project : WSIRadiomics

from __future__ import annotations

from typing import Dict, List, Any
import numpy as np

from wsiradiomics.aggregation.functions import (
    SUPPORTED_AGG_FUNCS,
    apply_agg,
)

_META_KEYS = {"__cell_type__", "__feature_id__"}


def compute_wsi_features(cell_rows: List[Dict[str, float]], cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Aggregate per-cell feature rows into WSI-level feature dict.

    cfg expected:
      cfg["aggregation"]["functions"] : list[str]
      cfg["aggregation"].get("by_cell_type") : list[str] | None

    Output naming convention (stable, no mapping):
      Always prefix group:
        {Group}__{FeatureName}__{AggFunc}

      Group is:
        - Always includes "All"
        - If by_cell_type is provided: includes only the listed types (no "Other")
        - If by_cell_type is None: only "All"
    """
    agg_cfg = (cfg or {}).get("aggregation", {}) or {}

    funcs = agg_cfg.get("functions", []) or []
    funcs = [f for f in funcs if f in SUPPORTED_AGG_FUNCS]
    if not funcs:
        return {}

    by_types = agg_cfg.get("by_cell_type", None)
    if isinstance(by_types, list) and len(by_types) == 0:
        by_types = None

    feature_names = _collect_feature_names(cell_rows)

    # -------------------------
    # Group rows
    # -------------------------
    groups: Dict[str, List[Dict[str, float]]] = {"All": cell_rows}

    if by_types is not None:
        # Only create groups listed in cfg; ignore everything else.
        type_set = [str(t) for t in by_types]
        for t in type_set:
            groups[t] = []

        for r in cell_rows:
            ct = r.get("__cell_type__", None)
            if ct in groups:  # only keep whitelisted types
                groups[ct].append(r)

    # -------------------------
    # Aggregate
    # -------------------------
    out: Dict[str, float] = {}

    for group_name, rows in groups.items():
        for feat in feature_names:
            vals = _values_for_feature(rows, feat)

            for func in funcs:
                v = apply_agg(func, vals)
                key = f"{group_name}__{feat}__{func}"
                out[key] = float(v)

    return out


# -------------------------
# internals
# -------------------------

def _collect_feature_names(rows: List[Dict[str, float]]) -> List[str]:
    names = set()
    for r in rows or []:
        for k in (r or {}).keys():
            if k in _META_KEYS:
                continue
            names.add(k)
    return sorted(names)


def _values_for_feature(rows: List[Dict[str, float]], feat: str) -> np.ndarray:
    vals: List[float] = []
    for r in rows or []:
        v = (r or {}).get(feat, np.nan)
        try:
            fv = float(v)
        except Exception:
            fv = np.nan
        vals.append(fv)
    return np.asarray(vals, dtype=np.float64)


