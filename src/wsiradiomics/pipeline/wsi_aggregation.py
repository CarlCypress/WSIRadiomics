# -*- coding: utf-8 -*-
# @Time    : 2025/12/23 10:13
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: wsi_aggregation.py
# @Project : WSIRadiomics

from __future__ import annotations

from typing import Dict, List, Any
import logging
import time
import numpy as np

from wsiradiomics.aggregation.functions import (
    SUPPORTED_AGG_FUNCS,
    apply_agg,
)

logger = logging.getLogger(__name__)

_META_KEYS = {"__cell_type__", "__feature_id__"}


def compute_wsi_features(cell_rows: List[Dict[str, float]], cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Aggregate per-cell feature rows into WSI-level feature dict.

    cfg expected:
      cfg["aggregation"]["functions"] : list[str]
      cfg["aggregation"].get("by_cell_type") : list[str] | None

    Output naming convention:
      {Group}__{FeatureName}__{AggFunc}
    """
    t0 = time.time()

    n_cells = len(cell_rows or [])
    logger.info("WSI aggregation started: n_cells=%d", n_cells)

    agg_cfg = (cfg or {}).get("aggregation", {}) or {}

    funcs = agg_cfg.get("functions", []) or []
    funcs = [f for f in funcs if f in SUPPORTED_AGG_FUNCS]

    if not funcs:
        logger.warning("No valid aggregation functions enabled; returning empty WSI features.")
        return {}

    logger.info("Aggregation functions: %s", funcs)

    by_types = agg_cfg.get("by_cell_type", None)
    if isinstance(by_types, list) and len(by_types) == 0:
        by_types = None

    if by_types is None:
        logger.info("Aggregation mode: All cells only")
    else:
        logger.info("Aggregation mode: by_cell_type=%s (plus All)", by_types)

    feature_names = _collect_feature_names(cell_rows)
    if not feature_names:
        logger.warning("No feature names found in cell rows; returning empty WSI features.")
        return {}

    logger.info("Number of cell-level features to aggregate: %d", len(feature_names))

    # -------------------------
    # Group rows
    # -------------------------
    groups: Dict[str, List[Dict[str, float]]] = {"All": cell_rows}

    if by_types is not None:
        type_set = [str(t) for t in by_types]
        for t in type_set:
            groups[t] = []

        for r in cell_rows or []:
            ct = r.get("__cell_type__", None)
            if ct in groups:
                groups[ct].append(r)

    for g, rows in groups.items():
        logger.info("Group '%s': n_cells=%d", g, len(rows))

    # -------------------------
    # Aggregate
    # -------------------------
    out: Dict[str, float] = {}

    for group_name, rows in groups.items():
        for feat in feature_names:
            vals = _values_for_feature(rows, feat)

            if logger.isEnabledFor(logging.DEBUG):
                finite = np.isfinite(vals).sum()
                logger.debug(
                    "Group=%s Feature=%s: n=%d finite=%d",
                    group_name, feat, vals.size, finite
                )

            for func in funcs:
                v = apply_agg(func, vals)
                key = f"{group_name}__{feat}__{func}"
                out[key] = float(v)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed %s = %s", key, v)

    logger.info(
        "WSI aggregation finished: n_features=%d, elapsed=%.2fs",
        len(out), time.time() - t0
    )

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