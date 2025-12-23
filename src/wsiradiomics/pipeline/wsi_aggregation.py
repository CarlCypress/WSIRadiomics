# -*- coding: utf-8 -*-
# @Time    : 2025/12/23 10:13
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: wsi_aggregation.py
# @Project : WSIRadiomics

# -*- coding: utf-8 -*-
# @FileName: wsi_aggregation.py
# @Project : WSIRadiomics

from __future__ import annotations

from typing import Dict, List, Any, Iterable, Optional
import numpy as np


_SUPPORTED_FUNCS = {
    "Mean",
    "Median",
    "StandardDeviation",
    "InterquartileRange",
    "Skewness",
    "Kurtosis",
}


_META_KEYS = {"__cell_type__", "__feature_id__"}


def compute_wsi_features(cell_rows: List[Dict[str, float]], cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Aggregate per-cell feature rows into WSI-level feature dict.

    cfg expected:
      cfg["aggregation"]["functions"] : list[str]
      cfg["aggregation"].get("by_cell_type") : list[str] | None

    Output naming convention (stable, no mapping):
      - If no by_cell_type:   {FeatureName}__{AggFunc}
      - If by_cell_type:      {CellType}__{FeatureName}__{AggFunc}
        and additionally always provide "All" group:
                              All__{FeatureName}__{AggFunc}

    Notes:
      - Skips metadata keys: __cell_type__, __feature_id__
      - Ignores non-numeric values
      - Uses NaN-aware statistics; if no valid values, outputs NaN
    """
    agg_cfg = (cfg or {}).get("aggregation", {}) or {}

    funcs = agg_cfg.get("functions", []) or []
    funcs = [f for f in funcs if f in _SUPPORTED_FUNCS]
    if not funcs:
        # 如果 cfg 里没给合法 functions，直接返回空（也可以 raise）
        return {}

    by_types = agg_cfg.get("by_cell_type", None)
    if isinstance(by_types, list) and len(by_types) == 0:
        by_types = None

    # 哪些 feature 字段需要聚合？
    feature_names = _collect_feature_names(cell_rows)

    # 按 cell_type 分组
    groups: Dict[str, List[Dict[str, float]]] = {}

    # 永远提供 All 组（后面命名也统一）
    groups["All"] = cell_rows

    if by_types is not None:
        # 只对 cfg 列出的类型建组；没在列表里的统一落到 "Other"
        type_set = set(str(t) for t in by_types)
        for t in type_set:
            groups[t] = []

        groups["Other"] = []

        for r in cell_rows:
            ct = str(r.get("__cell_type__", "Unknown"))
            if ct in type_set:
                groups[ct].append(r)
            else:
                groups["Other"].append(r)
    # 如果 by_types is None，只算 All 组即可（groups 已经有 All）

    out: Dict[str, float] = {}

    # 对每个组，每个 feature，算每个 aggregation
    for group_name, rows in groups.items():
        # 如果 by_cell_type=None，只算 All；此时 group_name="All"
        if by_types is None and group_name != "All":
            continue

        # 提取每个 feature 的数值向量
        for feat in feature_names:
            vals = _values_for_feature(rows, feat)  # np.ndarray float64
            # 这里允许全 NaN；各统计会输出 NaN

            for func in funcs:
                v = _apply_agg(func, vals)
                key = f"{group_name}__{feat}__{func}"
                out[key] = float(v)

    return out


# -------------------------
# internals
# -------------------------

def _collect_feature_names(rows: List[Dict[str, float]]) -> List[str]:
    names = set()
    for r in rows:
        for k in r.keys():
            if k in _META_KEYS:
                continue
            names.add(k)
    return sorted(names)


def _values_for_feature(rows: List[Dict[str, float]], feat: str) -> np.ndarray:
    vals: List[float] = []
    for r in rows:
        v = r.get(feat, np.nan)
        try:
            fv = float(v)
        except Exception:
            fv = np.nan
        vals.append(fv)
    arr = np.asarray(vals, dtype=np.float64)
    # 保留 NaN，让 nan-aware 聚合处理
    return arr


def _apply_agg(func: str, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return float("nan")

    if func == "Mean":
        return float(np.nanmean(x))
    if func == "Median":
        return float(np.nanmedian(x))
    if func == "StandardDeviation":
        # 与你 aggregation/readme.md 一致：样本标准差 (N-1)
        # 注意：nanstd 默认 ddof=0，需要 ddof=1
        return float(np.nanstd(x, ddof=1))
    if func == "InterquartileRange":
        q75 = float(np.nanpercentile(x, 75))
        q25 = float(np.nanpercentile(x, 25))
        return float(q75 - q25)
    if func == "Skewness":
        return float(_nan_skewness(x))
    if func == "Kurtosis":
        # 你 aggregation/readme.md 里是 Excess Kurtosis 的公式名叫 Kurtosis
        return float(_nan_excess_kurtosis(x))

    return float("nan")


def _nan_skewness(x: np.ndarray) -> float:
    """
    Match your aggregation/readme.md Skewness formula:
      sqrt(N(N-1))/(N-2) * ( (1/N)*sum((xi-m)^3) ) / ( ( (1/N)*sum((xi-m)^2) )^(3/2) )
      (N>=3)
    We compute it on finite values only.
    """
    z = x[np.isfinite(x)]
    n = z.size
    if n < 3:
        return float("nan")

    m = float(np.mean(z))
    m2 = float(np.mean((z - m) ** 2))
    if m2 <= 0:
        return float("nan")

    m3 = float(np.mean((z - m) ** 3))
    g1 = m3 / (m2 ** 1.5)

    corr = np.sqrt(n * (n - 1)) / (n - 2)
    return float(corr * g1)


def _nan_excess_kurtosis(x: np.ndarray) -> float:
    """
    Match your aggregation/readme.md Excess Kurtosis formula (N>=4).
    """
    z = x[np.isfinite(x)]
    n = z.size
    if n < 4:
        return float("nan")

    m = float(np.mean(z))
    m2 = float(np.mean((z - m) ** 2))
    if m2 <= 0:
        return float("nan")

    m4 = float(np.mean((z - m) ** 4))
    g2 = (m4 / (m2 ** 2)) - 3.0  # excess

    # Bias correction per your doc:
    # (n-1)/((n-2)(n-3)) * [ (n+1)*g2 + 6 ]   (since g2 already excess)
    return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0))


