# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 15:01
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: firstorder.py
# @Project : WSIRadiomics

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np


# Names MUST match params.yaml / cfg print
FEATURES_ALL = [
    "Energy",
    "TotalEnergy",
    "Entropy",
    "Minimum",
    "Percentile10",
    "Percentile90",
    "Maximum",
    "Mean",
    "Median",
    "InterquartileRange",
    "Range",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
    "RootMeanSquared",
    "StandardDeviation",
    "Skewness",
    "Kurtosis",
    "Variance",
    "Uniformity",
]

_FEATURE_DEPENDENCIES = {
    "Energy": {"xc"},
    "TotalEnergy": {"xc"},
    "RootMeanSquared": {"xc"},

    "Entropy": {"hist"},
    "Uniformity": {"hist"},

    "Minimum": {"minmax"},
    "Maximum": {"minmax"},
    "Range": {"minmax"},

    "Mean": {"mean"},
    "Median": {"percentiles"},
    "Percentile10": {"percentiles"},
    "Percentile90": {"percentiles"},
    "InterquartileRange": {"percentiles"},

    "MeanAbsoluteDeviation": {"mean"},
    "RobustMeanAbsoluteDeviation": {"percentiles"},

    "Variance": {"mean"},
    "StandardDeviation": {"mean"},
    "Skewness": {"mean", "moments34"},
    "Kurtosis": {"mean", "moments34"},
}


@dataclass(frozen=True)
class FirstOrderParams:
    c: float = 0.0
    voxel_volume: float = 1.0
    eps: float = 1e-12
    bin_width: Optional[float] = None
    n_bins: int = 64


def compute(
    X: Any,
    *,
    selection: Optional[Sequence[str]] = None,
    params: Optional[FirstOrderParams] = None,
) -> Dict[str, float]:

    p = params or FirstOrderParams()
    names = FEATURES_ALL if selection is None else list(selection)

    x = _to_1d(X)
    if x.size == 0:
        return {k: float("nan") for k in names}

    needed = _required_computations(names)

    # ---- Lazy intermediates ----
    minmax = None
    mean = None
    percentiles = None
    xc = None
    hist = None
    moments34 = None

    if "minmax" in needed:
        minmax = (float(x.min()), float(x.max()))

    if "mean" in needed:
        mean = float(x.mean())

    if "percentiles" in needed:
        percentiles = {
            10: float(np.percentile(x, 10)),
            25: float(np.percentile(x, 25)),
            50: float(np.percentile(x, 50)),
            75: float(np.percentile(x, 75)),
            90: float(np.percentile(x, 90)),
        }

    if "xc" in needed:
        xc = x + p.c

    if "hist" in needed:
        hist = _probabilities(x, p)

    if "moments34" in needed:
        if mean is None:
            mean = float(x.mean())
        var = float(np.mean((x - mean) ** 2))  # population variance per your formula
        if var > 0:
            m3 = float(np.mean((x - mean) ** 3))
            m4 = float(np.mean((x - mean) ** 4))
            moments34 = (var, m3, m4)
        else:
            moments34 = (0.0, np.nan, np.nan)

    # ---- Assemble output ----
    out: Dict[str, float] = {}

    for name in names:
        if name == "Energy":
            out[name] = float(np.sum(xc ** 2))
        elif name == "TotalEnergy":
            out[name] = float(p.voxel_volume) * float(np.sum(xc ** 2))
        elif name == "RootMeanSquared":
            out[name] = float(np.sqrt(np.mean(xc ** 2)))

        elif name == "Entropy":
            out[name] = float(-np.sum(hist * np.log2(hist + float(p.eps))))
        elif name == "Uniformity":
            out[name] = float(np.sum(hist ** 2))

        elif name == "Minimum":
            out[name] = float(minmax[0])
        elif name == "Maximum":
            out[name] = float(minmax[1])
        elif name == "Range":
            out[name] = float(minmax[1] - minmax[0])

        elif name == "Mean":
            out[name] = float(mean)
        elif name == "Median":
            out[name] = float(percentiles[50])
        elif name == "Percentile10":
            out[name] = float(percentiles[10])
        elif name == "Percentile90":
            out[name] = float(percentiles[90])
        elif name == "InterquartileRange":
            out[name] = float(percentiles[75] - percentiles[25])

        elif name == "MeanAbsoluteDeviation":
            out[name] = float(np.mean(np.abs(x - mean)))

        elif name == "RobustMeanAbsoluteDeviation":
            lo, hi = float(percentiles[10]), float(percentiles[90])
            x_ = x[(x >= lo) & (x <= hi)]
            out[name] = float(np.mean(np.abs(x_ - x_.mean()))) if x_.size else float("nan")

        elif name == "Variance":
            out[name] = float(moments34[0])
        elif name == "StandardDeviation":
            out[name] = float(np.sqrt(moments34[0]))
        elif name == "Skewness":
            var, m3, _ = moments34
            out[name] = float(m3 / (var ** 1.5)) if var > 0 else float("nan")
        elif name == "Kurtosis":
            var, _, m4 = moments34
            out[name] = float(m4 / (var ** 2)) if var > 0 else float("nan")
        else:
            # unknown feature name requested
            out[name] = float("nan")

    return out


# -------------------------
# helpers
# -------------------------

def _required_computations(selection: Sequence[str]) -> set[str]:
    req = set()
    for n in selection:
        req |= _FEATURE_DEPENDENCIES.get(n, set())
    return req


def _to_1d(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _probabilities(x: np.ndarray, p: FirstOrderParams) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.array([1.0], dtype=np.float64)

    if p.bin_width is not None and float(p.bin_width) > 0:
        bw = float(p.bin_width)
        bins = np.arange(mn, mx + bw, bw, dtype=np.float64)
        if bins.size < 2:
            bins = np.linspace(mn, mx, int(p.n_bins) + 1, dtype=np.float64)
    else:
        bins = np.linspace(mn, mx, int(p.n_bins) + 1, dtype=np.float64)

    hist, _ = np.histogram(x, bins=bins)
    s = hist.sum()
    if s <= 0:
        return np.array([1.0], dtype=np.float64)
    return (hist.astype(np.float64) / float(s)).astype(np.float64)


if __name__ == "__main__":
    """
    Minimal usage example for first-order features (names match params.yaml).
    """
    from pprint import pprint

    rng = np.random.default_rng(seed=42)
    X = rng.normal(loc=120, scale=15, size=500).astype(np.float64)
    X = np.clip(X, 0, 255)

    print(f"ROI size (N_p): {X.size}")
    print(f"Intensity range: [{X.min():.2f}, {X.max():.2f}]")

    selection = [
        "Mean",
        "StandardDeviation",
        "Entropy",
        "Energy",
        "Skewness",
        "Kurtosis",
    ]

    params = FirstOrderParams(
        c=0.0,
        voxel_volume=1.0,
        bin_width=None,
        n_bins=32,
        eps=1e-12,
    )

    feats = compute(X, selection=selection, params=params)
    print("\nComputed subset:")
    pprint(feats, sort_dicts=False)

    all_feats = compute(X, selection=None, params=params)
    print("\nComputed ALL:")
    print(f"Number of features: {len(all_feats)}")
    pprint(all_feats, sort_dicts=False)

