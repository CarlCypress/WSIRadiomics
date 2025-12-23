# -*- coding: utf-8 -*-
# @Time    : 2025/12/23 16:13
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: functions.py
# @Project : WSIRadiomics

from __future__ import annotations

import numpy as np

SUPPORTED_AGG_FUNCS = {
    "Mean",
    "Median",
    "StandardDeviation",
    "InterquartileRange",
    "Skewness",
    "Kurtosis",
}


def apply_agg(func: str, x: np.ndarray) -> float:
    """
    Apply one aggregation function to 1D array x.
    NaN-aware behavior:
      - mean/median/std/percentiles ignore NaNs
      - skewness/kurtosis computed on finite values only
    """
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return float("nan")

    if func == "Mean":
        return float(np.nanmean(x))
    if func == "Median":
        return float(np.nanmedian(x))
    if func == "StandardDeviation":
        # sample std (N-1), consistent with your aggregation/readme.md
        return float(np.nanstd(x, ddof=1))
    if func == "InterquartileRange":
        q75 = float(np.nanpercentile(x, 75))
        q25 = float(np.nanpercentile(x, 25))
        return float(q75 - q25)
    if func == "Skewness":
        return float(nan_skewness(x))
    if func == "Kurtosis":
        # your doc uses Excess Kurtosis but names it "Kurtosis"
        return float(nan_excess_kurtosis(x))

    return float("nan")


def nan_skewness(x: np.ndarray) -> float:
    """
    Match your aggregation/readme.md Skewness formula:
      sqrt(N(N-1))/(N-2) * ( (1/N)*sum((xi-m)^3) ) / ( ( (1/N)*sum((xi-m)^2) )^(3/2) )
      (N>=3)
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


def nan_excess_kurtosis(x: np.ndarray) -> float:
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
    return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0))

