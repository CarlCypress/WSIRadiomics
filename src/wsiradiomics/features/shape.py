# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 15:01
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: shape.py
# @Project : WSIRadiomics

"""
wsiradiomics.features.shape

Shape features for 2D ROI polygons (cell boundaries from GeoJSON).

Curvature notes:
- The docs mention principal curvatures (k1, k2), which are surface (3D mesh) concepts.
- For 2D polygons, we implement a 2D boundary-curvature substitute using planar curve curvature κ.

2D boundary curvature κ for a parametric curve (x(t), y(t)):
  κ = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

We compute κ on a resampled (approximately arc-length uniform) closed contour.
Then:
  MeanCurvature      := mean(|κ|)
  GaussianCurvature  := mean(κ^2)   (2D proxy: curvature energy; not 3D k1*k2)
  MaximumCurvature   := max(κ)
  MinimumCurvature   := min(κ)

This module is selection-driven: compute only requested features and needed intermediates.

IMPORTANT:
- Feature names aligned with params.yaml / cfg print (CamelCase), no name mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


FEATURES_ALL: List[str] = [
    "MeshSurface",
    "PixelSurface",
    "Perimeter",
    "PerimetertoSurfaceRatio",
    "Sphericity",
    "SphericalDisproportion",
    "MaximumDiameter",
    "MajorAxisLength",
    "MinorAxisLength",
    "Elongation",
    "MeanCurvature",
    "GaussianCurvature",
    "MaximumCurvature",
    "MinimumCurvature",
]

_FEATURE_DEPENDENCIES = {
    "MeshSurface": {"area"},
    "PixelSurface": {"area"},
    "Perimeter": {"perimeter"},

    "PerimetertoSurfaceRatio": {"area", "perimeter"},
    "Sphericity": {"area", "perimeter"},
    "SphericalDisproportion": {"area", "perimeter"},

    "MaximumDiameter": {"max_diameter"},
    "MajorAxisLength": {"pca_axes"},
    "MinorAxisLength": {"pca_axes"},
    "Elongation": {"pca_axes"},

    # curvature depends on boundary curvature computation
    "MeanCurvature": {"curvature"},
    "GaussianCurvature": {"curvature"},
    "MaximumCurvature": {"curvature"},
    "MinimumCurvature": {"curvature"},
}


@dataclass(frozen=True)
class ShapeParams:
    """
    Parameters for 2D shape computation.

    pixel_spacing:
      (sx, sy) physical size per pixel. If None, treat coords as unit pixels.

    max_diameter_mode:
      "vertices" => max pairwise distance among polygon vertices (O(n^2))
      "bbox"     => bbox diagonal (fast approximation)

    curvature_n_samples:
      Number of resampled points along the closed boundary for curvature estimation.
      (>= 64 recommended for stable curvature)

    curvature_smooth_sigma:
      Gaussian smoothing sigma (in sample index units) applied to resampled contour
      before derivative-based curvature. 0 disables.
    """
    pixel_spacing: Optional[Tuple[float, float]] = None
    max_diameter_mode: str = "vertices"

    curvature_n_samples: int = 128
    curvature_smooth_sigma: float = 1.5


def compute(
    polygon_xy: Any,
    *,
    selection: Optional[Sequence[str]] = None,
    params: Optional[ShapeParams] = None,
) -> Dict[str, float]:
    sp = params or ShapeParams()
    names = FEATURES_ALL if selection is None else list(selection)

    pts = _to_points(polygon_xy)
    if pts.shape[0] < 3:
        return {k: float("nan") for k in names}

    needed = _required_computations(names)
    pts_s = _apply_spacing(pts, sp.pixel_spacing)

    # lazy intermediates
    area: Optional[float] = None
    perim: Optional[float] = None
    max_d: Optional[float] = None
    pca_axes: Optional[Tuple[float, float, float]] = None  # (major, minor, elong)
    curv_stats: Optional[Tuple[float, float, float, float]] = None  # (mean_abs, mean_k2, kmax, kmin)

    if "area" in needed:
        area = float(_polygon_area(pts_s))

    if "perimeter" in needed:
        perim = float(_polygon_perimeter(pts_s))

    if "max_diameter" in needed:
        if sp.max_diameter_mode == "bbox":
            max_d = float(_bbox_diagonal(pts_s))
        else:
            max_d = float(_max_pairwise_distance(pts_s))

    if "pca_axes" in needed:
        pca_axes = _pca_axes_lengths(pts_s)

    if "curvature" in needed:
        curv_stats = _curvature_features_2d(
            pts_s,
            n_samples=int(sp.curvature_n_samples),
            smooth_sigma=float(sp.curvature_smooth_sigma),
        )

    out: Dict[str, float] = {}
    for name in names:
        if name == "MeshSurface":
            out[name] = float(area) if area is not None else float("nan")
        elif name == "PixelSurface":
            out[name] = float(area) if area is not None else float("nan")

        elif name == "Perimeter":
            out[name] = float(perim) if perim is not None else float("nan")

        elif name == "PerimetertoSurfaceRatio":
            out[name] = _safe_div(perim, area)

        elif name == "Sphericity":
            if area is None or perim is None or area <= 0 or perim <= 0:
                out[name] = float("nan")
            else:
                out[name] = float((2.0 * np.pi * np.sqrt(area / np.pi)) / perim)

        elif name == "SphericalDisproportion":
            if area is None or perim is None or area <= 0 or perim <= 0:
                out[name] = float("nan")
            else:
                denom = (2.0 * np.pi * np.sqrt(area / np.pi))
                out[name] = float(perim / denom) if denom > 0 else float("nan")

        elif name == "MaximumDiameter":
            out[name] = float(max_d) if max_d is not None else float("nan")

        elif name == "MajorAxisLength":
            out[name] = float(pca_axes[0]) if pca_axes is not None else float("nan")
        elif name == "MinorAxisLength":
            out[name] = float(pca_axes[1]) if pca_axes is not None else float("nan")
        elif name == "Elongation":
            out[name] = float(pca_axes[2]) if pca_axes is not None else float("nan")

        elif name == "MeanCurvature":
            out[name] = float(curv_stats[0]) if curv_stats is not None else float("nan")
        elif name == "GaussianCurvature":
            out[name] = float(curv_stats[1]) if curv_stats is not None else float("nan")
        elif name == "MaximumCurvature":
            out[name] = float(curv_stats[2]) if curv_stats is not None else float("nan")
        elif name == "MinimumCurvature":
            out[name] = float(curv_stats[3]) if curv_stats is not None else float("nan")

        else:
            out[name] = float("nan")

    return out


# -------------------------
# helpers
# -------------------------

def _required_computations(selection: Sequence[str]) -> set[str]:
    req: set[str] = set()
    for n in selection:
        req |= _FEATURE_DEPENDENCIES.get(n, set())
    return req


def _to_points(polygon_xy: Any) -> np.ndarray:
    """
    Convert [(x,y), ...] to Nx2 float64 array.
    Drops invalid points and removes duplicated closing point if present.
    """
    pts: List[Tuple[float, float]] = []
    for p in polygon_xy if polygon_xy is not None else []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                pts.append((float(x), float(y)))

    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]

    return np.asarray(pts, dtype=np.float64)


def _apply_spacing(pts: np.ndarray, spacing: Optional[Tuple[float, float]]) -> np.ndarray:
    if spacing is None:
        return pts
    sx, sy = float(spacing[0]), float(spacing[1])
    out = pts.copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula, absolute area."""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_perimeter(pts: np.ndarray) -> float:
    """Sum of edge lengths."""
    d = np.roll(pts, -1, axis=0) - pts
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _bbox_diagonal(pts: np.ndarray) -> float:
    minx, miny = np.min(pts[:, 0]), np.min(pts[:, 1])
    maxx, maxy = np.max(pts[:, 0]), np.max(pts[:, 1])
    dx = float(maxx - minx)
    dy = float(maxy - miny)
    return float(np.sqrt(dx * dx + dy * dy))


def _max_pairwise_distance(pts: np.ndarray) -> float:
    n = pts.shape[0]
    if n <= 1:
        return 0.0
    maxd2 = 0.0
    for i in range(n):
        d = pts[i] - pts
        d2 = np.sum(d * d, axis=1)
        m = float(np.max(d2))
        if m > maxd2:
            maxd2 = m
    return float(np.sqrt(maxd2))


def _pca_axes_lengths(pts: np.ndarray) -> Tuple[float, float, float]:
    if pts.shape[0] < 3:
        return float("nan"), float("nan"), float("nan")

    c = pts - np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(c.T, ddof=0)

    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0.0))
        if eigvals.size < 2:
            return float("nan"), float("nan"), float("nan")
        lam_minor = float(eigvals[-2])
        lam_major = float(eigvals[-1])
    except Exception:
        return float("nan"), float("nan"), float("nan")

    if not np.isfinite(lam_major) or lam_major <= 0:
        return float("nan"), float("nan"), float("nan")

    major = float(4.0 * np.sqrt(lam_major))
    minor = float(4.0 * np.sqrt(max(lam_minor, 0.0)))
    elong = float(np.sqrt(lam_minor / lam_major)) if lam_minor >= 0 else float("nan")
    return major, minor, elong


def _safe_div(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return float("nan")
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a / b)


# -------------------------
# curvature (2D boundary substitute)
# -------------------------

def _curvature_features_2d(
    pts: np.ndarray,
    *,
    n_samples: int = 128,
    smooth_sigma: float = 1.5,
) -> Tuple[float, float, float, float]:
    """
    Compute curvature statistics from a 2D closed contour.

    Returns:
      (mean_abs_kappa, mean_kappa2, max_kappa, min_kappa)
    """
    n = max(32, int(n_samples))
    contour = _resample_closed_polyline(pts, n)

    if smooth_sigma and smooth_sigma > 0:
        contour = _gaussian_smooth_circular(contour, float(smooth_sigma))

    x = contour[:, 0]
    y = contour[:, 1]

    # Central differences on a circular sequence
    x1 = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
    y1 = 0.5 * (np.roll(y, -1) - np.roll(y, 1))
    x2 = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
    y2 = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)

    denom = (x1 * x1 + y1 * y1) ** 1.5
    denom = np.where(denom > 0, denom, np.nan)

    kappa = (x1 * y2 - y1 * x2) / denom
    kappa = kappa[np.isfinite(kappa)]
    if kappa.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    mean_abs = float(np.mean(np.abs(kappa)))
    mean_k2 = float(np.mean(kappa ** 2))
    kmax = float(np.max(kappa))
    kmin = float(np.min(kappa))
    return mean_abs, mean_k2, kmax, kmin


def _resample_closed_polyline(pts: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Resample a (possibly not explicitly closed) polyline to n_samples points along arc length,
    treating it as a closed contour.
    """
    if pts.shape[0] < 3:
        return pts

    p = pts
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])

    seg = p[1:] - p[:-1]
    seglen = np.sqrt(np.sum(seg * seg, axis=1))
    total = float(np.sum(seglen))
    if total <= 0:
        return pts

    s = np.concatenate([[0.0], np.cumsum(seglen)])
    t = np.linspace(0.0, total, int(n_samples) + 1)[:-1]  # exclude endpoint

    out = np.zeros((int(n_samples), 2), dtype=np.float64)
    j = 0
    for i, ti in enumerate(t):
        while j + 1 < s.size and s[j + 1] < ti:
            j += 1
        if j + 1 >= s.size:
            out[i] = p[-2]
            continue
        ds = s[j + 1] - s[j]
        a = 0.0 if ds <= 0 else (ti - s[j]) / ds
        out[i] = (1 - a) * p[j] + a * p[j + 1]
    return out


def _gaussian_smooth_circular(pts: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian smoothing on a circular sequence with a wrapped kernel.
    """
    if sigma <= 0:
        return pts
    n = pts.shape[0]
    radius = int(max(1, np.ceil(3.0 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.int64)
    w = np.exp(-(k.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    w /= float(np.sum(w))

    out = np.zeros_like(pts)
    for i in range(n):
        idx = (i + k) % n
        out[i] = np.sum(pts[idx] * w[:, None], axis=0)
    return out


# -------------------------
# minimal demo
# -------------------------

if __name__ == "__main__":
    from pprint import pprint

    poly = [(0, 0), (10, 0), (10, 5), (0, 5)]

    feats = compute(
        poly,
        selection=[
            "MeshSurface",
            "Perimeter",
            "Sphericity",
            "MaximumDiameter",
            "Elongation",
            "MeanCurvature",
            "GaussianCurvature",
            "MaximumCurvature",
            "MinimumCurvature",
        ],
        params=ShapeParams(pixel_spacing=None, max_diameter_mode="vertices"),
    )
    print("Subset shape features:")
    pprint(feats, sort_dicts=False)

    all_feats = compute(poly, selection=None, params=ShapeParams(pixel_spacing=(0.25, 0.25)))
    print("\nAll shape features (with spacing=0.25):")
    pprint(all_feats, sort_dicts=False)

