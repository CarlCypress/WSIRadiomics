# -*- coding: utf-8 -*-
# @Time    : 2025/12/23 09:07
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: cell_features.py
# @Project : WSIRadiomics

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import logging
import time
import numpy as np

from wsiradiomics.features import firstorder as fo
from wsiradiomics.features import shape as sh

logger = logging.getLogger(__name__)

_LOG_EVERY_N_CELLS = 200


def compute_cell_features(slide: Any, cells: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    Compute enabled cell-level features for each cell.

    Config semantics (matching your params.yaml + pprint(cfg)):
      - missing group key => disabled
      - group value []    => compute ALL (selection=None)
      - group value list  => compute ONLY listed
      - group value null  => also treated as ALL

    Returns:
      list of dicts, each dict includes:
        - computed features
        - metadata: __cell_type__, optional __feature_id__
    """
    t0 = time.time()

    enabled = resolve_enabled_cell_features(cfg.get("cell_features", {}))

    logger.info("Cell feature extraction started: n_cells=%d", len(cells))
    logger.info("Enabled cell feature groups: %s", sorted(enabled.keys()) if enabled else [])

    if logger.isEnabledFor(logging.DEBUG):
        # show selection details (None means ALL)
        for g, sel in enabled.items():
            logger.debug("Group '%s' selection: %s", g, "ALL" if sel is None else list(sel))

    rows: List[Dict[str, float]] = []
    for idx, cell in enumerate(cells, start=1):
        feats: Dict[str, float] = {}

        # metadata for later by_cell_type aggregation
        feats["__cell_type__"] = cell.get("cell_type") or "Unknown"
        if cell.get("feature_id") is not None:
            feats["__feature_id__"] = str(cell["feature_id"])

        try:
            # ---- firstorder ----
            if "firstorder" in enabled:
                t1 = time.time()
                feats.update(compute_firstorder_features(slide, cell, enabled["firstorder"], cfg))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Cell %d: firstorder done in %.3fs", idx, time.time() - t1)

            # ---- shape ----
            if "shape" in enabled:
                t2 = time.time()
                feats.update(compute_shape_features(cell, enabled["shape"], cfg))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Cell %d: shape done in %.3fs", idx, time.time() - t2)

            # ---- textures: reserved, not implemented ----
            # if "glcm" in enabled: ...
            # if "gldm" in enabled: ...
            # etc.

        except Exception as e:
            # keep running even if one cell fails
            logger.warning(
                "Failed to compute features for cell %d/%d (cell_type=%s, feature_id=%s): %s",
                idx,
                len(cells),
                feats.get("__cell_type__", "Unknown"),
                feats.get("__feature_id__", ""),
                repr(e),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )

            # best-effort: fill enabled groups with NaNs so aggregation stays stable
            if "firstorder" in enabled:
                feats.update(_nan_features(enabled["firstorder"], fo.FEATURES_ALL))
            if "shape" in enabled:
                feats.update(_nan_features(enabled["shape"], sh.FEATURES_ALL))

        rows.append(feats)

        # coarse progress log
        if idx == 1 or idx % _LOG_EVERY_N_CELLS == 0 or idx == len(cells):
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else float("inf")
            logger.info(
                "Processed cells: %d/%d (%.1f%%), %.1f cells/s",
                idx, len(cells), 100.0 * idx / max(1, len(cells)), rate
            )

    logger.info("Cell feature extraction finished: n_rows=%d, elapsed=%.2fs", len(rows), time.time() - t0)
    return rows


def resolve_enabled_cell_features(cell_cfg: Dict[str, Any]) -> Dict[str, Optional[Sequence[str]]]:
    """
    Normalize YAML/cfg selection:
      - []   => None (ALL)
      - None => None (ALL)
      - list => list (ONLY)
    """
    enabled: Dict[str, Optional[Sequence[str]]] = {}
    for group, selection in (cell_cfg or {}).items():
        if selection is None:
            enabled[group] = None
        elif isinstance(selection, list) and len(selection) == 0:
            enabled[group] = None
        elif isinstance(selection, list):
            enabled[group] = selection
        else:
            enabled[group] = None
    return enabled


def compute_firstorder_features(
    slide: Any,
    cell: Dict[str, Any],
    selection: Optional[Sequence[str]],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute first-order features for a single cell polygon on the slide.

    Assumptions:
      - cell["polygon_xy"] coordinates are in level0 pixel space (as produced by your geojson loader)
      - slide is OpenSlide-like with read_region OR wrapper with read_region_xywh OR ndarray
    """
    poly = cell.get("polygon_xy")
    if not poly or len(poly) < 3:
        logger.debug("Firstorder: invalid polygon (len<3), returning NaNs")
        return _nan_features(selection, fo.FEATURES_ALL)

    level = int(cfg.get("slide_level", 0))
    pad = int(cfg.get("cell_patch_padding", 2))
    channel = cfg.get("channel", "gray")  # "gray" or int channel index

    params = fo.FirstOrderParams(
        c=float(cfg.get("intensity_shift", 0.0)),
        voxel_volume=float(cfg.get("voxel_volume", 1.0)),
        eps=float(cfg.get("eps", 1e-12)),
        bin_width=float(cfg["bin_width"]) if cfg.get("bin_width") is not None else None,
        n_bins=int(cfg.get("n_bins", 64)),
    )

    ds = _get_level_downsample(slide, level)

    # bbox in level0 coords
    xs0 = [p[0] for p in poly]
    ys0 = [p[1] for p in poly]
    minx0 = int(np.floor(min(xs0))) - pad
    miny0 = int(np.floor(min(ys0))) - pad
    maxx0 = int(np.ceil(max(xs0))) + pad
    maxy0 = int(np.ceil(max(ys0))) + pad

    # bbox at target level
    minx = int(np.floor(minx0 / ds))
    miny = int(np.floor(miny0 / ds))
    maxx = int(np.ceil(maxx0 / ds))
    maxy = int(np.ceil(maxy0 / ds))
    w = max(1, maxx - minx + 1)
    h = max(1, maxy - miny + 1)

    patch = _read_patch_safe(slide, minx, miny, w, h, level=level)
    if patch is None:
        logger.debug("Firstorder: patch is None, returning NaNs")
        return _nan_features(selection, fo.FEATURES_ALL)

    img = _to_gray_or_channel(patch, channel=channel)
    if img.size == 0:
        logger.debug("Firstorder: empty image patch, returning NaNs")
        return _nan_features(selection, fo.FEATURES_ALL)

    # polygon to target-level local coords
    poly_lvl = [(float(x / ds), float(y / ds)) for (x, y) in poly]
    poly_local = [(x - float(minx), y - float(miny)) for (x, y) in poly_lvl]

    mask = _polygon_mask(img.shape[0], img.shape[1], poly_local)
    values = img[mask]
    if values.size == 0:
        logger.debug("Firstorder: mask has no pixels, returning NaNs")
        return _nan_features(selection, fo.FEATURES_ALL)

    return fo.compute(values, selection=selection, params=params)


def compute_shape_features(
    cell: Dict[str, Any],
    selection: Optional[Sequence[str]],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute 2D shape features from polygon geometry.
    """
    poly = cell.get("polygon_xy")
    if not poly or len(poly) < 3:
        logger.debug("Shape: invalid polygon (len<3), returning NaNs")
        return _nan_features(selection, sh.FEATURES_ALL)

    # optional physical spacing
    pixel_spacing = cfg.get("pixel_spacing", None)
    spacing = None
    if isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) == 2:
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))

    params = sh.ShapeParams(
        pixel_spacing=spacing,
        max_diameter_mode=str(cfg.get("max_diameter_mode", "vertices")),
    )
    return sh.compute(poly, selection=selection, params=params)


# -------------------------
# internal helpers
# -------------------------

def _nan_features(selection: Optional[Sequence[str]], all_names: Sequence[str]) -> Dict[str, float]:
    names = list(all_names) if selection is None else list(selection)
    return {k: float("nan") for k in names}


def _get_level_downsample(slide: Any, level: int) -> float:
    if level <= 0:
        return 1.0
    if hasattr(slide, "level_downsamples"):
        try:
            return float(slide.level_downsamples[level])
        except Exception:
            pass
    if hasattr(slide, "get_downsample"):
        try:
            return float(slide.get_downsample(level))
        except Exception:
            pass
    return 1.0


def _read_patch_safe(slide: Any, x: int, y: int, w: int, h: int, *, level: int = 0) -> Any:
    """
    Read patch; if out-of-range, pad with white to requested size.
    """
    if w <= 0 or h <= 0:
        return None

    level_w = level_h = None
    if hasattr(slide, "level_dimensions"):
        try:
            level_w, level_h = slide.level_dimensions[level]
        except Exception:
            pass

    x0, y0 = int(x), int(y)
    x1, y1 = int(x + w), int(y + h)

    if level_w is not None and level_h is not None:
        x0c = max(0, min(x0, int(level_w)))
        y0c = max(0, min(y0, int(level_h)))
        x1c = max(0, min(x1, int(level_w)))
        y1c = max(0, min(y1, int(level_h)))
        wc = max(0, x1c - x0c)
        hc = max(0, y1c - y0c)
    else:
        x0c = max(0, x0)
        y0c = max(0, y0)
        wc = max(0, w - (x0c - x0))
        hc = max(0, h - (y0c - y0))

    if wc <= 0 or hc <= 0:
        return None

    patch = _read_patch(slide, x0c, y0c, wc, hc, level=level)

    dx = x0c - x0
    dy = y0c - y0
    if dx != 0 or dy != 0 or wc != w or hc != h:
        patch_arr = _as_rgb_array(patch)
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        canvas[dy:dy + hc, dx:dx + wc] = patch_arr[:hc, :wc]
        return canvas

    return patch


def _read_patch(slide: Any, x: int, y: int, w: int, h: int, *, level: int = 0) -> Any:
    if hasattr(slide, "read_region"):
        return slide.read_region((int(x), int(y)), int(level), (int(w), int(h)))
    if hasattr(slide, "read_region_xywh"):
        return slide.read_region_xywh(int(x), int(y), int(w), int(h), level=int(level))
    if isinstance(slide, np.ndarray):
        return slide[int(y):int(y + h), int(x):int(x + w)]
    raise TypeError("Unsupported slide backend.")


def _as_rgb_array(patch: Any) -> np.ndarray:
    if hasattr(patch, "convert"):
        return np.array(patch.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(patch)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2).astype(np.uint8)
    return arr.astype(np.uint8)


def _to_gray_or_channel(patch: Any, *, channel: Any = "gray") -> np.ndarray:
    if hasattr(patch, "convert"):
        rgb = np.array(patch.convert("RGB"), dtype=np.uint8)
        if channel == "gray":
            return rgb.mean(axis=2).astype(np.float64)
        if isinstance(channel, int):
            ch = max(0, min(int(channel), 2))
            return rgb[:, :, ch].astype(np.float64)
        return rgb.mean(axis=2).astype(np.float64)

    arr = np.asarray(patch)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        if channel == "gray":
            return arr[:, :, :3].mean(axis=2).astype(np.float64)
        if isinstance(channel, int):
            ch = max(0, min(int(channel), arr.shape[2] - 1))
            return arr[:, :, ch].astype(np.float64)
        return arr[:, :, :3].mean(axis=2).astype(np.float64)
    return np.asarray([], dtype=np.float64)


def _polygon_mask(h: int, w: int, poly_xy: List[tuple[float, float]]) -> np.ndarray:
    """
    Rasterize polygon to boolean mask of shape (h,w).
    Tries skimage.draw.polygon; falls back to matplotlib.path.
    """
    try:
        from skimage.draw import polygon as sk_polygon  # type: ignore
        xs = np.array([p[0] for p in poly_xy], dtype=np.float64)
        ys = np.array([p[1] for p in poly_xy], dtype=np.float64)
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        return mask
    except Exception:
        from matplotlib.path import Path as MplPath  # type: ignore
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        pts = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        path = MplPath(np.asarray(poly_xy, dtype=np.float64))
        inside = path.contains_points(pts)
        return inside.reshape((h, w))