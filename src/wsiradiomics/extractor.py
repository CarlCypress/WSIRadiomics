# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 09:00
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: extractor.py
# @Project : WSIRadiomics

"""
wsiradiomics.extractor

Repository-level extractor skeleton.

Inputs:
- svs_path: path to WSI (.svs/.tiff/.ndpi, etc.)
- geojson_path: path to per-cell polygons (GeoJSON)
- out_dir: directory to save WSI-level aggregated features (CSV)
- params_path: YAML config (PyRadiomics-like selection)

Design:
- Read params.yaml -> determine enabled cell feature groups + aggregation functions
- Load cells (polygons + optional properties like cell_type) from geojson
- For each cell: compute cell-level features (firstorder/shape/texture...) via wsiradiomics.features.*
- Aggregate across cells into WSI-level features via wsiradiomics.aggregation
- Save a single-row CSV under out_dir

This file intentionally keeps implementation minimal; fill in TODOs in your repo.
"""

from __future__ import annotations

from pprint import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from wsiradiomics.io.params import load_params, default_params
from wsiradiomics.io.geojson import load_cells_from_geojson
from wsiradiomics.io.slide import open_slide
from wsiradiomics.utils.patch_check import find_nonwhite_patch


# -------------------------
# Public config + API
# -------------------------

@dataclass
class ExtractorInputs:
    svs_path: Path
    geojson_path: Path
    out_dir: Path
    params_path: Optional[Path] = None


def extract(
    svs_path: str | Path,
    geojson_path: str | Path,
    out_dir: str | Path,
    params_path: str | Path | None = None,
) -> Dict[str, float]:
    """
    Main API: extract WSI-level radiomics features.

    Returns:
        A dict of {feature_name: value} (WSI-level), also written to a CSV in out_dir.
    """
    inputs = ExtractorInputs(
        svs_path=Path(svs_path),
        geojson_path=Path(geojson_path),
        out_dir=Path(out_dir),
        params_path=Path(params_path) if params_path else None,
    )

    cfg = load_params(inputs.params_path) if inputs.params_path else default_params()

    # 1) Load cell polygons (and optional properties like cell_type)
    cells = load_cells_from_geojson(inputs.geojson_path)

    # 2) Open slide / image backend (OpenSlide, tifffile, etc.)
    slide = open_slide(inputs.svs_path)

    # 3) Compute cell-level features (list of dicts)
    cell_feature_rows = compute_cell_features(slide, cells, cfg)

    # 4) Aggregate to WSI-level features (single dict)
    wsi_features = compute_wsi_features(cell_feature_rows, cfg)

    # 5) Build output CSV path from out_dir + slide name
    out_csv = build_out_csv_path(inputs.svs_path, inputs.out_dir, inputs.params_path)

    # 6) Save CSV
    save_wsi_features_csv(wsi_features, out_csv)

    return wsi_features


def build_out_csv_path(svs_path: Path, out_dir: Path, params_path: Optional[Path] = None) -> Path:
    """
    Decide output CSV filename.

    Default:
      {out_dir}/{slide_stem}_wsi_features.csv

    If params_path is provided, you may append config name, e.g.:
      {slide_stem}_{params_stem}.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_stem = svs_path.stem

    # Keep it simple; you can change naming later
    if params_path is not None:
        params_stem = params_path.stem
        return out_dir / f"{slide_stem}_{params_stem}_wsi_features.csv"

    return out_dir / f"{slide_stem}_wsi_features.csv"


# -------------------------
# Feature computation
# -------------------------


def compute_cell_features(slide: Any, cells: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    Compute enabled cell-level features for each cell.

    Expected output:
        list of dicts (one per cell), e.g.
        [{"Energy":..., "Perimeter":..., ...}, ...]
    """
    enabled = resolve_enabled_cell_features(cfg.get("cell_features", {}))

    rows: List[Dict[str, float]] = []
    for cell in cells:
        feats: Dict[str, float] = {}

        if enabled.get("firstorder") is not None:
            feats.update(compute_firstorder_features(slide, cell, enabled["firstorder"], cfg))

        if enabled.get("shape") is not None:
            feats.update(compute_shape_features(cell, enabled["shape"], cfg))

        # TODO: texture groups

        rows.append(feats)

    return rows


def resolve_enabled_cell_features(cell_cfg: Dict[str, Any]) -> Dict[str, Optional[Sequence[str]]]:
    """
    Convert YAML selection into an internal format:
    - None => ALL in that group
    - list[str] => only those
    - missing => group disabled
    """
    enabled: Dict[str, Optional[Sequence[str]]] = {}
    for group, selection in cell_cfg.items():
        enabled[group] = selection
    return enabled


def compute_firstorder_features(
    slide: Any,
    cell: Dict[str, Any],
    selection: Optional[Sequence[str]],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Compute first-order features for a single cell."""
    raise NotImplementedError


def compute_shape_features(
    cell: Dict[str, Any],
    selection: Optional[Sequence[str]],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Compute shape features from polygon geometry."""
    raise NotImplementedError


# -------------------------
# Aggregation (WSI-level)
# -------------------------

def compute_wsi_features(cell_rows: List[Dict[str, float]], cfg: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate cell-level feature rows into WSI-level features."""
    raise NotImplementedError


# -------------------------
# Output
# -------------------------

def save_wsi_features_csv(wsi_features: Dict[str, float], out_csv: Path) -> None:
    """Save a single-row CSV."""
    raise NotImplementedError


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("wsiradiomics")
    parser.add_argument("--svs", required=False,
                        default="/home/huangdn/dataset/HE_of_PC/陆懿红.svs",
                        help="Path to WSI file (.svs)")
    parser.add_argument("--geojson", required=False,
                        default="/home/huangdn/CellViT-plus-plus/result/陆懿红_cells.geojson",
                        help="Path to per-cell GeoJSON")
    parser.add_argument("--out_dir", required=False,
                        default="/home/huangdn/WSIRadiomics/result",
                        help="Directory to save WSI-level feature CSV")
    parser.add_argument("--params", required=False,
                        default="/home/huangdn/WSIRadiomics/examples/params.yaml",
                        help="Path to params.yaml")
    args = parser.parse_args()

    extract(
        args.svs,
        args.geojson,
        args.out_dir,
        params_path=args.params,
    )