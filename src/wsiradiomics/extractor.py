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
from wsiradiomics.pipeline.cell_features import compute_cell_features
from wsiradiomics.pipeline.wsi_aggregation import compute_wsi_features
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

# -------------------------
# Aggregation (WSI-level)
# -------------------------

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