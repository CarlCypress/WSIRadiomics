# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 09:00
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: extractor.py
# @Project : WSIRadiomics

# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 09:00
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: extractor.py
# @Project : WSIRadiomics

"""
wsiradiomics.extractor

Repository-level extractor.

Inputs:
- svs_path: path to WSI (.svs/.tiff/.ndpi, etc.)
- geojson_path: path to per-cell polygons (GeoJSON)
- params_path: YAML config

Behavior:
- Always compute and return WSI-level aggregated features.
- Optionally return cell-level features if cfg["output"]["save_cell_features"] is True.

IMPORTANT:
- This module does NOT write any files. Output writing should be handled elsewhere.
- This module uses Python logging (no prints). Users control handlers/level/output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from wsiradiomics.io.params import load_params, default_params
from wsiradiomics.io.geojson import load_cells_from_geojson
from wsiradiomics.io.slide import open_slide
from wsiradiomics.pipeline.cell_features import compute_cell_features
from wsiradiomics.pipeline.wsi_aggregation import compute_wsi_features

logger = logging.getLogger(__name__)


@dataclass
class ExtractorInputs:
    svs_path: Path
    geojson_path: Path
    params_path: Optional[Path] = None


def extract(
    svs_path: str | Path,
    geojson_path: str | Path,
    params_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Main API: extract radiomics features.

    Returns:
        Always returns:
          {
            "wsi_features": Dict[str, float]
          }

        Additionally returns (only when cfg["output"]["save_cell_features"] is True):
          {
            "cell_features": List[Dict[str, Any]]
          }
    """
    inputs = ExtractorInputs(
        svs_path=Path(svs_path),
        geojson_path=Path(geojson_path),
        params_path=Path(params_path) if params_path else None,
    )

    logger.info("Starting WSI radiomics extraction")
    logger.info("WSI: %s", inputs.svs_path)
    logger.info("GeoJSON: %s", inputs.geojson_path)
    logger.info("Params: %s", inputs.params_path if inputs.params_path else "(default params)")

    # 0) Load config
    cfg = load_params(inputs.params_path) if inputs.params_path else default_params()
    logger.debug("Loaded cfg keys: %s", sorted(list((cfg or {}).keys())))

    # 1) Load cell polygons (and optional properties like cell_type)
    logger.info("Loading cell polygons from GeoJSON")
    cells = load_cells_from_geojson(inputs.geojson_path)
    logger.info("Loaded %d cells", len(cells))

    # 2) Open slide / image backend (OpenSlide, tifffile, etc.)
    logger.info("Opening slide")
    slide = open_slide(inputs.svs_path)

    # 3) Compute cell-level features (list of dicts)
    logger.info("Computing cell-level features")
    cell_feature_rows: List[Dict[str, Any]] = compute_cell_features(slide, cells, cfg)
    logger.info("Computed cell-level features for %d cells", len(cell_feature_rows))

    # 4) Aggregate to WSI-level features (single dict)
    logger.info("Aggregating to WSI-level features")
    wsi_features: Dict[str, float] = compute_wsi_features(cell_feature_rows, cfg)
    logger.info("Generated %d WSI features", len(wsi_features))

    # 5) Decide whether to return cell features based on cfg
    save_cell_features = bool(((cfg or {}).get("output", {}) or {}).get("save_cell_features", False))

    out: Dict[str, Any] = {"wsi_features": wsi_features}
    if save_cell_features:
        logger.info("Returning cell-level features (cfg.output.save_cell_features=True)")
        out["cell_features"] = cell_feature_rows
    else:
        logger.info("Not returning cell-level features (cfg.output.save_cell_features=False/missing)")

    logger.info("WSI radiomics extraction finished")
    return out

def main() -> None:
    import argparse
    from pprint import pprint

    ap = argparse.ArgumentParser("wsiradiomics-extract")
    ap.add_argument("--svs", required=True, help="Path to WSI file (.svs/.tiff/.ndpi)")
    ap.add_argument("--geojson", required=True, help="Path to per-cell polygons GeoJSON")
    ap.add_argument("--params", default=None, help="Path to params.yaml (optional)")
    args = ap.parse_args()

    res = extract(args.svs, args.geojson, params_path=args.params)
    pprint(res["wsi_features"])


if __name__ == "__main__":
    import argparse

    # NOTE: Example CLI runner for local debugging.
    # For pip users, they should configure logging in their own code.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser("wsiradiomics")
    parser.add_argument(
        "--svs",
        required=False,
        default="/home/huangdn/dataset/HE_of_PC/陆懿红.svs",
        help="Path to WSI file (.svs)",
    )
    parser.add_argument(
        "--geojson",
        required=False,
        default="/home/huangdn/CellViT-plus-plus/result/陆懿红_cells.geojson",
        help="Path to per-cell GeoJSON",
    )
    parser.add_argument(
        "--params",
        required=False,
        default="/home/huangdn/WSIRadiomics/examples/params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()

    res = extract(
        args.svs,
        args.geojson,
        params_path=args.params,
    )

    # If you want to quickly inspect counts:
    logger.info("Returned keys: %s", list(res.keys()))
    logger.info("WSI features: %d", len(res.get("wsi_features", {}) or {}))
    if "cell_features" in res:
        logger.info("Cell features rows: %d", len(res.get("cell_features", []) or []))
