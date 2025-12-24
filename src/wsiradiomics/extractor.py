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
    Extract radiomics features from a Whole Slide Image (WSI).

    This function computes cell-level radiomics features from a WSI and
    aggregates them into WSI-level features. The behavior is fully
    controlled by the configuration file.

    Parameters
    ----------
    svs_path : str or pathlib.Path
        Path to the whole slide image file (e.g. .svs, .tiff, .ndpi).
    geojson_path : str or pathlib.Path
        Path to the GeoJSON file containing per-cell polygon annotations.
    params_path : str or pathlib.Path or None, optional
        Path to a YAML configuration file. If None, default parameters
        will be used.

    Returns
    -------
    dict
        A dictionary containing extracted features:

        - Always includes:
          {
              "wsi_features": Dict[str, float]
          }

        - Additionally includes (only if cfg["output"]["save_cell_features"] is True):
          {
              "cell_features": List[Dict[str, Any]]
          }

    Notes
    -----
    - This function does NOT write any files.
    - All outputs are returned in memory and must be handled by the caller.
    - Slide resources are safely released after extraction.
    - Logging is used instead of print statements.
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

    cfg = load_params(inputs.params_path) if inputs.params_path else default_params()
    logger.debug("Loaded cfg keys: %s", sorted(list((cfg or {}).keys())))

    logger.info("Loading cell polygons from GeoJSON")
    cells = load_cells_from_geojson(inputs.geojson_path)
    logger.info("Loaded %d cells", len(cells))

    logger.info("Opening slide")
    slide = open_slide(inputs.svs_path)

    try:
        # 3) Cell-level features
        logger.info("Computing cell-level features")
        cell_feature_rows: List[Dict[str, Any]] = compute_cell_features(slide, cells, cfg)
        logger.info("Computed cell-level features for %d cells", len(cell_feature_rows))

        # 4) WSI-level aggregation
        logger.info("Aggregating to WSI-level features")
        wsi_features: Dict[str, float] = compute_wsi_features(cell_feature_rows, cfg)
        logger.info("Generated %d WSI features", len(wsi_features))

        save_cell_features = bool(
            ((cfg or {}).get("output", {}) or {}).get("save_cell_features", False)
        )

        out: Dict[str, Any] = {"wsi_features": wsi_features}
        if save_cell_features:
            logger.info("Returning cell-level features")
            out["cell_features"] = cell_feature_rows

        logger.info("WSI radiomics extraction finished")
        return out

    finally:
        try:
            if hasattr(slide, "close"):
                slide.close()
                logger.debug("Slide closed successfully")
        except Exception:
            logger.warning("Failed to close slide", exc_info=True)


def main() -> None:
    import argparse
    from pprint import pprint

    ap = argparse.ArgumentParser("wsiradiomics-extract")
    ap.add_argument("--svs", required=True)
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--params", default=None)
    args = ap.parse_args()

    res = extract(args.svs, args.geojson, params_path=args.params)
    pprint(res["wsi_features"])


if __name__ == "__main__":
    import argparse
    import logging
    from pprint import pprint

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser("wsiradiomics-extract")
    parser.add_argument("--svs", required=True)
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--params", default=None)
    args = parser.parse_args()

    res = extract(args.svs, args.geojson, params_path=args.params)
    pprint(res["wsi_features"])

    logger.info("Returned keys: %s", list(res.keys()))
    logger.info("WSI features: %d", len(res.get("wsi_features", {}) or {}))
    if "cell_features" in res:
        logger.info("Cell features rows: %d", len(res.get("cell_features", []) or []))
