# -*- coding: utf-8 -*-
# @Time    : 2025/12/23 10:42
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: output.py
# @Project : WSIRadiomics

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Any


def build_out_csv_path(svs_path: Path, out_dir: Path, params_path: Optional[Path] = None) -> Path:
    """
    Decide output CSV filename.

    Default:
      {out_dir}/{slide_stem}_wsi_features.csv

    If params_path is provided:
      {out_dir}/{slide_stem}_{params_stem}_wsi_features.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_stem = Path(svs_path).stem

    if params_path is not None:
        params_stem = Path(params_path).stem
        return out_dir / f"{slide_stem}_{params_stem}_wsi_features.csv"

    return out_dir / f"{slide_stem}_wsi_features.csv"


def save_wsi_features_csv(
    wsi_features: Dict[str, float],
    out_csv: Path,
    *,
    sort_columns: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Save a single-row CSV of WSI-level features.

    - Column order is sorted by default (stable for ML pipelines).
    - If overwrite=False and file exists, raises FileExistsError.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        raise FileExistsError(f"Output CSV already exists: {out_csv}")

    if wsi_features is None:
        wsi_features = {}

    keys = list(wsi_features.keys())
    if sort_columns:
        keys = sorted(keys)

    # Ensure all values are serializable floats
    row = []
    for k in keys:
        v = wsi_features.get(k, float("nan"))
        try:
            fv = float(v)
        except Exception:
            fv = float("nan")
        row.append(fv)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerow(row)


def write_wsi_features_csv(
    wsi_features: Dict[str, float],
    *,
    svs_path: Path,
    out_dir: Path,
    params_path: Optional[Path] = None,
    sort_columns: bool = True,
    overwrite: bool = True,
    add_meta: bool = False,
) -> Path:
    """
    Combined step for extractor: build output path + save CSV.

    Returns:
        out_csv path

    add_meta:
      If True, will add a few non-feature columns that are helpful for bookkeeping.
      Default False to keep pure feature table.
    """
    out_csv = build_out_csv_path(svs_path=Path(svs_path), out_dir=Path(out_dir), params_path=params_path)

    feats = dict(wsi_features or {})

    if add_meta:
        feats = _add_meta_fields(feats, svs_path=Path(svs_path), params_path=params_path)

    save_wsi_features_csv(
        feats,
        out_csv,
        sort_columns=sort_columns,
        overwrite=overwrite,
    )
    return out_csv


# -------------------------
# internals
# -------------------------

def _add_meta_fields(
    feats: Dict[str, Any],
    *,
    svs_path: Path,
    params_path: Optional[Path],
) -> Dict[str, Any]:
    """
    Add optional metadata fields.
    We prefix with __ to reduce collision with real feature names.
    """
    out = dict(feats)
    out["__slide__"] = svs_path.name
    out["__svs_path__"] = str(svs_path)
    out["__params__"] = params_path.name if params_path is not None else ""
    out["__n_features__"] = float(len(feats))
    return out


if __name__ == "__main__":
    # Minimal usage example
    from pprint import pprint
    import tempfile

    dummy_features = {
        "All__Energy__Mean": 123.4,
        "All__Perimeter__Mean": 56.7,
        "Neoplastic__Energy__Mean": 200.1,
    }

    with tempfile.TemporaryDirectory() as td:
        out = write_wsi_features_csv(
            dummy_features,
            svs_path=Path("demo_slide.svs"),
            out_dir=Path(td),
            params_path=Path("params.yaml"),
            add_meta=False,
        )
        print("Wrote:", out)
        pprint(dummy_features, sort_dicts=False)
