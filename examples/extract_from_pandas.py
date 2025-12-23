# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 09:20
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: extract_from_pandas.py
# @Project : WSIRadiomics

from __future__ import annotations

import os
import argparse
import time
import logging
from pathlib import Path

import pandas as pd

from wsiradiomics.extractor import extract


def main(input_csv: str | Path, out_dir: str | Path, params_path: str | Path | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(out_dir, "run_wsi_feature_extract.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),                       # console
            logging.FileHandler(log_file, mode="w"),       # file
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("Batch extraction started")
    logger.info("Input CSV: %s", input_csv)
    logger.info("Params YAML: %s", params_path if params_path else "(default)")
    logger.info("Output dir: %s", out_dir)

    df = pd.read_csv(input_csv)
    if "wsi_path" not in df.columns or "mask_path" not in df.columns:
        raise ValueError("CSV must contain columns: wsi_path, mask_path")

    rows = []
    total = len(df)
    ok, fail = 0, 0

    for i, row in df.iterrows():
        wsi_path = Path(row["wsi_path"])
        mask_path = Path(row["mask_path"])

        logger.info("[%d/%d] Processing: %s", i + 1, total, wsi_path.name)
        t0 = time.time()

        try:
            res = extract(
                svs_path=wsi_path,
                geojson_path=mask_path,
                params_path=params_path,
            )

            wsi_features = res["wsi_features"]
            cell_features = res.get("cell_features", None)
            # If you need to further process cell-level features, add logic here.

            out_row = {
                "wsi_path": str(wsi_path),
                "mask_path": str(mask_path),
                **{k: float(v) for k, v in (wsi_features or {}).items()},
            }
            rows.append(out_row)

            ok += 1
            logger.info(
                "[%d/%d] DONE: %s | n_features=%d | %.2fs",
                i + 1,
                total,
                wsi_path.name,
                len(wsi_features),
                time.time() - t0,
            )

        except Exception as e:
            fail += 1
            logger.exception(
                "[%d/%d] FAILED: %s | %.2fs",
                i + 1,
                total,
                wsi_path.name,
                time.time() - t0,
            )

    out_df = pd.DataFrame(rows)

    fixed = ["wsi_path", "mask_path"]
    feat_cols = sorted([c for c in out_df.columns if c not in fixed])
    out_df = out_df[fixed + feat_cols]

    out_csv = os.path.join(out_dir, "wsi_features.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    logger.info("Batch finished: ok=%d fail=%d total=%d", ok, fail, total)
    logger.info("WSI features saved to: %s", out_csv)
    logger.info("Log file saved to: %s", log_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="/home/huangdn/WSIRadiomics/examples/example_file.csv")
    ap.add_argument("--params", default="/home/huangdn/WSIRadiomics/examples/params.yaml")
    ap.add_argument("--out_dir", default="/home/huangdn/WSIRadiomics/result")
    args = ap.parse_args()

    main(args.input_csv, args.out_dir, args.params)