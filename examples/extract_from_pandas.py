# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 09:20
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: extract_from_pandas.py
# @Project : WSIRadiomics

from __future__ import annotations

import os, argparse
from pathlib import Path
import pandas as pd

from wsiradiomics.extractor import extract


def main(input_csv: str | Path, out_dir: str | Path, params_path: str | Path | None = None) -> None:
    df = pd.read_csv(input_csv)
    if "wsi_path" not in df.columns or "mask_path" not in df.columns:
        raise ValueError("CSV must contain columns: wsi_path, mask_path")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        wsi_path = Path(row["wsi_path"])
        mask_path = Path(row["mask_path"])

        print(f"[{i+1}/{len(df)}] {wsi_path.name}")

        res = extract(
            svs_path=wsi_path,
            geojson_path=mask_path,
            params_path=params_path,
        )

        wsi_features: dict = res["wsi_features"]
        cell_features = res.get("cell_features", None)  # If you need to further process cell-level features, add logic here.

        # one patient/slide -> one row
        out_row = {
            "wsi_path": str(wsi_path),
            "mask_path": str(mask_path),
            **{k: float(v) for k, v in (wsi_features or {}).items()},
        }
        rows.append(out_row)

    out_df = pd.DataFrame(rows)

    # stable column order: paths first, then sorted feature names
    fixed = ["wsi_path", "mask_path"]
    feat_cols = sorted([c for c in out_df.columns if c not in fixed])
    out_df = out_df[fixed + feat_cols]

    out_csv = os.path.join(out_dir, "wsi_features.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DONE] Wrote: {out_csv}")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=False,
                    default="/home/huangdn/WSIRadiomics/examples/example_file.csv")
    ap.add_argument("--params", required=False,
                    default="/home/huangdn/WSIRadiomics/examples/params.yaml")
    ap.add_argument("--out_dir", required=False,
                    default="/home/huangdn/WSIRadiomics/result")
    args = ap.parse_args()

    main(args.input_csv, args.out_dir, args.params)
