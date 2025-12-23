# -*- coding: utf-8 -*-
# @Time    : 2025/12/23 19:47
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: make_csv_file.py
# @Project : WSIRadiomics
import os
import pandas as pd


def make_csv(wsi_dir="/home/huangdn/dataset/HE_of_PC",
             mask_dir="/home/huangdn/CellViT-plus-plus/result",
             out_csv="/home/huangdn/WSIRadiomics/examples/example_file.csv",):
    rows = []

    for fname in os.listdir(wsi_dir):
        if not fname.endswith(".svs"):
            continue

        stem = os.path.splitext(fname)[0]
        wsi_path = os.path.join(wsi_dir, fname)

        mask_name = f"{stem}_cells.geojson"
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Skip (no mask): {fname}")
            continue

        rows.append({
            "wsi_path": wsi_path,
            "mask_path": mask_path,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    make_csv()
