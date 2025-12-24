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
from typing import Any, Dict, Tuple, Optional

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from wsiradiomics.extractor import extract


def _init_worker_logging(log_file: str) -> None:
    """
    Initialize logging in worker process (Windows spawn won't inherit main process logging).
    Keep it minimal: root logger -> console + same log file (append).
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def _worker_extract_one(
    wsi_path: str,
    mask_path: str,
    params_path: Optional[str],
    log_file: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Worker process entry:
      - return (ok, payload)
      - payload on success: {"wsi_path":..., "mask_path":..., "wsi_features": {...}}
      - payload on failure: {"wsi_path":..., "mask_path":..., "error": "...", "traceback": "..."}
    """
    import traceback

    # IMPORTANT: configure logging inside worker
    _init_worker_logging(log_file)

    try:
        res = extract(
            svs_path=Path(wsi_path),
            geojson_path=Path(mask_path),
            params_path=Path(params_path) if params_path else None,
        )
        wsi_features = res["wsi_features"]

        return True, {
            "wsi_path": wsi_path,
            "mask_path": mask_path,
            "wsi_features": wsi_features,
        }
    except Exception as e:
        return False, {
            "wsi_path": wsi_path,
            "mask_path": mask_path,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def main(
    input_csv: str | Path,
    out_dir: str | Path,
    params_path: str | Path | None = None,
    num_workers: int = 4,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(out_dir, "run_wsi_feature_extract.log")

    # main process logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),                 # console
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),  # file (overwrite)
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("Batch extraction started")
    logger.info("Input CSV: %s", input_csv)
    logger.info("Params YAML: %s", params_path if params_path else "(default)")
    logger.info("Output dir: %s", out_dir)
    logger.info("num_workers: %s", num_workers)

    df = pd.read_csv(input_csv)
    if "wsi_path" not in df.columns or "mask_path" not in df.columns:
        raise ValueError("CSV must contain columns: wsi_path, mask_path")

    total = len(df)
    ok, fail = 0, 0
    rows = []

    tasks = []
    for _, r in df.iterrows():
        tasks.append((
            str(Path(r["wsi_path"])),
            str(Path(r["mask_path"])),
            str(Path(params_path)) if params_path else None,
        ))

    t_all = time.time()

    # -------------------------
    # multiprocessing
    # -------------------------
    with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
        future_map: Dict[Any, Tuple[int, int, str, float]] = {}

        # submit
        for idx, (wsi_p, mask_p, params_p) in enumerate(tasks, start=1):
            t_submit = time.time()
            fut = ex.submit(_worker_extract_one, wsi_p, mask_p, params_p, log_file)
            future_map[fut] = (idx, total, Path(wsi_p).name, t_submit)

        # collect
        for fut in as_completed(future_map):
            idx, total, name, t_submit = future_map[fut]

            try:
                success, payload = fut.result()

                if success:
                    wsi_features = payload["wsi_features"]

                    out_row = {
                        "wsi_path": payload["wsi_path"],
                        "mask_path": payload["mask_path"],
                        **{k: float(v) for k, v in (wsi_features or {}).items()},
                    }
                    rows.append(out_row)

                    ok += 1
                    logger.info(
                        "[%d/%d] DONE: %s | n_features=%d | %.2fs",
                        idx, total, name, len(wsi_features), time.time() - t_submit
                    )
                else:
                    fail += 1
                    logger.error("[%d/%d] FAILED: %s", idx, total, name)
                    logger.error("Error: %s", payload.get("error", ""))
                    logger.error("Traceback:\n%s", payload.get("traceback", ""))

            except Exception:
                fail += 1
                logger.exception("[%d/%d] FAILED (unexpected): %s", idx, total, name)

    # -------------------------
    # save merged CSV
    # -------------------------
    out_df = pd.DataFrame(rows)

    fixed = ["wsi_path", "mask_path"]
    feat_cols = sorted([c for c in out_df.columns if c not in fixed])
    out_df = out_df[fixed + feat_cols]

    out_csv = os.path.join(out_dir, "wsi_features.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    logger.info("Batch finished: ok=%d fail=%d total=%d", ok, fail, total)
    logger.info("WSI features saved to: %s", out_csv)
    logger.info("Log file saved to: %s", log_file)
    logger.info("Total wall time: %.2fs", time.time() - t_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=r"D:\research\WSIRadiomics\examples\HE_list.csv")
    ap.add_argument("--params", default=r"D:\research\WSIRadiomics\examples\params.yaml")
    ap.add_argument("--out_dir", default=r"D:\research\WSIRadiomics\result")
    ap.add_argument("--num_workers", type=int, default=16, help="Number of worker processes")
    args = ap.parse_args()

    main(args.input_csv, args.out_dir, args.params, args.num_workers)