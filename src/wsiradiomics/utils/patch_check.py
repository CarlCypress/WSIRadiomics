# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 21:53
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: patch_check.py
# @Project : WSIRadiomics
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


def find_nonwhite_patch(slide, level: int = 0, patch_size: Tuple[int, int] = (512, 512),
                        grid: int = 8) -> Tuple[Tuple[int, int], float]:
    """
    Find a patch location that is least likely to be blank background.

    Returns:
        (best_xy, best_mean)
        best_xy is in level-0 coordinates (OpenSlide convention).
        best_mean is mean pixel intensity (lower -> less white).
    """
    w0, h0 = slide.level_dimensions[0]
    best_mean: Optional[float] = None
    best_xy: Tuple[int, int] = (0, 0)

    def patch_mean(img) -> float:
        arr = np.asarray(img, dtype=np.float32)
        return float(arr.mean())

    for gy in range(grid):
        for gx in range(grid):
            x = int((gx + 0.5) / grid * w0)
            y = int((gy + 0.5) / grid * h0)

            img = slide.read_region((x, y), level=level, size_wh=patch_size)
            m = patch_mean(img)

            if best_mean is None or m < best_mean:
                best_mean = m
                best_xy = (x, y)

    return best_xy, float(best_mean if best_mean is not None else 255.0)
