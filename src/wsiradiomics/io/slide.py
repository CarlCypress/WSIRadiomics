# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 21:00
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: slide.py
# @Project : WSIRadiomics

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Optional
import logging


# -------------------------
# Module logger (library-friendly)
# -------------------------
logger = logging.getLogger(__name__)


# -------------------------
# Public interface
# -------------------------

SizeWH = Tuple[int, int]
PointXY = Tuple[int, int]


class SlideReader:
    """
    Minimal backend-agnostic interface for reading WSIs.

    Your feature extractors should only depend on this interface.
    """

    def close(self) -> None:
        raise NotImplementedError

    @property
    def level_count(self) -> int:
        raise NotImplementedError

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        """[(width, height), ...] per level."""
        raise NotImplementedError

    @property
    def level_downsamples(self) -> Sequence[float]:
        """[downsample, ...] per level (level0=1.0)."""
        raise NotImplementedError

    def read_region(self, location_xy: PointXY, level: int, size_wh: SizeWH):
        """
        Read an RGB image patch.

        Args:
            location_xy: (x, y) at level-0 reference frame (OpenSlide convention)
            level: pyramid level
            size_wh: (w, h) in pixels at the requested level

        Returns:
            PIL.Image (RGB)
        """
        raise NotImplementedError


def open_slide(svs_path: Path | str) -> SlideReader:
    """
    Create a slide reader.

    Backends (in order):
      1) openslide-python (best for .svs/.ndpi/.mrxs/...)
      2) tifffile (fallback for TIFF-like files)

    Raises:
      FileNotFoundError if path not found
      RuntimeError if no backend can open the file
    """
    path = Path(svs_path)
    if not path.exists():
        raise FileNotFoundError(f"WSI not found: {path}")

    logger.info("Opening slide: %s", path)

    # --- Try OpenSlide ---
    reader = _try_openslide(path)
    if reader is not None:
        logger.info("Slide opened with backend: OpenSlide")
        return reader

    # --- Try tifffile fallback ---
    reader = _try_tifffile(path)
    if reader is not None:
        logger.info("Slide opened with backend: tifffile")
        return reader

    raise RuntimeError(
        "Failed to open slide with available backends. "
        "Tried: openslide-python, tifffile. "
        "Please install openslide (recommended) or provide a supported TIFF."
    )


# -------------------------
# OpenSlide backend
# -------------------------

@dataclass
class OpenSlideReader(SlideReader):
    _osr: Any  # openslide.OpenSlide

    def close(self) -> None:
        try:
            self._osr.close()
        except Exception:
            pass

    @property
    def level_count(self) -> int:
        return int(self._osr.level_count)

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        # OpenSlide provides (w, h) tuples
        return list(self._osr.level_dimensions)

    @property
    def level_downsamples(self) -> Sequence[float]:
        return list(self._osr.level_downsamples)

    def read_region(self, location_xy: PointXY, level: int, size_wh: SizeWH):
        x, y = int(location_xy[0]), int(location_xy[1])
        w, h = int(size_wh[0]), int(size_wh[1])

        # OpenSlide returns RGBA PIL.Image
        img = self._osr.read_region((x, y), int(level), (w, h))
        # Convert to RGB for downstream consistency
        return img.convert("RGB")


def _try_openslide(path: Path) -> Optional[SlideReader]:
    # Lazy import: do NOT import openslide at module import time
    try:
        import openslide  # type: ignore
    except Exception as e:
        logger.debug("OpenSlide import failed (%s). Will try fallback backends.", repr(e))
        return None

    try:
        osr = openslide.OpenSlide(str(path))
        reader = OpenSlideReader(osr)

        # Light introspection log (safe)
        try:
            logger.info(
                "OpenSlide levels=%d level0=%s",
                reader.level_count,
                reader.level_dimensions[0] if reader.level_count > 0 else None,
            )
        except Exception:
            pass

        return reader
    except Exception as e:
        logger.debug("OpenSlide failed to open %s (%s).", path, repr(e))
        return None


# -------------------------
# tifffile backend (fallback)
# -------------------------

@dataclass
class TiffFileReader(SlideReader):
    """
    Simple fallback reader for TIFF-like images.

    Limitations:
      - Might not support vendor pyramids the same way as OpenSlide.
      - For many SVS files, tifffile may fail or be slow.
    """
    _tif: Any  # tifffile.TiffFile
    _levels: List[Any]  # list of pages/series entries

    def close(self) -> None:
        try:
            self._tif.close()
        except Exception:
            pass

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        dims: List[Tuple[int, int]] = []
        for lv in self._levels:
            # tifffile pages typically have shape (h, w, c) or (h, w)
            shape = getattr(lv, "shape", None)
            if shape is None:
                dims.append((0, 0))
                continue
            if len(shape) >= 2:
                h, w = int(shape[0]), int(shape[1])
                dims.append((w, h))
            else:
                dims.append((0, 0))
        return dims

    @property
    def level_downsamples(self) -> Sequence[float]:
        # Estimate downsample from level0 width
        dims = self.level_dimensions
        if not dims or dims[0][0] == 0:
            return [1.0] * self.level_count
        w0 = float(dims[0][0])
        downs: List[float] = []
        for (w, _h) in dims:
            if w == 0:
                downs.append(1.0)
            else:
                downs.append(w0 / float(w))
        # Ensure level0 is 1.0-ish
        if downs:
            downs[0] = 1.0
        return downs

    def read_region(self, location_xy: PointXY, level: int, size_wh: SizeWH):
        """
        Read patch from a TIFF level.

        Convention:
          - location_xy given in level-0 coordinates.
          - We convert to requested level using estimated downsample.
        """
        from PIL import Image
        import numpy as np  # type: ignore

        level = int(level)
        w, h = int(size_wh[0]), int(size_wh[1])

        ds = float(self.level_downsamples[level])
        x0, y0 = int(location_xy[0]), int(location_xy[1])

        # convert level-0 coords to this level coords
        xL = int(round(x0 / ds))
        yL = int(round(y0 / ds))

        page = self._levels[level]
        arr = page.asarray()

        # arr shape could be (h, w) or (h, w, c)
        y1 = max(0, yL)
        x1 = max(0, xL)
        y2 = max(y1, y1 + h)
        x2 = max(x1, x1 + w)

        patch = arr[y1:y2, x1:x2]

        # pad if out-of-bounds
        if patch.shape[0] != h or patch.shape[1] != w:
            if patch.ndim == 2:
                out = np.zeros((h, w), dtype=patch.dtype)
                out[: patch.shape[0], : patch.shape[1]] = patch
            else:
                c = patch.shape[2] if patch.ndim == 3 else 3
                out = np.zeros((h, w, c), dtype=patch.dtype)
                out[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = patch
            patch = out

        # to PIL RGB
        if patch.ndim == 2:
            img = Image.fromarray(patch)
            return img.convert("RGB")
        else:
            # if RGBA or other, convert to RGB via PIL
            img = Image.fromarray(patch)
            return img.convert("RGB")


def _try_tifffile(path: Path) -> Optional[SlideReader]:
    # Lazy import: do NOT import tifffile at module import time
    try:
        import tifffile  # type: ignore
    except Exception as e:
        logger.debug("tifffile import failed (%s).", repr(e))
        return None

    tf = None
    try:
        tf = tifffile.TiffFile(str(path))

        # Try to interpret pyramid levels:
        # Prefer series[0].levels if present, else pages.
        levels = []
        if getattr(tf, "series", None):
            s0 = tf.series[0]
            if hasattr(s0, "levels") and s0.levels:
                levels = list(s0.levels)
            else:
                # some files have multi-page pyramid
                levels = list(tf.pages)
        else:
            levels = list(tf.pages)

        if not levels:
            tf.close()
            return None

        reader = TiffFileReader(tf, levels)

        try:
            logger.info(
                "tifffile levels=%d level0=%s",
                reader.level_count,
                reader.level_dimensions[0] if reader.level_count > 0 else None,
            )
        except Exception:
            pass

        return reader

    except Exception as e:
        logger.debug("tifffile failed to open %s (%s).", path, repr(e))
        try:
            if tf is not None:
                tf.close()
        except Exception:
            pass
        return None
