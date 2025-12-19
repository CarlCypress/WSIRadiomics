# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 20:31
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: params.py
# @Project : WSIRadiomics

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


# -------------------------
# Feature universe (reference)
# -------------------------

# Cell-level feature names (as in your full params.yaml)
_CELL_FEATURE_UNIVERSE: Dict[str, List[str]] = {
    "firstorder": [
        "Energy",
        "TotalEnergy",
        "Entropy",
        "Minimum",
        "Percentile10",
        "Percentile90",
        "Maximum",
        "Mean",
        "Median",
        "InterquartileRange",
        "Range",
        "MeanAbsoluteDeviation",
        "RobustMeanAbsoluteDeviation",
        "RootMeanSquared",
        "StandardDeviation",
        "Skewness",
        "Kurtosis",
        "Variance",
        "Uniformity",
    ],
    "shape": [
        "MeshSurface",
        "PixelSurface",
        "Perimeter",
        "PerimetertoSurfaceRatio",
        "Sphericity",
        "SphericalDisproportion",
        "MaximumDiameter",
        "MajorAxisLength",
        "MinorAxisLength",
        "Elongation",
        "MeanCurvature",
        "GaussianCurvature",
        "MaximumCurvature",
        "MinimumCurvature",
    ],
    # Texture groups: currently "empty => ALL" is allowed, but universe is not specified yet.
    # Keep them as known groups with empty universe; you can fill later.
    "glcm": [],
    "glrlm": [],
    "glszm": [],
    "ngtdm": [],
    "gldm": [],
}

# WSI-level aggregation functions (as in your full params.yaml)
_AGG_FUNCTION_UNIVERSE: List[str] = [
    "Mean",
    "Median",
    "StandardDeviation",
    "InterquartileRange",
    "Skewness",
    "Kurtosis",
]


# -------------------------
# Default params
# -------------------------

def default_params() -> Dict[str, Any]:
    """
    Default configuration used when params.yaml is not provided.

    Your rule:
    - If params_path is empty, default should return the MOST COMPLETE cfg.
      That means: all supported feature groups are enabled and fully expanded to lists.

    Note:
    - For texture groups whose universe is not implemented yet, they will be enabled with [].
      You may later replace [] with full names, or remove these groups if you prefer "not enabled by default".
    """
    return {
        "cell_features": {k: v.copy() for k, v in _CELL_FEATURE_UNIVERSE.items()},
        "aggregation": {
            "functions": _AGG_FUNCTION_UNIVERSE.copy(),
            "by_cell_type": None,
        },
    }


# -------------------------
# Load params.yaml
# -------------------------

def load_params(params_path: Path) -> Dict[str, Any]:
    """
    Load and normalize params.yaml.

    Rules (your design):
    - Missing feature group  -> do NOT compute (key absent in cfg)
    - Empty feature group    -> compute ALL features in the group (expand to full list)
    - Listed feature names   -> compute ONLY listed features (validated)
    """
    params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {params_path}")

    cfg = _read_yaml(params_path)
    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValueError("params.yaml must contain a YAML mapping (dict) at top level.")

    return _normalize_params(cfg)


# -------------------------
# Internal helpers
# -------------------------

def _read_yaml(path: Path) -> Any:
    """Read YAML file using PyYAML."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load params.yaml. "
            "Please install it via `pip install pyyaml`."
        ) from e

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw YAML config into an executable cfg.

    Output format:
    {
        "cell_features": {
            "<group>": [feature_name, ...],   # group enabled
            # missing group key => disabled
        },
        "aggregation": {
            "functions": [func_name, ...],     # enabled
            "by_cell_type": None | [str, ...], # optional grouping
        }
    }
    """
    out: Dict[str, Any] = {}

    # ---- cell_features ----
    if "cell_features" in cfg:
        cell_cfg = cfg["cell_features"]
        if not isinstance(cell_cfg, dict):
            raise ValueError("cell_features must be a mapping (dict).")

        out_cell: Dict[str, List[str]] = {}

        for group, selection in cell_cfg.items():
            # unknown group: allow but treat as error (helps catch typos early)
            if group not in _CELL_FEATURE_UNIVERSE:
                raise ValueError(
                    f"Unknown cell feature group: {group}. "
                    f"Supported groups: {sorted(_CELL_FEATURE_UNIVERSE.keys())}"
                )

            out_cell[group] = _expand_and_validate_cell_group(group, selection)

        out["cell_features"] = out_cell

    # ---- aggregation ----
    if "aggregation" in cfg:
        agg_cfg = cfg["aggregation"]
        if not isinstance(agg_cfg, dict):
            raise ValueError("aggregation must be a mapping (dict).")

        out_agg: Dict[str, Any] = {}

        if "functions" in agg_cfg:
            out_agg["functions"] = _expand_and_validate_agg_functions(agg_cfg.get("functions"))

        if "by_cell_type" in agg_cfg:
            out_agg["by_cell_type"] = _normalize_by_cell_type(agg_cfg.get("by_cell_type"))

        out["aggregation"] = out_agg

    return out


def _expand_and_validate_cell_group(group: str, selection: Any) -> List[str]:
    """
    Expand/validate selection for a cell feature group.

    - selection is None (empty in YAML) => expand to ALL from universe
    - selection is list[str]            => validate every feature exists in universe (if universe not empty)
    """
    universe = _CELL_FEATURE_UNIVERSE[group]

    # Empty => ALL
    if selection is None:
        # If universe is empty (e.g., texture groups not filled yet), return []
        return universe.copy()

    # Listed => validate
    if isinstance(selection, list):
        for x in selection:
            if not isinstance(x, str):
                raise ValueError(f"cell_features.{group} must be a list of strings.")

        # if universe is defined (non-empty), validate names
        if len(universe) > 0:
            unknown = [x for x in selection if x not in universe]
            if unknown:
                raise ValueError(
                    f"Unknown feature name(s) in cell_features.{group}: {unknown}. "
                    f"Supported: {universe}"
                )

        return selection

    raise ValueError(
        f"Invalid value for cell_features.{group}: {selection}. "
        "Use empty/null for ALL, or a list of feature names."
    )


def _expand_and_validate_agg_functions(selection: Any) -> List[str]:
    """
    Expand/validate aggregation.functions.

    - None (empty in YAML) => ALL aggregation functions
    - list[str]            => validate each is supported
    """
    universe = _AGG_FUNCTION_UNIVERSE

    if selection is None:
        return universe.copy()

    if isinstance(selection, list):
        for x in selection:
            if not isinstance(x, str):
                raise ValueError("aggregation.functions must be a list of strings.")
        unknown = [x for x in selection if x not in universe]
        if unknown:
            raise ValueError(
                f"Unknown aggregation function(s): {unknown}. "
                f"Supported: {universe}"
            )
        return selection

    raise ValueError(
        f"Invalid value for aggregation.functions: {selection}. "
        "Use empty/null for ALL, or a list of function names."
    )


def _normalize_by_cell_type(selection: Any) -> Optional[List[str]]:
    """
    Normalize aggregation.by_cell_type:
    - None / empty => None (aggregate across all cells)
    - list[str]    => aggregate by listed cell types
    """
    if selection is None:
        return None

    if isinstance(selection, list):
        for x in selection:
            if not isinstance(x, str):
                raise ValueError("aggregation.by_cell_type must be a list of strings.")
        return selection

    raise ValueError(
        f"Invalid value for aggregation.by_cell_type: {selection}. "
        "Use empty/null for no grouping, or a list of cell type names."
    )

