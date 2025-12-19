# -*- coding: utf-8 -*-
# @Time    : 2025/12/19 21:00
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: geojson.py
# @Project : WSIRadiomics

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

PointXY = Tuple[float, float]
CellRecord = Dict[str, Any]


def load_cells_from_geojson(
    geojson_path: str | Path,
    *,
    keep_holes: bool = False,
    flatten_multipolygon: bool = True,
    add_feature_id: bool = True,
) -> List[CellRecord]:
    """
    Load per-cell polygons from a GeoJSON file.

    Supported input formats:
      - top-level list[Feature] (QuPath/CellViT++ style)
      - FeatureCollection
      - Feature
      - Polygon / MultiPolygon geometry at root

    Return format (flatten_multipolygon=True):
      [
        {
          "polygon_xy": [(x, y), ...],     # exterior ring only (no closing point)
          "holes_xy": [[(x,y),...], ...],  # only if keep_holes=True and holes exist
          "properties": {...},            # original properties
          "geometry_type": "Polygon" | "MultiPolygon",
          "cell_type": str | None,         # properties.classification.name if present
          "feature_id": str | None,        # optional: original feature id
        },
        ...
      ]

    Notes:
      - Coordinates are expected in pixel space (x, y).
      - If classification info is missing, cell_type is None (no error).
      - For MultiPolygon:
          - flatten_multipolygon=True: returns one record per polygon part
          - flatten_multipolygon=False: returns one record with "polygon_xy_list"
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")

    data = _read_json(path)
    features = _extract_features(data)

    cells: List[CellRecord] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue

        geom = feat.get("geometry", None)
        props = feat.get("properties", {}) or {}

        # ---- Extract cell_type if possible (QuPath style) ----
        # props["classification"] may be {"name": "...", "color": [...]}
        cls = props.get("classification")
        cell_type = cls.get("name") if isinstance(cls, dict) else None

        feat_id = feat.get("id", None) if add_feature_id else None

        if not isinstance(geom, dict):
            continue

        gtype = geom.get("type", None)
        coords = geom.get("coordinates", None)

        if gtype not in ("Polygon", "MultiPolygon"):
            continue
        if coords is None:
            continue

        if gtype == "Polygon":
            recs = _handle_polygon(coords, props, keep_holes=keep_holes)
            for r in recs:
                r["geometry_type"] = "Polygon"
                r["cell_type"] = cell_type
                if feat_id is not None:
                    r["feature_id"] = feat_id
            cells.extend(recs)

        elif gtype == "MultiPolygon":
            recs = _handle_multipolygon(
                coords,
                props,
                keep_holes=keep_holes,
                flatten=flatten_multipolygon,
            )
            for r in recs:
                r["geometry_type"] = "MultiPolygon"
                r["cell_type"] = cell_type
                if feat_id is not None:
                    r["feature_id"] = feat_id
            cells.extend(recs)

    return cells


# -------------------------
# Internal helpers
# -------------------------

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_features(data: Any) -> List[Dict[str, Any]]:
    """
    Normalize input geojson to a list of Feature dicts.

    Supports:
      - top-level list[Feature]             (QuPath/CellViT++ style)
      - FeatureCollection (dict)
      - Feature (dict)
      - Polygon / MultiPolygon geometry at root (dict)
    """
    # --- Case 1: top-level list (your current files) ---
    if isinstance(data, list):
        feats = [x for x in data if isinstance(x, dict)]
        if not feats:
            raise ValueError(
                "GeoJSON top-level is a list, but contains no dict items. "
                f"Got element types: {sorted({type(x).__name__ for x in data})}"
            )
        return feats

    # --- Case 2: standard GeoJSON dict forms ---
    if isinstance(data, dict):
        t = data.get("type", None)

        if t == "FeatureCollection":
            feats = data.get("features", [])
            if not isinstance(feats, list):
                raise ValueError("GeoJSON FeatureCollection.features must be a list.")
            return [f for f in feats if isinstance(f, dict)]

        if t == "Feature":
            return [data]

        # Geometry stored directly at root
        if t in ("Polygon", "MultiPolygon"):
            return [{"type": "Feature", "geometry": data, "properties": {}}]

        # Some exporters omit "type" but still use "features"
        if "features" in data and isinstance(data["features"], list):
            return [f for f in data["features"] if isinstance(f, dict)]

    # --- Friendly error with diagnostics ---
    msg = [
        "Unsupported GeoJSON format.",
        f"Top-level type: {type(data).__name__}",
        "Expected one of: list[Feature], FeatureCollection, Feature, Polygon/MultiPolygon.",
    ]
    if isinstance(data, dict):
        msg.append(f"Top-level keys: {list(data.keys())[:30]}")
        msg.append(f"Top-level 'type': {data.get('type', None)}")
    if isinstance(data, list) and len(data) > 0:
        msg.append(f"Top-level list length: {len(data)}")
        first = data[0]
        msg.append(f"First element type: {type(first).__name__}")
        if isinstance(first, dict):
            msg.append(f"First element keys: {list(first.keys())[:30]}")
            geom = first.get("geometry")
            if isinstance(geom, dict):
                msg.append(f"First element geometry.type: {geom.get('type', None)}")
    raise ValueError(" ".join(str(x) for x in msg))


def _handle_polygon(
    polygon_coords: Any,
    properties: Dict[str, Any],
    *,
    keep_holes: bool,
) -> List[CellRecord]:
    """
    polygon_coords (GeoJSON):
      [
        exterior_ring: [[x,y], [x,y], ...],
        hole1: [[x,y], ...],
        hole2: ...
      ]
    """
    if not isinstance(polygon_coords, list) or len(polygon_coords) == 0:
        return []

    exterior = _as_ring_xy(polygon_coords[0])
    if len(exterior) < 3:
        return []

    rec: CellRecord = {
        "polygon_xy": exterior,
        "properties": properties,
    }

    if keep_holes and len(polygon_coords) > 1:
        holes: List[List[PointXY]] = []
        for hole in polygon_coords[1:]:
            ring = _as_ring_xy(hole)
            if len(ring) >= 3:
                holes.append(ring)
        if holes:
            rec["holes_xy"] = holes

    return [rec]


def _handle_multipolygon(
    multipoly_coords: Any,
    properties: Dict[str, Any],
    *,
    keep_holes: bool,
    flatten: bool,
) -> List[CellRecord]:
    """
    multipoly_coords:
      [
        polygon1_coords,
        polygon2_coords,
        ...
      ]
    where polygon_k_coords is the same format as Polygon coords (list of rings).
    """
    if not isinstance(multipoly_coords, list) or len(multipoly_coords) == 0:
        return []

    if flatten:
        out: List[CellRecord] = []
        for poly in multipoly_coords:
            out.extend(_handle_polygon(poly, properties, keep_holes=keep_holes))
        return out

    # Non-flatten: keep as a single record with list of polygons
    polygon_xy_list: List[List[PointXY]] = []
    holes_xy_list: List[List[List[PointXY]]] = []

    for poly in multipoly_coords:
        recs = _handle_polygon(poly, properties, keep_holes=keep_holes)
        if not recs:
            continue
        rec0 = recs[0]
        polygon_xy_list.append(rec0["polygon_xy"])
        if keep_holes:
            holes_xy_list.append(rec0.get("holes_xy", []))

    rec: CellRecord = {
        "polygon_xy_list": polygon_xy_list,
        "properties": properties,
    }
    if keep_holes:
        rec["holes_xy_list"] = holes_xy_list

    return [rec]


def _as_ring_xy(ring: Any) -> List[PointXY]:
    """
    Convert a GeoJSON ring [[x,y], ...] into [(x,y), ...].
    Drops duplicated closing point if present.
    """
    if not isinstance(ring, list):
        return []

    pts: List[PointXY] = []
    for p in ring:
        if (
            isinstance(p, (list, tuple))
            and len(p) >= 2
            and isinstance(p[0], (int, float))
            and isinstance(p[1], (int, float))
        ):
            pts.append((float(p[0]), float(p[1])))

    # Drop closing point if ring is explicitly closed
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]

    return pts


if __name__ == "__main__":
    # Quick sanity check
    from collections import Counter

    p = Path("/home/huangdn/CellViT-plus-plus/result/陆懿红_cells.geojson")
    cells = load_cells_from_geojson(p)

    print("num_cells:", len(cells))
    print("first keys:", cells[0].keys())
    print("first cell_type:", cells[0].get("cell_type"))

    cnt = Counter(c.get("cell_type") for c in cells)
    print("cell_type Counter (top 20):", cnt.most_common(20))

