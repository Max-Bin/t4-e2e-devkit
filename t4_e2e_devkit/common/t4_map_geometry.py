"""Geometry the map layers share: repair, resampling, scoring, row reading.

These are the pure functions of the map stack -- no map, no API, no parse state,
just shapely and numpy.  Both the parser and the query API used them, which is
why they are here rather than private to either: a scoring rule that differs
between "what the parser accepted" and "what the matcher measures" is a bug
waiting to be written.

Every name stays importable from ``common.t4_map``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from t4_e2e_devkit.common.t4_map_types import T4Lanelet, T4MapObject


def _repair_polygon(points: np.ndarray) -> Optional[Polygon]:
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if polygon.is_empty:
        return None
    if polygon.is_valid:
        return polygon
    repaired = polygon.buffer(0)
    if isinstance(repaired, Polygon):
        return repaired
    if hasattr(repaired, "geoms"):
        polygons = [geometry for geometry in repaired.geoms if isinstance(geometry, Polygon)]
        if polygons:
            return max(polygons, key=lambda geometry: geometry.area)
    return None


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if distances[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [np.interp(targets, distances, points[:, axis]) for axis in range(points.shape[1])]
    )


def _local_to_global(points: np.ndarray, center_pose: np.ndarray) -> np.ndarray:
    c = float(center_pose[2])
    s = float(center_pose[3])
    x = points[:, 0] * c - points[:, 1] * s + float(center_pose[0])
    y = points[:, 0] * s + points[:, 1] * c + float(center_pose[1])
    return np.column_stack((x, y))


def _normalize_object_type(object_type: Optional[str]) -> Optional[str]:
    if object_type is None:
        return None
    value = str(object_type).strip().lower().replace("-", "_")
    return {
        "lane": "lanelet",
        "lanes": "lanelet",
        "lanelets": "lanelet",
        "polygon": "area",
        "polygons": "area",
        "line": "line_string",
        "lines": "line_string",
        "linestring": "line_string",
        "line_strings": "line_string",
        "crosswalks": "crosswalk",
        "stop_lines": "stop_line",
        "traffic_lights": "traffic_light",
        "regulatory_elements": "regulatory_element",
        "drivable_areas": "drivable_area",
        "lane_connectors": "lane_connector",
        "roadblocks": "roadblock",
        "roadblock_connectors": "roadblock_connector",
        "intersections": "intersection",
        "sidewalk": "walkway",
        "sidewalks": "walkway",
        "parking": "carpark_area",
    }.get(value, value)


def _object_geometry(obj: T4Lanelet | T4MapObject) -> BaseGeometry:
    return obj.polygon if isinstance(obj, T4Lanelet) else obj.geometry


def _source_geometry(obj: "T4Lanelet | T4MapObject") -> BaseGeometry:
    """The shapely geometry of a candidate, whichever kind of object it is.

    A lanelet carries ``polygon`` and ``centerline``; only a source object
    carries ``geometry``.  Reading ``geometry`` off a candidate therefore raised
    AttributeError for every match against lanelets -- which is the most natural
    thing to match a lane tensor against.

    A lanelet scores through its centerline rather than its polygon: the score is
    a symmetric Hausdorff distance, so matching a centre-line row against the
    lane's area costs half the lane width for free (measured 1.98 m on a real
    lanelet, against 0.0 for the centerline), which is most of the default 3 m
    threshold. The polygon is the fallback for a lanelet whose centerline is too
    short to be a line.

    :param obj: a lanelet or a source object.
    :return: the geometry to score against.
    """
    geometry = getattr(obj, "geometry", None)
    if geometry is not None:
        return geometry
    centerline = np.asarray(getattr(obj, "centerline", ()), dtype=np.float64)
    if centerline.ndim == 2 and len(centerline) >= 2:
        return LineString(centerline[:, :2])
    return obj.polygon


def _geometry_score(query: BaseGeometry, reference: BaseGeometry) -> float:
    """Symmetric geometric distance used for source-ID recovery."""

    try:
        return float(query.hausdorff_distance(reference))
    except (TypeError, ValueError, AttributeError):
        return float(query.distance(reference))


def _row_points(row: np.ndarray) -> tuple[np.ndarray, bool]:
    """Trim trailing padding without discarding a valid point at the origin."""
    values = np.asarray(row, dtype=np.float64)
    populated = np.isfinite(values).all(axis=1) & (np.abs(values).sum(axis=1) > 1e-6)
    if not populated.any():
        return np.empty((0, 2), dtype=np.float64), False
    last = int(np.flatnonzero(populated)[-1])
    points = values[: last + 1, :2]
    points = points[np.isfinite(points).all(axis=1)]
    return points, True


def _polyline_score(points: np.ndarray, reference: np.ndarray) -> float:
    count = max(2, min(40, len(points)))
    source = _resample(points, count)
    target = _resample(reference, count)
    same = float(np.linalg.norm(source - target, axis=1).mean())
    reverse = float(np.linalg.norm(source - target[::-1], axis=1).mean())
    return min(same, reverse)
