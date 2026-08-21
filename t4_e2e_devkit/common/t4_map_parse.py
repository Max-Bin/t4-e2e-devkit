"""Reading a Lanelet2 OSM file into the devkit's map records.

One direction only: XML in, :class:`~t4_e2e_devkit.common.t4_map_types.T4Lanelet`
and :class:`~t4_e2e_devkit.common.t4_map_types.T4MapObject` out, plus the lane
graph recovered from endpoint adjacency.  Nothing here queries a map, which is
why it is a module and not part of the API: the parse is cached per file
(:func:`cached_parse`), and a query object is cheap to build on top of it.

The tag vocabulary is the part that ages: which OSM tags mean "crosswalk" or
"drivable area" is a property of the exporter, and keeping it in one file makes
the next exporter change a diff in one place.

Split out of ``common.t4_map``; the entry points stay importable from there.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from t4_e2e_devkit.common.t4_map_geometry import _repair_polygon, _resample
from t4_e2e_devkit.common.t4_map_types import T4Lanelet, T4MapObject

_SPEED_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class _ParsedMap:
    lanes: tuple[T4Lanelet, ...]
    objects: tuple[T4MapObject, ...]


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _tags(element: ET.Element) -> dict[str, str]:
    return {
        str(child.attrib["k"]): str(child.attrib.get("v", ""))
        for child in element
        if _tag_name(child) == "tag" and "k" in child.attrib
    }


def _parse_osm(path: Path) -> _ParsedMap:
    nodes: dict[str, tuple[float, float]] = {}
    node_tags: dict[str, dict[str, str]] = {}
    ways: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {}
    relations: list[tuple[str, tuple[tuple[str, str, str], ...], dict[str, str]]] = []

    try:
        stream = ET.iterparse(path, events=("end",))
        for _, element in stream:
            kind = _tag_name(element)
            if kind == "node":
                tags = _tags(element)
                try:
                    x = float(tags["local_x"])
                    y = float(tags["local_y"])
                except (KeyError, TypeError, ValueError):
                    element.clear()
                    continue
                node_id = str(element.attrib["id"])
                nodes[node_id] = (x, y)
                node_tags[node_id] = tags
                element.clear()
            elif kind == "way":
                refs = tuple(
                    str(child.attrib["ref"])
                    for child in element
                    if _tag_name(child) == "nd" and "ref" in child.attrib
                )
                ways[str(element.attrib["id"])] = (refs, _tags(element))
                element.clear()
            elif kind == "relation":
                members = tuple(
                    (
                        str(child.attrib.get("type", "")),
                        str(child.attrib.get("role", "")),
                        str(child.attrib.get("ref", "")),
                    )
                    for child in element
                    if _tag_name(child) == "member"
                )
                relations.append((str(element.attrib["id"]), members, _tags(element)))
                element.clear()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"cannot parse Lanelet2 map {path}: {error}") from error

    lanes: list[T4Lanelet] = []
    for relation_id, members, tags in relations:
        if tags.get("type") != "lanelet":
            continue
        left_id = next(
            (ref for kind, role, ref in members if kind == "way" and role == "left"), None
        )
        right_id = next(
            (ref for kind, role, ref in members if kind == "way" and role == "right"), None
        )
        if left_id is None or right_id is None:
            continue
        left = _way_points(ways.get(left_id, ((), {}))[0], nodes)
        right = _way_points(ways.get(right_id, ((), {}))[0], nodes)
        if left is None or right is None or len(left) < 2 or len(right) < 2:
            continue
        if _endpoint_cost(left, right[::-1]) < _endpoint_cost(left, right):
            right = right[::-1].copy()
        count = max(2, min(40, max(len(left), len(right))))
        left_resampled = _resample(left, count)
        right_resampled = _resample(right, count)
        centerline = (left_resampled + right_resampled) / 2.0
        polygon = _lane_polygon(left_resampled, right_resampled)
        lanes.append(
            T4Lanelet(
                id=relation_id,
                left_boundary_id=left_id,
                right_boundary_id=right_id,
                left_boundary=np.ascontiguousarray(left_resampled),
                right_boundary=np.ascontiguousarray(right_resampled),
                centerline=np.ascontiguousarray(centerline),
                polygon=polygon,
                speed_limit_mps=_speed_limit_mps(tags),
                tags=dict(tags),
                lanelet_type=_lanelet_type(tags),
                turn_direction=_turn_direction(tags),
                regulatory_element_ids=tuple(
                    ref
                    for kind, role, ref in members
                    if kind == "relation" or "regulatory" in role.lower()
                ),
            )
        )

    _connect_lanes(lanes)
    lane_boundary_ids = {
        boundary_id
        for lane in lanes
        for boundary_id in (lane.left_boundary_id, lane.right_boundary_id)
    }
    objects: list[T4MapObject] = []
    for node_id, point in nodes.items():
        tags = node_tags.get(node_id, {})
        object_type = _classify_object_type(tags)
        if object_type is not None:
            objects.append(
                T4MapObject(
                    id=node_id,
                    object_type=object_type,
                    geometry=Point(point),
                    tags=dict(tags),
                    source_kind="node",
                )
            )

    for way_id, (refs, tags) in ways.items():
        if way_id in lane_boundary_ids:
            continue
        object_type = _classify_object_type(tags) or ("line_string" if tags else None)
        if object_type is None:
            continue
        geometry = _way_geometry(refs, nodes, object_type)
        if geometry is not None:
            objects.append(
                T4MapObject(
                    id=way_id,
                    object_type=object_type,
                    geometry=geometry,
                    tags=dict(tags),
                    source_kind="way",
                    member_ids=tuple(refs),
                )
            )

    for relation_id, members, tags in relations:
        if tags.get("type") == "lanelet":
            continue
        object_type = _classify_object_type(tags)
        if object_type is None:
            continue
        geometry = _relation_geometry(members, ways, nodes, object_type)
        if geometry is not None and not geometry.is_empty:
            objects.append(
                T4MapObject(
                    id=relation_id,
                    object_type=object_type,
                    geometry=geometry,
                    tags=dict(tags),
                    source_kind="relation",
                    member_ids=tuple(ref for _, _, ref in members),
                )
            )

    return _ParsedMap(tuple(lanes), tuple(objects))


def _way_points(
    refs: Sequence[str], nodes: Mapping[str, tuple[float, float]]
) -> Optional[np.ndarray]:
    points = [nodes[ref] for ref in refs if ref in nodes]
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.float64)


def _classify_object_type(tags: Mapping[str, str]) -> Optional[str]:
    """Map Lanelet2 tags to a small, stable semantic vocabulary."""

    if not tags:
        return None
    values = " ".join(
        str(tags.get(key, "")).strip().lower()
        for key in ("type", "subtype", "role", "classification", "kind")
    )
    if "roadblock_connector" in values or "roadblock connector" in values:
        return "roadblock_connector"
    if "roadblock" in values:
        return "roadblock"
    if "intersection" in values:
        return "intersection"
    if "lane_connector" in values or "lane connector" in values:
        return "lane_connector"
    if "traffic_light" in values or "traffic light" in values:
        return "traffic_light"
    if "stop_line" in values or "stop line" in values:
        return "stop_line"
    if "crosswalk" in values or "crossing" in values or "zebra" in values:
        return "crosswalk"
    if "drivable_area" in values or tags.get("drivable_area", "").lower() in {"1", "yes", "true"}:
        return "drivable_area"
    if "speed_bump" in values or "speed bump" in values:
        return "speed_bump"
    if "stop_sign" in values or "stop sign" in values:
        return "stop_sign"
    if "yield" in values:
        return "yield"
    if "walkway" in values or "sidewalk" in values:
        return "walkway"
    if "carpark" in values or "car_park" in values or "parking" in values:
        return "carpark_area"
    if "pudo" in values or "pick_up" in values or "drop_off" in values:
        return "pudo"
    if "turn_stop" in values or "turn stop" in values:
        return "turn_stop"
    if tags.get("area", "").lower() in {"1", "yes", "true"} or "area" in values:
        return "area"
    if (
        tags.get("type", "").lower() in {"line", "line_string", "linestring"}
        or tags.get("subtype", "").lower() in {"line", "line_string", "road_marking"}
        or any(value in values for value in ("road_marking", "curb", "boundary"))
    ):
        return "line_string"
    if tags.get("type", "").lower() == "regulatory_element":
        return "regulatory_element"
    return None


def _lanelet_type(tags: Mapping[str, str]) -> str:
    values = " ".join(
        str(tags.get(key, "")).strip().lower()
        for key in ("type", "subtype", "kind", "lanelet_type", "role")
    )
    return "lane_connector" if "connector" in values else "lanelet"


def _turn_direction(tags: Mapping[str, str]) -> str:
    value = (
        str(tags.get("turn_direction", tags.get("turn", tags.get("lane_connector_type", ""))))
        .strip()
        .lower()
    )
    if value in {"left", "right", "straight", "uturn", "u_turn", "u-turn"}:
        return "uturn" if value in {"u_turn", "u-turn"} else value
    return "unknown"


def _way_geometry(
    refs: Sequence[str],
    nodes: Mapping[str, tuple[float, float]],
    object_type: str,
) -> Optional[BaseGeometry]:
    points = _way_points(refs, nodes)
    if points is None:
        return None
    if object_type in {
        "area",
        "drivable_area",
        "crosswalk",
        "roadblock",
        "roadblock_connector",
        "intersection",
        "carpark_area",
    }:
        polygon = _repair_polygon(points)
        if polygon is not None:
            return polygon
    try:
        return LineString(points)
    except (TypeError, ValueError):
        return None


def _relation_geometry(
    members: Sequence[tuple[str, str, str]],
    ways: Mapping[str, tuple[tuple[str, ...], dict[str, str]]],
    nodes: Mapping[str, tuple[float, float]],
    object_type: str,
) -> Optional[BaseGeometry]:
    geometries: list[BaseGeometry] = []
    for member_kind, _, member_id in members:
        if member_kind == "way" and member_id in ways:
            refs, _ = ways[member_id]
            geometry = _way_geometry(refs, nodes, object_type)
            if geometry is not None:
                geometries.append(geometry)
        elif member_kind == "node" and member_id in nodes:
            geometries.append(Point(nodes[member_id]))
    if not geometries:
        return None
    if object_type in {
        "area",
        "drivable_area",
        "crosswalk",
        "roadblock",
        "roadblock_connector",
        "intersection",
        "carpark_area",
    }:
        polygons = [geometry for geometry in geometries if isinstance(geometry, Polygon)]
        if polygons:
            return unary_union(polygons)
    return unary_union(geometries)


def _endpoint_cost(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1]))


def _lane_polygon(left: np.ndarray, right: np.ndarray) -> Polygon:
    polygon = Polygon(np.vstack((left, right[::-1])))
    if polygon.is_valid and not polygon.is_empty:
        return polygon
    repaired = polygon.buffer(0)
    if repaired.geom_type == "Polygon":
        return repaired
    if hasattr(repaired, "geoms") and repaired.geoms:
        largest = max(repaired.geoms, key=lambda geometry: geometry.area)
        if isinstance(largest, Polygon):
            return largest
    return polygon.convex_hull


def _speed_limit_mps(tags: Mapping[str, str]) -> Optional[float]:
    value = tags.get("speed_limit")
    if value in (None, ""):
        return None
    match = _SPEED_RE.search(str(value))
    if match is None:
        return None
    number = float(match.group())
    unit = str(tags.get("speed_limit_unit", tags.get("speed_limit_units", "km/h"))).lower()
    if "mph" in unit:
        return number * 0.44704
    if "m/s" in unit or "mps" in unit:
        return number
    return number / 3.6


def _endpoint_bucket(point: np.ndarray, tolerance: float) -> tuple[int, int]:
    return (math.floor(float(point[0]) / tolerance), math.floor(float(point[1]) / tolerance))


def _connect_lanes(lanes: list[T4Lanelet], tolerance: float = 2.0) -> None:
    starts: dict[tuple[int, int], list[int]] = {}
    for index, lane in enumerate(lanes):
        starts.setdefault(_endpoint_bucket(lane.centerline[0], tolerance), []).append(index)

    incoming: list[set[str]] = [set() for _ in lanes]
    outgoing: list[set[str]] = [set() for _ in lanes]
    for index, lane in enumerate(lanes):
        end = lane.centerline[-1]
        heading = lane.centerline[-1] - lane.centerline[-2]
        norm = float(np.linalg.norm(heading))
        if norm > 1e-9:
            heading = heading / norm
        bucket = _endpoint_bucket(end, tolerance)
        for bx in range(bucket[0] - 1, bucket[0] + 2):
            for by in range(bucket[1] - 1, bucket[1] + 2):
                for other_index in starts.get((bx, by), []):
                    if other_index == index:
                        continue
                    other = lanes[other_index]
                    distance = float(np.linalg.norm(end - other.centerline[0]))
                    if distance > tolerance:
                        continue
                    other_heading = other.centerline[1] - other.centerline[0]
                    other_norm = float(np.linalg.norm(other_heading))
                    if norm > 1e-9 and other_norm > 1e-9:
                        other_heading = other_heading / other_norm
                        if float(np.dot(heading, other_heading)) < -0.5:
                            continue
                    outgoing[index].add(other.id)
                    incoming[other_index].add(lane.id)

    for index, lane in enumerate(lanes):
        object.__setattr__(lane, "incoming_ids", tuple(sorted(incoming[index])))
        object.__setattr__(lane, "outgoing_ids", tuple(sorted(outgoing[index])))


@lru_cache(maxsize=8)
def _cached_parse(path: str) -> _ParsedMap:
    return _parse_osm(Path(path))
