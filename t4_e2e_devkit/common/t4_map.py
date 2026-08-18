"""T4-specific Lanelet2 map access.

The model-facing map remains the compact scene-local tensor. This module is a
separate, optional map facade for consumers that need stable source IDs,
lanelet geometry or graph connectivity. It reads the original Lanelet2 OSM
file referenced by a scene's ``area_map`` metadata and never serializes IDs
into the numeric tensor contract.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from t4_e2e_devkit.common.dataclasses import MapObjectMatch

_SPEED_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class T4Lanelet:
    """A lanelet reconstructed from one Lanelet2 ``lanelet`` relation."""

    id: str
    left_boundary_id: str
    right_boundary_id: str
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    centerline: np.ndarray
    polygon: Polygon
    speed_limit_mps: Optional[float]
    tags: Mapping[str, str]
    incoming_ids: tuple[str, ...] = ()
    outgoing_ids: tuple[str, ...] = ()
    lanelet_type: str = "lanelet"
    turn_direction: str = "unknown"
    regulatory_element_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class T4MapObject:
    """A source Lanelet2 object other than the lanelet graph.

    ``id`` is the source way/relation/node ID.  The object keeps the original
    tags and geometry so downstream code can use a semantic layer without
    depending on the compact model tensor.  Geometry is expressed in the map
    coordinate frame used by the T4 OSM export.
    """

    id: str
    object_type: str
    geometry: BaseGeometry
    tags: Mapping[str, str]
    source_kind: str
    member_ids: tuple[str, ...] = ()

    @property
    def polygon(self) -> Optional[BaseGeometry]:
        """Return the polygon geometry when this object has one."""

        return self.geometry if self.geometry.geom_type in {"Polygon", "MultiPolygon"} else None

    @property
    def is_area(self) -> bool:
        """:return: whether the object is represented by an areal geometry."""

        return self.geometry.geom_type in {"Polygon", "MultiPolygon"}


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
        left_id = next((ref for kind, role, ref in members if kind == "way" and role == "left"), None)
        right_id = next((ref for kind, role, ref in members if kind == "way" and role == "right"), None)
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
                    if kind == "relation"
                    or "regulatory" in role.lower()
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


def _way_points(refs: Sequence[str], nodes: Mapping[str, tuple[float, float]]) -> Optional[np.ndarray]:
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
    value = str(
        tags.get("turn_direction", tags.get("turn", tags.get("lane_connector_type", "")))
    ).strip().lower()
    if value in {"left", "right", "straight", "uturn", "u_turn", "u-turn"}:
        return "uturn" if value in {"u_turn", "u-turn"} else value
    return "unknown"


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


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if distances[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [np.interp(targets, distances, points[:, axis]) for axis in range(points.shape[1])]
    )


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


def resolve_t4_map_path(
    scene_dir: str | Path,
    maps_root: str | Path | None = None,
    *,
    strict: bool = False,
) -> Optional[Path]:
    """Resolve a scene's ``area_map`` metadata to ``lanelet2_map.osm``."""
    scene = Path(scene_dir)
    metadata_path = scene / "metadata.json"
    area_map: Mapping[str, Any] = {}
    if metadata_path.is_file():
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping) and isinstance(value.get("area_map"), Mapping):
                area_map = value["area_map"]
        except (OSError, json.JSONDecodeError):
            pass
    map_id = area_map.get("id")
    version_id = area_map.get("version_id")
    if map_id is None or version_id is None:
        if strict:
            raise ValueError(f"{scene}: metadata.json has no complete area_map identity")
        return None

    if maps_root is None:
        # For the standard layout, .../tier4/t4_dataset/<scene> sits beside
        # .../tier4/maps. The explicit config remains the portable option.
        try:
            maps = scene.parents[3].parent / "maps"
        except IndexError:
            maps = scene.parent / "maps"
    else:
        maps = Path(maps_root).expanduser()
    candidates = (
        maps / str(map_id) / str(version_id) / "lanelet2_map.osm",
        maps / str(version_id) / "lanelet2_map.osm",
        maps / "lanelet2_map.osm",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if strict:
        raise FileNotFoundError(
            f"{scene}: no Lanelet2 map for area_map={map_id}/{version_id}; checked {list(candidates)}"
        )
    return None


class T4MapAPI:
    """Stable-ID queries over one T4 Lanelet2 map."""

    def __init__(
        self,
        osm_path: str | Path,
        *,
        route_lane_ids: Sequence[str] = (),
        map_name: Optional[str] = None,
    ) -> None:
        self.osm_path = Path(osm_path).expanduser().resolve()
        if not self.osm_path.is_file():
            raise FileNotFoundError(f"T4 Lanelet2 map not found: {self.osm_path}")
        parsed = _cached_parse(str(self.osm_path))
        self._lanes = parsed.lanes
        self._by_id = {lane.id: lane for lane in self._lanes}
        self._objects = parsed.objects
        self._objects_by_type: dict[str, tuple[T4MapObject, ...]] = {}
        for object_type in {obj.object_type for obj in self._objects}:
            self._objects_by_type[object_type] = tuple(
                obj for obj in self._objects if obj.object_type == object_type
            )
        self.route_lane_ids = tuple(str(value) for value in route_lane_ids)
        self._map_name = map_name or self.osm_path.parent.name
        self._polygon_tree: Optional[STRtree] = None
        self._polygon_geometries: Optional[tuple[Polygon, ...]] = None
        self._centerline_tree: Optional[STRtree] = None

    @classmethod
    def from_scene(
        cls,
        scene_dir: str | Path,
        maps_root: str | Path | None = None,
        *,
        strict: bool = False,
        route_lane_ids: Sequence[str] = (),
    ) -> Optional["T4MapAPI"]:
        path = resolve_t4_map_path(scene_dir, maps_root, strict=strict)
        if path is None:
            return None
        return cls(path, route_lane_ids=route_lane_ids)

    @property
    def map_name(self) -> str:
        return self._map_name

    @property
    def source_label(self) -> str:
        """A portable source-file label safe for logs and serialized metadata."""
        return self.osm_path.name

    @property
    def lanes(self) -> tuple[T4Lanelet, ...]:
        return self._lanes

    @property
    def objects(self) -> tuple[T4MapObject, ...]:
        """:return: all non-lanelet source objects recovered from the OSM file."""

        return self._objects

    @property
    def available_object_types(self) -> tuple[str, ...]:
        """:return: semantic object types available in this map."""

        types = {"lanelet"}
        types.update(self._objects_by_type)
        if self.get_lane_connectors():
            types.add("lane_connector")
        if any(obj.tags.get("type", "") == "regulatory_element" for obj in self._objects):
            types.add("regulatory_element")
        return tuple(sorted(types))

    def available_ids(self, object_type: Optional[str] = None) -> tuple[str, ...]:
        """Return stable source IDs, optionally restricted to a semantic type."""

        return tuple(obj.id for obj in self.get_objects(object_type))

    def get_objects(self, object_type: Optional[str] = None) -> tuple[T4Lanelet | T4MapObject, ...]:
        """Return lanelets and semantic source objects for a requested type.

        ``lane``/``lanes`` are accepted as aliases for ``lanelet``.  With no
        type, lanelets are returned first followed by source objects ordered by
        their source ID; this makes the result deterministic across parses.
        """

        raw_type = None if object_type is None else str(object_type).strip().lower()
        normalized = _normalize_object_type(object_type)
        if normalized is None:
            return tuple(self._lanes) + tuple(sorted(self._objects, key=lambda obj: (obj.object_type, obj.id)))
        if normalized == "lanelet":
            return self._lanes
        if normalized == "lane_connector":
            return self.get_lane_connectors()
        if raw_type in {"polygon", "polygons", "areas"}:
            return tuple(obj for obj in self._objects if obj.is_area)
        if normalized == "regulatory_element":
            return tuple(
                obj
                for obj in self._objects
                if obj.object_type == "regulatory_element"
                or obj.tags.get("type", "") == "regulatory_element"
            )
        return self._objects_by_type.get(normalized, ())

    def query_objects(
        self,
        point: Sequence[float],
        radius: float,
        object_types: Optional[Sequence[str] | str] = None,
    ) -> tuple[T4Lanelet | T4MapObject, ...]:
        """Query semantic objects whose geometry is within ``radius`` meters."""

        if radius < 0.0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        query = Point(float(point[0]), float(point[1]))
        if object_types is None:
            candidates = self.get_objects()
        elif isinstance(object_types, str):
            candidates = self.get_objects(object_types)
        else:
            candidates = tuple(
                obj for object_type in object_types for obj in self.get_objects(object_type)
            )
        unique: dict[tuple[str, str], T4Lanelet | T4MapObject] = {
            ("lanelet" if isinstance(obj, T4Lanelet) else obj.object_type, obj.id): obj
            for obj in candidates
        }
        nearby = [
            obj
            for obj in unique.values()
            if _object_geometry(obj).distance(query) <= float(radius) + 1.0e-9
        ]
        return tuple(
            sorted(
                nearby,
                key=lambda obj: (
                    _object_geometry(obj).distance(query),
                    obj.id,
                ),
            )
        )

    def get_crosswalks(self) -> tuple[T4MapObject, ...]:
        return self._objects_by_type.get("crosswalk", ())

    def get_stop_lines(self) -> tuple[T4MapObject, ...]:
        return self._objects_by_type.get("stop_line", ())

    def get_traffic_lights(self) -> tuple[T4MapObject, ...]:
        return self._objects_by_type.get("traffic_light", ())

    def get_regulatory_elements(self) -> tuple[T4MapObject, ...]:
        return tuple(self.get_objects("regulatory_element"))

    def get_drivable_areas(self) -> tuple[T4MapObject, ...]:
        return self._objects_by_type.get("drivable_area", ())

    def get_lane_connectors(self) -> tuple[T4Lanelet, ...]:
        """Return lanelets marked as connectors by their source tags."""

        return tuple(lane for lane in self._lanes if lane.lanelet_type == "lane_connector")

    def get_roadblocks(self) -> tuple[T4MapObject, ...]:
        """Return source road-block objects when the export contains them."""

        return self._objects_by_type.get("roadblock", ())

    def get_roadblock_connectors(self) -> tuple[T4MapObject, ...]:
        """Return source road-block connector objects when available."""

        return self._objects_by_type.get("roadblock_connector", ())

    def get_intersections(self) -> tuple[T4MapObject, ...]:
        """Return source intersection objects when available."""

        return self._objects_by_type.get("intersection", ())

    def get_route_lanes(self) -> tuple[T4Lanelet, ...]:
        """Return route lanes in the order provided by route metadata."""

        return tuple(
            lane for lane_id in self.route_lane_ids if (lane := self.get_lane(lane_id)) is not None
        )

    def get_lane_connector_type(self, lane_id: str | int) -> Optional[str]:
        """Return a normalized turn direction for a lane connector."""

        lane = self.get_lane(lane_id)
        if lane is None or lane.lanelet_type != "lane_connector":
            return None
        return lane.turn_direction

    def get_adjacent_lanes(
        self,
        lane_id: str | int,
        *,
        max_distance: float = 0.25,
    ) -> tuple[T4Lanelet, ...]:
        """Return parallel lanelets whose polygons touch or nearly touch."""

        if max_distance < 0.0:
            raise ValueError("max_distance must be non-negative")
        lane = self.get_lane(lane_id)
        if lane is None:
            return ()
        candidates = (
            other
            for other in self._lanes
            if other.id != lane.id and lane.polygon.distance(other.polygon) <= max_distance
        )
        return tuple(sorted(candidates, key=lambda item: item.id))

    def get_lane_chain(
        self,
        lane_id: str | int,
        *,
        max_length: Optional[int] = None,
    ) -> tuple[T4Lanelet, ...]:
        """Follow the deterministic first successor until a branch or end."""

        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive when provided")
        result: list[T4Lanelet] = []
        visited: set[str] = set()
        current = self.get_lane(lane_id)
        while current is not None and current.id not in visited:
            result.append(current)
            visited.add(current.id)
            if max_length is not None and len(result) >= max_length:
                break
            successors = self.get_successors(current.id)
            if len(successors) != 1:
                break
            current = successors[0]
        return tuple(result)

    def get_related_objects(
        self,
        lane_id: str | int,
        object_type: Optional[str] = None,
        *,
        radius: float = 2.0,
    ) -> tuple[T4MapObject, ...]:
        """Find semantic objects associated with a lane geometry.

        Lanelet2 exports differ in whether regulatory members are explicit. A
        geometric fallback keeps this query useful for both forms while never
        inventing source IDs.
        """

        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        lane = self.get_lane(lane_id)
        if lane is None:
            return ()
        requested = self.get_objects(object_type)
        objects = tuple(obj for obj in requested if isinstance(obj, T4MapObject))
        expanded = lane.polygon.buffer(float(radius))
        related_ids = set(lane.regulatory_element_ids)
        return tuple(
            sorted(
                (
                    obj
                    for obj in objects
                    if obj.id in related_ids or expanded.intersects(obj.geometry)
                ),
                key=lambda obj: (lane.polygon.distance(obj.geometry), obj.id),
            )
        )

    def get_stop_lines_for_lane(
        self, lane_id: str | int, *, radius: float = 2.0
    ) -> tuple[T4MapObject, ...]:
        """Return stop lines associated with a lane."""

        return self.get_related_objects(lane_id, "stop_line", radius=radius)

    def get_traffic_lights_for_lane(
        self, lane_id: str | int, *, radius: float = 10.0
    ) -> tuple[T4MapObject, ...]:
        """Return traffic lights associated with a lane."""

        return self.get_related_objects(lane_id, "traffic_light", radius=radius)

    def get_lane(self, lane_id: str | int) -> Optional[T4Lanelet]:
        return self._by_id.get(str(lane_id))

    def get_map_object(
        self,
        object_id: str | int,
        object_type: Optional[str] = None,
    ) -> Optional[T4Lanelet | T4MapObject]:
        normalized = _normalize_object_type(object_type)
        if normalized in (None, "lanelet"):
            lane = self.get_lane(object_id)
            if lane is not None:
                return lane
            if normalized == "lanelet":
                return None
        for obj in self.get_objects(normalized):
            if obj.id == str(object_id):
                return obj
        return None

    def get_proximal_lanes(
        self, point: Sequence[float], radius: float
    ) -> tuple[T4Lanelet, ...]:
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if not self._lanes:
            return ()
        query = Point(float(point[0]), float(point[1]))
        tree = self._get_polygon_tree()
        indices = tree.query(query.buffer(float(radius)))
        lanes = [self._lanes[int(index)] for index in indices]
        return tuple(sorted((lane for lane in lanes if lane.polygon.distance(query) <= radius), key=lambda lane: lane.id))

    def get_nearest_lane(self, point: Sequence[float]) -> Optional[T4Lanelet]:
        if not self._lanes:
            return None
        query = Point(float(point[0]), float(point[1]))
        tree = self._get_polygon_tree()
        index = int(tree.nearest(query))
        return self._lanes[index]

    def get_successors(self, lane_id: str | int) -> tuple[T4Lanelet, ...]:
        lane = self.get_lane(lane_id)
        if lane is None:
            return ()
        return tuple(self._by_id[value] for value in lane.outgoing_ids if value in self._by_id)

    def get_successor_ids(self, lane_id: str | int) -> tuple[str, ...]:
        """Return stable successor IDs without materializing lane objects."""

        lane = self.get_lane(lane_id)
        return () if lane is None else lane.outgoing_ids

    def get_predecessors(self, lane_id: str | int) -> tuple[T4Lanelet, ...]:
        lane = self.get_lane(lane_id)
        if lane is None:
            return ()
        return tuple(self._by_id[value] for value in lane.incoming_ids if value in self._by_id)

    def get_predecessor_ids(self, lane_id: str | int) -> tuple[str, ...]:
        """Return stable predecessor IDs without materializing lane objects."""

        lane = self.get_lane(lane_id)
        return () if lane is None else lane.incoming_ids

    def match_local_centerlines(
        self,
        segments: np.ndarray,
        center_pose: Sequence[float],
        *,
        allowed_ids: Optional[Sequence[str]] = None,
        max_distance: float = 3.0,
    ) -> tuple[Optional[str], ...]:
        """Match scene-local centerline rows to stable lanelet IDs.

        Padded rows and rows without a sufficiently close source geometry return
        ``None``. Matching is geometric and therefore remains valid when the
        exporter changes the tensor's row order or decimates its points.
        """
        return tuple(
            match.source_object_id
            for match in self.match_local_centerlines_detailed(
                segments,
                center_pose,
                allowed_ids=allowed_ids,
                max_distance=max_distance,
            )
        )

    def match_local_centerlines_detailed(
        self,
        segments: np.ndarray,
        center_pose: Sequence[float],
        *,
        layer: str = "lanes",
        frame_index: Optional[int] = None,
        allowed_ids: Optional[Sequence[str]] = None,
        max_distance: float = 3.0,
        candidate_limit: int = 5,
    ) -> tuple[MapObjectMatch, ...]:
        """Match centerline rows and retain evidence for every row.

        ``match_distance_m`` is the mean point-to-point distance after
        resampling both polylines. It is a matching score, not a geometric
        Hausdorff distance. Rows that cannot be assigned a source lane keep
        their nearest candidates and an explicit ``reason``.
        """
        values = np.asarray(segments, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] < 2:
            raise ValueError(f"centerline segments must be [N, P, >=2], got {values.shape}")
        if max_distance < 0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        if candidate_limit <= 0:
            raise ValueError(f"candidate_limit must be positive, got {candidate_limit}")
        if not str(layer):
            raise ValueError("layer must not be empty")
        pose = np.asarray(center_pose, dtype=np.float64).reshape(-1)
        if pose.size < 4:
            raise ValueError(f"center_pose must contain [x, y, cos, sin], got {pose.shape}")
        allowed = None if allowed_ids is None else {str(value) for value in allowed_ids}
        result: list[MapObjectMatch] = []
        for row_index, row in enumerate(values):
            points, has_data = _row_points(row)
            if len(points) < 2:
                result.append(
                    self._map_match(
                        layer=layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="padding" if not has_data else "invalid_geometry",
                    )
                )
                continue
            if not self._lanes:
                result.append(
                    self._map_match(
                        layer=layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="no_source_geometry",
                    )
                )
                continue
            global_points = _local_to_global(points, pose)
            ranked = self._rank_polyline_matches(global_points, allowed, max_distance)
            candidates = tuple(candidate_id for candidate_id, _ in ranked[:candidate_limit])
            if not ranked:
                result.append(
                    self._map_match(
                        layer=layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="no_candidate",
                    )
                )
                continue
            best_id, best_score = ranked[0]
            matched = best_score <= max_distance
            result.append(
                self._map_match(
                    layer=layer,
                    row_index=row_index,
                    frame_index=frame_index,
                    source_object_id=best_id if matched else None,
                    match_distance_m=best_score,
                    candidate_ids=candidates,
                    reason="matched" if matched else "above_threshold",
                )
            )
        return tuple(result)

    def match_local_geometries_detailed(
        self,
        segments: np.ndarray,
        center_pose: Sequence[float],
        *,
        layer: str,
        frame_index: Optional[int] = None,
        object_types: Optional[Sequence[str]] = None,
        max_distance: float = 3.0,
        candidate_limit: int = 5,
    ) -> tuple[MapObjectMatch, ...]:
        """Match polygon/line tensor rows to source semantic objects.

        The numeric tensor does not carry IDs.  This method therefore records
        the best geometric evidence and leaves rows unmatched when the source
        map is too far away, rather than manufacturing an unstable row index.
        """

        values = np.asarray(segments, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] < 2:
            raise ValueError(f"map segments must be [N, P, >=2], got {values.shape}")
        if max_distance < 0.0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        if candidate_limit <= 0:
            raise ValueError(f"candidate_limit must be positive, got {candidate_limit}")
        pose = np.asarray(center_pose, dtype=np.float64).reshape(-1)
        if pose.size < 4:
            raise ValueError(f"center_pose must contain [x, y, cos, sin], got {pose.shape}")
        normalized_layer = str(layer)
        if object_types is None:
            if normalized_layer in {"polygons", "areas"}:
                requested = ("area", "drivable_area", "crosswalk")
            else:
                requested = ("line_string", "stop_line", "traffic_light", "regulatory_element")
        else:
            requested = tuple(object_types)
        candidates = tuple(
            obj
            for object_type in requested
            for obj in self.get_objects(object_type)
        )
        result: list[MapObjectMatch] = []
        for row_index, row in enumerate(values):
            points, has_data = _row_points(row)
            if len(points) < 2:
                result.append(
                    self._map_match(
                        layer=normalized_layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="padding" if not has_data else "invalid_geometry",
                    )
                )
                continue
            if not candidates:
                result.append(
                    self._map_match(
                        layer=normalized_layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="no_source_geometry",
                    )
                )
                continue
            global_points = _local_to_global(points, pose)
            if normalized_layer in {"polygons", "areas"} and len(global_points) >= 3:
                query_geometry: BaseGeometry = _repair_polygon(global_points) or LineString(global_points)
            else:
                query_geometry = LineString(global_points)
            ranked = sorted(
                (
                    (obj.id, _geometry_score(query_geometry, obj.geometry))
                    for obj in candidates
                ),
                key=lambda item: (item[1], item[0]),
            )
            candidates_ids = tuple(item[0] for item in ranked[:candidate_limit])
            best_id, best_score = ranked[0]
            matched = best_score <= max_distance
            result.append(
                self._map_match(
                    layer=normalized_layer,
                    row_index=row_index,
                    frame_index=frame_index,
                    source_object_id=best_id if matched else None,
                    match_distance_m=best_score,
                    candidate_ids=candidates_ids,
                    reason="matched" if matched else "above_threshold",
                )
            )
        return tuple(result)

    def unmatched_rows(
        self,
        segments: np.ndarray,
        *,
        layer: str,
        frame_index: Optional[int] = None,
        reason: str = "unsupported_source_type",
    ) -> tuple[MapObjectMatch, ...]:
        """Describe rows whose source-object type is not recovered.

        T4 tensors contain polygons and line strings, but the current source
        map parser only has reliable lanelet relation IDs. These rows are
        represented explicitly as unmatched instead of receiving fabricated
        IDs.
        """
        values = np.asarray(segments, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] < 2:
            raise ValueError(f"map segments must be [N, P, >=2], got {values.shape}")
        if not str(layer):
            raise ValueError("layer must not be empty")
        result: list[MapObjectMatch] = []
        for row_index, row in enumerate(values):
            _, has_data = _row_points(row)
            result.append(
                self._map_match(
                    layer=layer,
                    row_index=row_index,
                    frame_index=frame_index,
                    source_object_id=None,
                    match_distance_m=None,
                    candidate_ids=(),
                    reason="padding" if not has_data else reason,
                )
            )
        return tuple(result)

    def _map_match(
        self,
        *,
        layer: str,
        row_index: int,
        frame_index: Optional[int],
        source_object_id: Optional[str],
        match_distance_m: Optional[float],
        candidate_ids: tuple[str, ...],
        reason: str,
    ) -> MapObjectMatch:
        return MapObjectMatch(
            layer=str(layer),
            row_index=int(row_index),
            source_object_id=source_object_id,
            source_path=self.source_label,
            frame_index=None if frame_index is None else int(frame_index),
            match_distance_m=(
                None if match_distance_m is None else float(match_distance_m)
            ),
            candidate_ids=tuple(candidate_ids),
            reason=str(reason),
        )

    def _rank_polyline_matches(
        self,
        points: np.ndarray,
        allowed: Optional[set[str]],
        max_distance: float,
    ) -> list[tuple[str, float]]:
        line = LineString(points)
        if self._centerline_tree is None:
            geometries = tuple(LineString(lane.centerline) for lane in self._lanes)
            self._centerline_geometries = geometries
            self._centerline_tree = STRtree(geometries)
        candidate_indices = self._centerline_tree.query(line.buffer(max_distance))
        if len(candidate_indices) == 0:
            candidate_indices = [self._centerline_tree.nearest(line)]
        ranked: list[tuple[str, float]] = []
        for index_value in candidate_indices:
            lane = self._lanes[int(index_value)]
            if allowed is not None and lane.id not in allowed:
                continue
            score = _polyline_score(points, lane.centerline)
            ranked.append((lane.id, score))
        return sorted(ranked, key=lambda item: (item[1], item[0]))

    def _get_polygon_tree(self) -> STRtree:
        if self._polygon_tree is None:
            geometries = tuple(lane.polygon for lane in self._lanes)
            self._polygon_geometries = geometries
            self._polygon_tree = STRtree(geometries)
        return self._polygon_tree


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


__all__ = [
    "MapObjectMatch",
    "T4Lanelet",
    "T4MapAPI",
    "T4MapObject",
    "resolve_t4_map_path",
]
