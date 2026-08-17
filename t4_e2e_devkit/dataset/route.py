"""Lightweight readers for the route and area-map metadata in a T4 scene."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class T4RoutePrimitive:
    """One primitive from a serialized route segment."""

    id: str
    primitive_type: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class T4RouteSegment:
    """An ordered route segment and its preferred primitive."""

    preferred_primitive: Optional[T4RoutePrimitive]
    primitives: tuple[T4RoutePrimitive, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class T4RouteMetadata:
    """Route identity and geometry-independent metadata for one scene."""

    source_path: str
    start_pose: Optional[tuple[float, ...]]
    goal_pose: Optional[tuple[float, ...]]
    segments: tuple[T4RouteSegment, ...]
    area_map_id: Optional[str]
    area_map_version_id: Optional[str]
    raw: Mapping[str, Any]

    @property
    def primitive_ids(self) -> tuple[str, ...]:
        """All ordered route primitive IDs, preserving segment order."""
        return tuple(
            primitive.id
            for segment in self.segments
            for primitive in segment.primitives
        )

    @property
    def route_lane_ids(self) -> tuple[str, ...]:
        """Ordered lanelet IDs from route primitives."""
        return tuple(
            primitive.id
            for segment in self.segments
            for primitive in segment.primitives
            if primitive.primitive_type.lower() == "lane"
        )

    @property
    def preferred_lane_ids(self) -> tuple[str, ...]:
        """One preferred lane ID per route segment where available."""
        return tuple(
            primitive.id
            for segment in self.segments
            if (primitive := segment.preferred_primitive) is not None
            and primitive.primitive_type.lower() == "lane"
        )

    @property
    def route_roadblock_ids(self) -> tuple[str, ...]:
        """Ordered road-block primitives when the route export contains them."""
        return tuple(
            primitive.id
            for segment in self.segments
            for primitive in segment.primitives
            if primitive.primitive_type.lower() in {"roadblock", "roadblock_connector"}
        )


def _pose(value: Any) -> Optional[tuple[float, ...]]:
    if not isinstance(value, Mapping):
        return None
    position = value.get("position")
    orientation = value.get("orientation")
    if not isinstance(position, Mapping) or not isinstance(orientation, Mapping):
        return None
    try:
        return tuple(
            float(position[key]) for key in ("x", "y", "z")
        ) + tuple(float(orientation[key]) for key in ("x", "y", "z", "w"))
    except (KeyError, TypeError, ValueError):
        return None


def _primitive(value: Any) -> Optional[T4RoutePrimitive]:
    if not isinstance(value, Mapping) or value.get("id") is None:
        return None
    return T4RoutePrimitive(
        id=str(value["id"]),
        primitive_type=str(value.get("primitive_type", "unknown")),
        raw=copy.deepcopy(dict(value)),
    )


def _route_segment(value: Any) -> Optional[T4RouteSegment]:
    if not isinstance(value, Mapping):
        return None
    preferred = _primitive(value.get("preferred_primitive"))
    primitive_values = value.get("primitives", [])
    if not isinstance(primitive_values, list):
        primitive_values = []
    primitives = tuple(
        primitive
        for primitive_value in primitive_values
        if (primitive := _primitive(primitive_value)) is not None
    )
    return T4RouteSegment(
        preferred_primitive=preferred,
        primitives=primitives,
        raw=copy.deepcopy(dict(value)),
    )


def _area_map(scene_dir: Path) -> tuple[Optional[str], Optional[str]]:
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.is_file():
        return None, None
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    area_map = value.get("area_map") if isinstance(value, Mapping) else None
    if not isinstance(area_map, Mapping):
        return None, None
    return (
        None if area_map.get("id") is None else str(area_map["id"]),
        None if area_map.get("version_id") is None else str(area_map["version_id"]),
    )


def load_t4_route(scene_dir: str | Path, *, strict: bool = False) -> Optional[T4RouteMetadata]:
    """Read ``route.json`` and the map identity from ``metadata.json``.

    A scene without a route is valid for readers that only need perception
    inputs, so the default is to return ``None``. Set ``strict=True`` when a
    route is required by a planner or map audit.
    """
    scene = Path(scene_dir)
    path = scene / "route.json"
    if not path.is_file():
        if strict:
            raise FileNotFoundError(f"T4 scene has no route.json: {scene}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        if strict:
            raise ValueError(f"cannot read T4 route file {path}: {error}") from error
        return None
    if not isinstance(value, Mapping):
        if strict:
            raise ValueError(f"T4 route file must contain an object: {path}")
        return None

    segment_values = value.get("segments", [])
    if not isinstance(segment_values, list):
        segment_values = []
    segments = tuple(
        segment
        for segment_value in segment_values
        if (segment := _route_segment(segment_value)) is not None
    )
    area_map_id, area_map_version_id = _area_map(scene)
    return T4RouteMetadata(
        source_path=path.name,
        start_pose=_pose(value.get("start_pose")),
        goal_pose=_pose(value.get("goal_pose")),
        segments=segments,
        area_map_id=area_map_id,
        area_map_version_id=area_map_version_id,
        raw=copy.deepcopy(dict(value)),
    )


__all__ = ["T4RouteMetadata", "T4RoutePrimitive", "T4RouteSegment", "load_t4_route"]
