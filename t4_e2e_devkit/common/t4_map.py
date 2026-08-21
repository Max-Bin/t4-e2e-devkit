"""T4-specific Lanelet2 map access.

The model-facing map remains the compact scene-local tensor. This module is a
separate, optional map facade for consumers that need stable source IDs,
lanelet geometry or graph connectivity. It reads the original Lanelet2 OSM
file referenced by a scene's ``area_map`` metadata and never serializes IDs
into the numeric tensor contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from shapely.geometry import Point

from t4_e2e_devkit.common.dataclasses import MapObjectMatch

# Re-exported: the records, the geometry helpers and the parser moved to their own
# modules, and ``common.t4_map`` stays the import path every caller already uses.
from t4_e2e_devkit.common.t4_map_geometry import (  # noqa: F401
    _geometry_score,
    _local_to_global,
    _normalize_object_type,
    _object_geometry,
    _polyline_score,
    _repair_polygon,
    _resample,
    _row_points,
    _source_geometry,
)
from t4_e2e_devkit.common.t4_map_matching import MapMatcher
from t4_e2e_devkit.common.t4_map_parse import _cached_parse, _ParsedMap  # noqa: F401
from t4_e2e_devkit.common.t4_map_types import T4Lanelet, T4MapObject  # noqa: F401


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
        # The row-matching algorithm and its spatial indexes live in
        # t4_map_matching; this class stays the vocabulary over the parse.
        self._matcher = MapMatcher(self._lanes, self.source_label)

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
            return tuple(self._lanes) + tuple(
                sorted(self._objects, key=lambda obj: (obj.object_type, obj.id))
            )
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
        """Match centerline rows to lanelet IDs, with the evidence for each row.

        :param segments: ``[N, P, >=2]`` rows in the window's centre frame.
        :param center_pose: ``[x, y, cos, sin]`` of that frame in the map frame.
        :param layer: the tensor layer being matched -- ``lanes`` or
            ``route_lanes`` -- recorded in each match.
        :param frame_index: recorded in each match, for a windowed caller.
        :param allowed_ids: restrict candidates, e.g. to a route.
        :param max_distance: the score above which a row stays unmatched.
        :param candidate_limit: how many near candidates to record per row.
        :return: one :class:`MapObjectMatch` per row.
        """
        return self._matcher.match_local_centerlines_detailed(
            segments,
            center_pose,
            layer=layer,
            frame_index=frame_index,
            allowed_ids=allowed_ids,
            max_distance=max_distance,
            candidate_limit=candidate_limit,
        )

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

        Which objects are candidates for a layer is this class's business -- it
        owns the object index -- and the geometry is the matcher's.

        :param segments: ``[N, P, >=2]`` rows in the window's centre frame.
        :param center_pose: ``[x, y, cos, sin]`` of that frame in the map frame.
        :param layer: the tensor layer being matched, recorded in each match.
        :param frame_index: recorded in each match, for a windowed caller.
        :param object_types: candidate types; a layer-appropriate default
            otherwise.
        :param max_distance: the score above which a row stays unmatched.
        :param candidate_limit: how many near candidates to record per row.
        :return: one :class:`MapObjectMatch` per row.
        """
        normalized_layer = str(layer)
        if object_types is None:
            if normalized_layer in {"polygons", "areas"}:
                requested: Sequence[str] = ("area", "drivable_area", "crosswalk")
            else:
                requested = ("line_string", "stop_line", "traffic_light", "regulatory_element")
        else:
            requested = tuple(object_types)
        candidates = tuple(
            obj for object_type in requested for obj in self.get_objects(object_type)
        )
        return self._matcher.match_local_geometries_detailed(
            segments,
            center_pose,
            layer=normalized_layer,
            candidates=candidates,
            frame_index=frame_index,
            max_distance=max_distance,
            candidate_limit=candidate_limit,
        )

    def unmatched_rows(
        self,
        segments: np.ndarray,
        *,
        layer: str,
        frame_index: Optional[int] = None,
        reason: str = "unsupported_source_type",
    ) -> tuple[MapObjectMatch, ...]:
        """Describe rows whose source-object type is not recovered.

        A layer the matcher has no vocabulary for still owes the caller one
        record per row, saying so, rather than a shorter list that reads as
        matched rows.

        :param segments: ``[N, P, >=2]`` rows in the window's centre frame.
        :param layer: the tensor layer, recorded in each match.
        :param frame_index: recorded in each match, for a windowed caller.
        :param reason: the reason to record.
        :return: one unmatched :class:`MapObjectMatch` per row.
        """
        return self._matcher.unmatched_rows(
            segments, layer=layer, frame_index=frame_index, reason=reason
        )

    def get_proximal_lanes(self, point: Sequence[float], radius: float) -> tuple[T4Lanelet, ...]:
        """Lanelets whose polygon lies within ``radius`` of a map-frame point.

        :param point: ``(x, y)`` in the map frame.
        :param radius: search radius in metres.
        :return: the lanelets, nearest first.
        """
        return self._matcher.get_proximal_lanes(point, radius)

    def get_nearest_lane(self, point: Sequence[float]) -> Optional[T4Lanelet]:
        """The lanelet whose polygon is nearest a map-frame point.

        :param point: ``(x, y)`` in the map frame.
        :return: the nearest lanelet, or ``None`` when the map has none.
        """
        return self._matcher.get_nearest_lane(point)


__all__ = [
    "MapObjectMatch",
    "T4Lanelet",
    "T4MapAPI",
    "T4MapObject",
    "resolve_t4_map_path",
]
