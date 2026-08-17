"""NuPlan-shaped map facade backed by :class:`T4MapAPI`.

The adapter exposes the standard vector/raster query surface without changing
the T4 tensor contract.  Raster layers are generated from source geometry on
demand when the T4 export does not carry a pre-rendered layer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from t4_e2e_devkit.common.actor_state.state_representation import Point2D, StateSE2
from t4_e2e_devkit.common.maps.abstract_map import AbstractMap, MapObject
from t4_e2e_devkit.common.maps.abstract_map_objects import (
    Intersection,
    Lane,
    LaneConnector,
    LaneGraphEdgeMapObject,
    PolygonMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from t4_e2e_devkit.common.maps.maps_datatypes import (
    IntersectionType,
    LaneConnectorType,
    RasterLayer,
    RasterMap,
    SemanticMapLayer,
    StopLineType,
)
from t4_e2e_devkit.common.t4_map import T4Lanelet, T4MapAPI, T4MapObject


def _point(value: Point2D) -> Point:
    return Point(float(value.x), float(value.y))


class _T4Polyline(PolylineMapObject):
    def __init__(self, object_id: str, points: np.ndarray) -> None:
        super().__init__(str(object_id))
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(values) < 2:
            raise ValueError("a map polyline needs at least two points")
        self._points = values
        self._line = LineString(values)

    @property
    def linestring(self) -> LineString:
        return self._line

    @property
    def length(self) -> float:
        return float(self._line.length)

    @property
    def discrete_path(self) -> List[StateSE2]:
        result: list[StateSE2] = []
        for index, point in enumerate(self._points):
            if index == 0:
                delta = self._points[1] - self._points[0]
            elif index == len(self._points) - 1:
                delta = self._points[-1] - self._points[-2]
            else:
                delta = self._points[index + 1] - self._points[index - 1]
            result.append(StateSE2(float(point[0]), float(point[1]), math.atan2(delta[1], delta[0])))
        return result

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        return float(self._line.project(_point(point)))

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        arc = self.get_nearest_arc_length_from_position(point)
        projected = self._line.interpolate(arc)
        discrete = self.discrete_path
        if len(discrete) == 1:
            heading = discrete[0].heading
        else:
            nearest = min(
                range(len(discrete)),
                key=lambda index: (discrete[index].x - projected.x) ** 2
                + (discrete[index].y - projected.y) ** 2,
            )
            heading = discrete[nearest].heading
        return StateSE2(float(projected.x), float(projected.y), float(heading))

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        if self.length <= 1.0e-6:
            return 0.0
        epsilon = min(0.5, max(self.length * 0.01, 1.0e-3))
        left = self.get_nearest_pose_from_position(_point(self._line.interpolate(max(0.0, arc_length - epsilon))))
        center = self.get_nearest_pose_from_position(_point(self._line.interpolate(float(np.clip(arc_length, 0.0, self.length)))))
        right = self.get_nearest_pose_from_position(_point(self._line.interpolate(min(self.length, arc_length + epsilon))))
        turn = _wrap_angle(right.heading - left.heading)
        return float(turn / max(2.0 * epsilon, 1.0e-6)) if center else 0.0


class _T4Polygon(PolygonMapObject):
    def __init__(self, object_id: str, geometry: BaseGeometry) -> None:
        super().__init__(str(object_id))
        self._geometry = geometry

    @property
    def polygon(self) -> Polygon:
        if isinstance(self._geometry, Polygon):
            return self._geometry
        if hasattr(self._geometry, "geoms"):
            polygons = [item for item in self._geometry.geoms if isinstance(item, Polygon)]
            if polygons:
                return max(polygons, key=lambda item: item.area)
        # Point and line semantic objects still need a finite footprint for
        # ``AbstractMap`` containment and raster queries.  The source geometry
        # remains available through the T4 API; this is only the query footprint.
        return self._geometry.buffer(0.05)  # type: ignore[return-value]


class _T4Roadblock(RoadBlockGraphEdgeMapObject):
    def __init__(self, adapter: "T4MapAdapter", object_id: str, geometry: BaseGeometry) -> None:
        super().__init__(str(object_id))
        self.adapter = adapter
        self._geometry = geometry

    @property
    def polygon(self) -> Polygon:
        return _T4Polygon(self.id, self._geometry).polygon

    @property
    def incoming_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        return []

    @property
    def outgoing_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        return []

    @property
    def parallel_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        return [self]

    @property
    def interior_edges(self) -> List[LaneGraphEdgeMapObject]:
        return [
            self.adapter._lane_view(lane)
            for lane in self.adapter.api.lanes
            if self.id == _roadblock_id(lane)
        ]

    @property
    def children_stop_lines(self) -> List[StopLine]:
        return [
            self.adapter._stop_line_view(obj)
            for obj in self.adapter.api.get_stop_lines()
            if self.polygon.distance(obj.geometry) <= 2.0
        ]

    def intersection(self) -> Optional[Intersection]:
        return None


class _T4LaneMixin:
    adapter: "T4MapAdapter"
    source: T4Lanelet

    @property
    def polygon(self) -> Polygon:
        return self.source.polygon

    @property
    def incoming_edges(self) -> List[Lane]:
        return [self.adapter._lane_view(lane) for lane in self.adapter.api.get_predecessors(self.id)]

    @property
    def outgoing_edges(self) -> List[Lane]:
        return [self.adapter._lane_view(lane) for lane in self.adapter.api.get_successors(self.id)]

    @property
    def parallel_edges(self) -> List[Lane]:
        return [self] + [self.adapter._lane_view(lane) for lane in self.adapter.api.get_adjacent_lanes(self.id)]

    @property
    def baseline_path(self) -> _T4Polyline:
        return _T4Polyline(f"{self.id}:centerline", self.source.centerline)

    @property
    def left_boundary(self) -> _T4Polyline:
        return _T4Polyline(self.source.left_boundary_id, self.source.left_boundary)

    @property
    def right_boundary(self) -> _T4Polyline:
        return _T4Polyline(self.source.right_boundary_id, self.source.right_boundary)

    @property
    def speed_limit_mps(self) -> Optional[float]:
        return self.source.speed_limit_mps

    def get_roadblock_id(self) -> str:
        return _roadblock_id(self.source)

    def parent(self) -> RoadBlockGraphEdgeMapObject:
        roadblock_id = self.get_roadblock_id()
        if roadblock_id:
            object_value = self.adapter.api.get_map_object(roadblock_id, "roadblock")
            if isinstance(object_value, T4MapObject):
                return self.adapter._roadblock_view(object_value)
        return _T4Roadblock(self.adapter, "", self.source.polygon)

    def has_traffic_lights(self) -> bool:
        return bool(self.adapter.get_traffic_lights_for_lane(self.id))

    @property
    def stop_lines(self) -> List[StopLine]:
        return [self.adapter._stop_line_view(obj) for obj in self.adapter.api.get_stop_lines_for_lane(self.id)]

    def is_left_of(self, other: Lane) -> bool:
        return self._lateral_relation(other) > 0.0

    def is_right_of(self, other: Lane) -> bool:
        return self._lateral_relation(other) < 0.0

    @property
    def adjacent_edges(self) -> Tuple[Optional[Lane], Optional[Lane]]:
        candidates = self.adapter.api.get_adjacent_lanes(self.id)
        if not candidates:
            return None, None
        ordered = sorted(candidates, key=lambda lane: self._lateral_relation(self.adapter._lane_view(lane)))
        return self.adapter._lane_view(ordered[0]), self.adapter._lane_view(ordered[-1])

    def get_width_left_right(self, point: Point2D, include_outside: bool = False) -> Tuple[float, float]:
        del include_outside
        query = _point(point)
        center = self.baseline_path.linestring.project(query)
        projected = self.baseline_path.linestring.interpolate(center)
        return (
            float(projected.distance(self.left_boundary.linestring)),
            float(projected.distance(self.right_boundary.linestring)),
        )

    def oriented_distance(self, point: Point2D) -> float:
        query = _point(point)
        arc = self.baseline_path.linestring.project(query)
        projected = self.baseline_path.linestring.interpolate(arc)
        pose = self.baseline_path.get_nearest_pose_from_position(point)
        sign = np.sign((query.x - projected.x) * math.sin(pose.heading) - (query.y - projected.y) * math.cos(pose.heading))
        return float(sign * query.distance(self.baseline_path.linestring))

    def _lateral_relation(self, other: Lane) -> float:
        other_polygon = getattr(other, "polygon", None)
        if other_polygon is None:
            return 0.0
        heading = self.source.centerline[-1] - self.source.centerline[0]
        delta = np.asarray(other_polygon.centroid.coords[0]) - np.asarray(self.source.polygon.centroid.coords[0])
        return float(-heading[1] * delta[0] + heading[0] * delta[1])


class _T4Lane(_T4LaneMixin, Lane):
    def __init__(self, adapter: "T4MapAdapter", source: T4Lanelet) -> None:
        Lane.__init__(self, source.id)
        self.adapter = adapter
        self.source = source

    def index(self) -> int:
        return int(sorted(lane.id for lane in self.adapter.api.lanes).index(self.id))


class _T4LaneConnector(_T4LaneMixin, LaneConnector):
    def __init__(self, adapter: "T4MapAdapter", source: T4Lanelet) -> None:
        LaneConnector.__init__(self, source.id)
        self.adapter = adapter
        self.source = source

    @property
    def turn_type(self) -> LaneConnectorType:
        return {
            "straight": LaneConnectorType.STRAIGHT,
            "left": LaneConnectorType.LEFT,
            "right": LaneConnectorType.RIGHT,
            "uturn": LaneConnectorType.UTURN,
        }.get(self.source.turn_direction, LaneConnectorType.UNKNOWN)


class _T4StopLine(StopLine):
    def __init__(self, adapter: "T4MapAdapter", source: T4MapObject) -> None:
        super().__init__(source.id, StopLineType.UNKNOWN)
        self.adapter = adapter
        self.source = source

    @property
    def polygon(self) -> Polygon:
        geometry = self.source.geometry
        return geometry if isinstance(geometry, Polygon) else geometry.buffer(0.05)

    @property
    def intersection_from(self) -> Optional[Intersection]:
        return None

    @property
    def layer_type(self) -> StopLineType:
        return self.stop_line_type

    @property
    def parent(self) -> Optional[RoadBlockGraphEdgeMapObject]:
        return None


class _T4Intersection(Intersection):
    def __init__(self, adapter: "T4MapAdapter", source: T4MapObject) -> None:
        super().__init__(source.id, IntersectionType.DEFAULT)
        self.adapter = adapter
        self.source = source

    @property
    def polygon(self) -> Polygon:
        geometry = self.source.geometry
        return geometry if isinstance(geometry, Polygon) else geometry.buffer(0)

    @property
    def interior_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        return []

    @property
    def incoming_edges(self) -> List[Lane]:
        return []

    @property
    def is_signaled(self) -> bool:
        return bool(self.adapter.api.get_related_objects(self.id, "traffic_light"))


class T4MapAdapter(AbstractMap):
    """Implement the NuPlan ``AbstractMap`` query surface over a T4 map."""

    def __init__(
        self,
        api: T4MapAPI,
        *,
        raster_precision: float = 0.2,
        raster_padding_m: float = 2.0,
    ) -> None:
        if raster_precision <= 0.0 or raster_padding_m < 0.0:
            raise ValueError("raster_precision must be positive and padding non-negative")
        self.api = api
        self.raster_precision = float(raster_precision)
        self.raster_padding_m = float(raster_padding_m)
        self._lane_views: dict[str, Lane] = {}
        self._roadblock_views: dict[str, _T4Roadblock] = {}
        self._raster_layers: dict[SemanticMapLayer, RasterLayer] = {}

    @classmethod
    def from_scene(
        cls,
        scene_dir: str | Path,
        maps_root: Optional[str | Path] = None,
        *,
        strict: bool = False,
        route_lane_ids: Iterable[str] = (),
        raster_precision: float = 0.2,
        raster_padding_m: float = 2.0,
    ) -> Optional["T4MapAdapter"]:
        """Resolve a T4 scene's map metadata and construct the facade."""

        api = T4MapAPI.from_scene(
            scene_dir,
            maps_root,
            strict=strict,
            route_lane_ids=tuple(route_lane_ids),
        )
        return None if api is None else cls(
            api,
            raster_precision=raster_precision,
            raster_padding_m=raster_padding_m,
        )

    @property
    def map_name(self) -> str:
        return self.api.map_name

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        return [layer for layer in SemanticMapLayer if self._objects_for_layer(layer)]

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        return list(self.get_available_map_objects())

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        layer = SemanticMapLayer(layer)
        if layer not in self._raster_layers:
            self._raster_layers[layer] = self._rasterize(layer)
        return self._raster_layers[layer]

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        return RasterMap({SemanticMapLayer(layer): self.get_raster_map_layer(layer) for layer in layers})

    def get_vector_map(
        self, layers: Optional[Iterable[SemanticMapLayer]] = None
    ) -> Dict[SemanticMapLayer, tuple[MapObject, ...]]:
        """Return deterministic vector objects grouped by semantic layer."""

        selected = self.get_available_map_objects() if layers is None else [SemanticMapLayer(layer) for layer in layers]
        return {layer: tuple(self._objects_for_layer(layer)) for layer in selected}

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        query = _point(point)
        return [obj for obj in self._objects_for_layer(layer) if _geometry(obj).covers(query)]

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        objects = self.get_all_map_objects(point, layer)
        if len(objects) > 1:
            raise AssertionError(f"more than one {SemanticMapLayer(layer).name} contains {point}")
        return objects[0] if objects else None

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        if not self._objects_for_layer(layer):
            raise ValueError(f"map layer is unavailable: {SemanticMapLayer(layer).name}")
        return bool(self.get_all_map_objects(point, layer))

    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        query = _point(point)
        return {
            SemanticMapLayer(layer): [
                obj for obj in self._objects_for_layer(layer) if _geometry(obj).distance(query) <= radius
            ]
            for layer in layers
        }

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[MapObject]:
        layer = SemanticMapLayer(layer)
        if layer in {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}:
            lane = self.api.get_lane(object_id)
            return None if lane is None else self._lane_view(lane)
        for obj in self._objects_for_layer(layer):
            if getattr(obj, "id", None) == str(object_id):
                return obj
        return None

    def get_distance_to_nearest_map_object(
        self, point: Point2D, layer: SemanticMapLayer
    ) -> Tuple[Optional[str], Optional[float]]:
        query = _point(point)
        objects = self._objects_for_layer(layer)
        if not objects:
            return None, None
        nearest = min(objects, key=lambda obj: _geometry(obj).distance(query))
        return str(nearest.id), float(_geometry(nearest).distance(query))

    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        object_id, distance = self.get_distance_to_nearest_map_object(point, layer)
        del object_id
        return float("nan") if distance is None else distance

    def get_distances_matrix_to_nearest_map_object(
        self, points: List[Point2D], layer: SemanticMapLayer
    ) -> Optional[np.ndarray]:
        objects = self._objects_for_layer(layer)
        if not objects:
            return None
        geometries = [_geometry(obj) for obj in objects]
        return np.asarray(
            [min(geometry.distance(_point(point)) for geometry in geometries) for point in points],
            dtype=np.float64,
        )

    def initialize_all_layers(self) -> None:
        for layer in self.get_available_raster_layers():
            self.get_raster_map_layer(layer)

    def _objects_for_layer(self, layer: SemanticMapLayer) -> list[MapObject]:
        layer = SemanticMapLayer(layer)
        if layer == SemanticMapLayer.LANE:
            return [self._lane_view(lane) for lane in self.api.lanes if lane.lanelet_type != "lane_connector"]
        if layer == SemanticMapLayer.LANE_CONNECTOR:
            return [self._lane_view(lane) for lane in self.api.get_lane_connectors()]
        if layer == SemanticMapLayer.ROADBLOCK:
            return [self._roadblock_view(obj) for obj in self.api.get_roadblocks()]
        if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
            return [self._polygon_view(obj) for obj in self.api.get_roadblock_connectors()]
        if layer == SemanticMapLayer.INTERSECTION:
            return [self._intersection_view(obj) for obj in self.api.get_intersections()]
        mapping = {
            SemanticMapLayer.CROSSWALK: "crosswalk",
            SemanticMapLayer.STOP_LINE: "stop_line",
            SemanticMapLayer.TRAFFIC_LIGHT: "traffic_light",
            SemanticMapLayer.DRIVABLE_AREA: "drivable_area",
            SemanticMapLayer.SPEED_BUMP: "speed_bump",
            SemanticMapLayer.STOP_SIGN: "stop_sign",
            SemanticMapLayer.WALKWAYS: "walkway",
            SemanticMapLayer.CARPARK_AREA: "carpark_area",
            SemanticMapLayer.PUDO: "pudo",
            SemanticMapLayer.TURN_STOP: "turn_stop",
            SemanticMapLayer.YIELD: "yield",
            SemanticMapLayer.EXTENDED_PUDO: "pudo",
            SemanticMapLayer.BOUNDARIES: "line_string",
            SemanticMapLayer.BASELINE_PATHS: "line_string",
        }
        object_type = mapping.get(layer)
        if object_type is None:
            return []
        objects = self.api.get_objects(object_type)
        if layer == SemanticMapLayer.STOP_LINE:
            return [self._stop_line_view(obj) for obj in objects if isinstance(obj, T4MapObject)]
        return [self._polygon_view(obj) for obj in objects if isinstance(obj, T4MapObject)]

    def _lane_view(self, lane: T4Lanelet) -> Lane:
        if lane.id not in self._lane_views:
            view_type = _T4LaneConnector if lane.lanelet_type == "lane_connector" else _T4Lane
            self._lane_views[lane.id] = view_type(self, lane)
        return self._lane_views[lane.id]

    def _polygon_view(self, obj: T4MapObject) -> _T4Polygon:
        return _T4Polygon(obj.id, obj.geometry)

    def _roadblock_view(self, obj: T4MapObject) -> _T4Roadblock:
        if obj.id not in self._roadblock_views:
            self._roadblock_views[obj.id] = _T4Roadblock(self, obj.id, obj.geometry)
        return self._roadblock_views[obj.id]

    def _intersection_view(self, obj: T4MapObject) -> _T4Intersection:
        return _T4Intersection(self, obj)

    def _stop_line_view(self, obj: T4MapObject) -> _T4StopLine:
        return _T4StopLine(self, obj)

    def _rasterize(self, layer: SemanticMapLayer) -> RasterLayer:
        objects = self._objects_for_layer(layer)
        if not objects:
            raise ValueError(f"map layer is unavailable: {layer.name}")
        geometry = unary_union([_geometry(obj) for obj in objects])
        min_x, min_y, max_x, max_y = geometry.bounds
        min_x -= self.raster_padding_m
        min_y -= self.raster_padding_m
        max_x += self.raster_padding_m
        max_y += self.raster_padding_m
        width = max(1, int(math.ceil((max_x - min_x) / self.raster_precision)))
        height = max(1, int(math.ceil((max_y - min_y) / self.raster_precision)))
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for obj in objects:
            geometry = _geometry(obj)
            for polygon in _polygons(geometry):
                coordinates = [
                    (
                        int(round((x - min_x) / self.raster_precision)),
                        int(round((max_y - y) / self.raster_precision)),
                    )
                    for x, y in polygon.exterior.coords
                ]
                draw.polygon(coordinates, fill=255)
        transform = np.eye(4, dtype=np.float32)
        transform[0, 0] = 1.0 / self.raster_precision
        transform[1, 1] = -1.0 / self.raster_precision
        transform[0, 3] = -min_x / self.raster_precision
        transform[1, 3] = max_y / self.raster_precision
        return RasterLayer(np.asarray(image, dtype=np.uint8), self.raster_precision, transform)


def _roadblock_id(lane: T4Lanelet) -> str:
    for key in ("roadblock_id", "road_block_id", "parent_id"):
        value = lane.tags.get(key)
        if value:
            return str(value)
    return ""


def _geometry(obj: object) -> BaseGeometry:
    if isinstance(obj, T4Lanelet):
        return obj.polygon
    if isinstance(obj, T4MapObject):
        return obj.geometry
    if hasattr(obj, "polygon"):
        return obj.polygon
    raise TypeError(f"map object has no geometry: {type(obj).__name__}")


def _polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif hasattr(geometry, "geoms"):
        yield from (item for item in geometry.geoms if isinstance(item, Polygon))


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


__all__ = ["T4MapAdapter"]
