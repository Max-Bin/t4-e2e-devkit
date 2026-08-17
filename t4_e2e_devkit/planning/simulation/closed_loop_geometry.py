"""Geometry events for the T4 sensor-replay closed loop.

The module turns the annotations and local vector map from one replay frame
into scalar events around a simulated ego state.  It does not move the sensor
payload or claim to simulate other agents.  Missing annotations/map layers are
represented by ``None`` so an unavailable metric is never reported as a safe
zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from t4_e2e_devkit.common.dataclasses import MapTensors, T4Scene
from t4_e2e_devkit.common.enums import T4BoxIndex


@dataclass(frozen=True)
class ReplayGeometry:
    """One tick of geometry events in the simulated world frame."""

    agent_count: Optional[int] = None
    min_agent_clearance_m: Optional[float] = None
    ttc_s: Optional[float] = None
    ttc_violation: Optional[bool] = None
    drivable_violation: Optional[bool] = None
    road_border_violation: Optional[bool] = None
    road_border_distance_m: Optional[float] = None
    collision_tokens: Optional[tuple[str, ...]] = None

    @property
    def available(self) -> bool:
        """Whether at least one geometry source produced an event value."""

        return any(
            value is not None
            for value in (
                self.agent_count,
                self.min_agent_clearance_m,
                self.ttc_s,
                self.ttc_violation,
                self.drivable_violation,
                self.road_border_violation,
                self.road_border_distance_m,
            )
        )


def compute_replay_geometry(
    state: Any,
    scene: T4Scene,
    *,
    ttc_horizon_s: Optional[float] = 1.0,
    ttc_step_s: float = 0.1,
) -> ReplayGeometry:
    """Compute geometry events for ``state`` against one replay frame.

    Agent boxes and map tensors are stored in the recorded ego frame.  They are
    transformed to the scene world frame before comparison with the simulated
    ego.  TTC is a constant-velocity projection of the replayed boxes; it is a
    diagnostic event, not a reactive traffic policy.
    """

    if ttc_horizon_s is not None and ttc_horizon_s <= 0.0:
        raise ValueError("ttc_horizon_s must be positive or None")
    if ttc_step_s <= 0.0:
        raise ValueError("ttc_step_s must be positive")

    recorded_pose = _scene_pose(scene)
    ego_shape = scene.current_frame.ego_status.ego_shape
    ego_polygon = _ego_polygon(state, ego_shape)

    annotations = scene.current_frame.annotations
    agent_values: dict[str, object] = {
        "agent_count": None,
        "min_agent_clearance_m": None,
        "ttc_s": None,
        "ttc_violation": None,
        "collision_tokens": None,
    }
    if annotations is not None:
        boxes = np.asarray(annotations.boxes, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] < T4BoxIndex.HEADING + 1:
            boxes = np.zeros((0, T4BoxIndex.size()), dtype=np.float64)
        elif getattr(annotations, "velocities", None) is not None:
            velocities = np.asarray(annotations.velocities, dtype=np.float64)
            if boxes.shape[1] < T4BoxIndex.VELOCITY_Y + 1:
                padded = np.zeros((len(boxes), T4BoxIndex.size()), dtype=np.float64)
                padded[:, : boxes.shape[1]] = boxes
                boxes = padded
            boxes[:, T4BoxIndex.VELOCITY_X : T4BoxIndex.VELOCITY_Y + 1] = velocities
        agent_values["agent_count"] = int(len(boxes))
        world_boxes = _world_boxes(boxes, recorded_pose)
        clearances: list[float] = []
        collisions: list[str] = []
        for index, world_box in enumerate(world_boxes):
            width = float(world_box[T4BoxIndex.WIDTH])
            length = float(world_box[T4BoxIndex.LENGTH])
            if width <= 0.0 or length <= 0.0:
                continue
            agent_polygon = _box_polygon(world_box)
            if ego_polygon.intersects(agent_polygon):
                tokens = getattr(annotations, "track_tokens", None)
                collisions.append(str(tokens[index]) if tokens is not None else str(index))
            clearances.append(_signed_clearance(ego_polygon, agent_polygon))

        if clearances:
            agent_values["min_agent_clearance_m"] = float(min(clearances))
        agent_values["collision_tokens"] = tuple(collisions)

        if ttc_horizon_s is not None and len(world_boxes):
            ttc, available = _time_to_collision(
                state,
                ego_shape,
                world_boxes,
                horizon_s=ttc_horizon_s,
                step_s=ttc_step_s,
            )
            if available:
                agent_values["ttc_s"] = ttc
                agent_values["ttc_violation"] = ttc is not None

    map_values = _map_events(state, ego_shape, scene.current_frame.map_tensors, recorded_pose)
    return ReplayGeometry(**agent_values, **map_values)


def _scene_pose(scene: T4Scene) -> np.ndarray:
    values = scene.scene_metadata.global_center_pose
    if values is None:
        raise ValueError(
            f"scene {scene.scene_metadata.token} has no global_center_pose; "
            "closed-loop geometry needs a global frame"
        )
    pose = np.asarray(values, dtype=np.float64).reshape(-1)
    if pose.shape != (4,):
        raise ValueError(f"global_center_pose must have four values, got {pose.shape}")
    return np.array([pose[0], pose[1], math.atan2(pose[3], pose[2])], dtype=np.float64)


def _world_boxes(boxes: np.ndarray, recorded_pose: np.ndarray) -> np.ndarray:
    result = np.asarray(boxes, dtype=np.float64).copy()
    if not len(result):
        return result.reshape(0, max(9, result.shape[1] if result.ndim == 2 else 9))
    c, s = math.cos(float(recorded_pose[2])), math.sin(float(recorded_pose[2]))
    x = result[:, T4BoxIndex.X].copy()
    y = result[:, T4BoxIndex.Y].copy()
    result[:, T4BoxIndex.X] = recorded_pose[0] + c * x - s * y
    result[:, T4BoxIndex.Y] = recorded_pose[1] + s * x + c * y
    result[:, T4BoxIndex.HEADING] += recorded_pose[2]
    if result.shape[1] >= T4BoxIndex.VELOCITY_Y + 1:
        vx = result[:, T4BoxIndex.VELOCITY_X].copy()
        vy = result[:, T4BoxIndex.VELOCITY_Y].copy()
        result[:, T4BoxIndex.VELOCITY_X] = c * vx - s * vy
        result[:, T4BoxIndex.VELOCITY_Y] = s * vx + c * vy
    return result


def _ego_polygon(state: Any, ego_shape: Any):
    center_x = float(state.x) + ego_shape.rear_axle_to_center * math.cos(float(state.heading))
    center_y = float(state.y) + ego_shape.rear_axle_to_center * math.sin(float(state.heading))
    return _polygon(
        _box_corners(
            center_x,
            center_y,
            float(state.heading),
            float(ego_shape.length),
            float(ego_shape.width),
        )
    )


def _box_polygon(box: np.ndarray):
    return _polygon(
        _box_corners(
            float(box[T4BoxIndex.X]),
            float(box[T4BoxIndex.Y]),
            float(box[T4BoxIndex.HEADING]),
            float(box[T4BoxIndex.LENGTH]),
            float(box[T4BoxIndex.WIDTH]),
        )
    )


def _signed_clearance(first, second) -> float:
    """Return positive separation and negative OBB penetration depth."""

    if not first.intersects(second):
        return float(first.distance(second))
    first_points = np.asarray(first.exterior.coords[:-1], dtype=np.float64)
    second_points = np.asarray(second.exterior.coords[:-1], dtype=np.float64)
    axes = np.concatenate((_rectangle_axes(first_points), _rectangle_axes(second_points)))
    overlaps: list[float] = []
    for axis in axes:
        first_projection = first_points @ axis
        second_projection = second_points @ axis
        overlaps.append(
            min(float(first_projection.max()), float(second_projection.max()))
            - max(float(first_projection.min()), float(second_projection.min()))
        )
    return -float(min(overlaps))


def _rectangle_axes(points: np.ndarray) -> np.ndarray:
    edges = np.roll(points, -1, axis=0) - points
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    return axes / np.maximum(norms, 1.0e-12)


def _time_to_collision(
    state: Any,
    ego_shape: Any,
    world_boxes: np.ndarray,
    *,
    horizon_s: float,
    step_s: float,
) -> tuple[Optional[float], bool]:
    if world_boxes.shape[1] < T4BoxIndex.VELOCITY_Y + 1:
        return None, False
    velocities = world_boxes[:, T4BoxIndex.VELOCITY_X : T4BoxIndex.VELOCITY_Y + 1]
    valid_velocity = np.isfinite(velocities).all(axis=1)
    if not bool(valid_velocity.any()):
        return None, False

    times = np.arange(0.0, horizon_s + step_s * 0.5, step_s, dtype=np.float64)
    earliest: Optional[float] = None
    for time_s in times:
        ego_x = float(state.x) + float(state.speed_mps) * math.cos(float(state.heading)) * time_s
        ego_y = float(state.y) + float(state.speed_mps) * math.sin(float(state.heading)) * time_s
        ego_center_x = ego_x + ego_shape.rear_axle_to_center * math.cos(float(state.heading))
        ego_center_y = ego_y + ego_shape.rear_axle_to_center * math.sin(float(state.heading))
        ego_polygon = _polygon(
            _box_corners(
                ego_center_x,
                ego_center_y,
                float(state.heading),
                float(ego_shape.length),
                float(ego_shape.width),
            )
        )
        for index, box in enumerate(world_boxes):
            if not valid_velocity[index]:
                continue
            projected = box.copy()
            projected[T4BoxIndex.X] += velocities[index, 0] * time_s
            projected[T4BoxIndex.Y] += velocities[index, 1] * time_s
            if ego_polygon.intersects(_box_polygon(projected)):
                earliest = float(time_s)
                return earliest, True
    return earliest, True


def _map_events(
    state: Any,
    ego_shape: Any,
    map_tensors: Optional[MapTensors],
    recorded_pose: np.ndarray,
) -> dict[str, object]:
    if map_tensors is None:
        return {
            "drivable_violation": None,
            "road_border_violation": None,
            "road_border_distance_m": None,
        }

    ego_polygon = _ego_polygon(state, ego_shape)
    ego_points = np.asarray(ego_polygon.exterior.coords[:-1], dtype=np.float64)
    drivable = _map_lane_polygons(map_tensors.lanes, recorded_pose)
    drivable.extend(_map_ring_polygons(map_tensors.polygons, recorded_pose))
    drivable_violation: Optional[bool] = None
    if drivable:
        drivable_violation = not all(
            any(polygon.covers(_point(point)) for polygon in drivable) for point in ego_points
        )

    borders = _map_border_lines(map_tensors.line_strings, recorded_pose)
    road_border_violation: Optional[bool] = None
    road_border_distance: Optional[float] = None
    if borders:
        distances = [float(ego_polygon.distance(border)) for border in borders]
        road_border_distance = min(distances)
        road_border_violation = any(ego_polygon.intersects(border) for border in borders)

    return {
        "drivable_violation": drivable_violation,
        "road_border_violation": road_border_violation,
        "road_border_distance_m": road_border_distance,
    }


def _map_lane_polygons(rows: np.ndarray, recorded_pose: np.ndarray) -> list:
    polygons = []
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < 8:
        return polygons
    for row in values:
        valid = np.linalg.norm(row[:, :2], axis=1) > 1.0e-3
        if valid.sum() < 2:
            continue
        center = row[valid, :2]
        left = center + row[valid, 4:6]
        right = center + row[valid, 6:8]
        ring = np.concatenate((left, right[::-1]), axis=0)
        polygon = _valid_polygon(_local_to_world(ring, recorded_pose))
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def _map_ring_polygons(rows: np.ndarray, recorded_pose: np.ndarray) -> list:
    polygons = []
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < 2:
        return polygons
    for row in values:
        valid = np.linalg.norm(row[:, :2], axis=1) > 1.0e-3
        if valid.sum() < 3:
            continue
        polygon = _valid_polygon(_local_to_world(row[valid, :2], recorded_pose))
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def _map_border_lines(rows: np.ndarray, recorded_pose: np.ndarray) -> list:
    lines = []
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < 4:
        return lines
    for row in values:
        valid = (row[:, 3] > 0.5) & (np.linalg.norm(row[:, :2], axis=1) > 1.0e-3)
        indices = np.flatnonzero(valid)
        if len(indices) < 2:
            continue
        breaks = np.flatnonzero(np.diff(indices) > 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [len(indices)]))
        for start, end in zip(starts, ends, strict=True):
            if end - start < 2:
                continue
            points = _local_to_world(row[indices[start:end], :2], recorded_pose)
            line = _valid_line(points)
            if line is not None:
                lines.append(line)
    return lines


def _local_to_world(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    c, s = math.cos(float(origin[2])), math.sin(float(origin[2]))
    return np.column_stack(
        (
            origin[0] + c * values[:, 0] - s * values[:, 1],
            origin[1] + s * values[:, 0] + c * values[:, 1],
        )
    )


def _box_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    half_length = length / 2.0
    half_width = width / 2.0
    c, s = math.cos(heading), math.sin(heading)
    local = np.array(
        [
            [half_length, half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
            [half_length, -half_width],
        ],
        dtype=np.float64,
    )
    return np.column_stack(
        (
            x + local[:, 0] * c - local[:, 1] * s,
            y + local[:, 0] * s + local[:, 1] * c,
        )
    )


def _polygon(points: np.ndarray):
    from shapely.geometry import Polygon

    polygon = Polygon(np.asarray(points, dtype=np.float64))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _valid_polygon(points: np.ndarray):
    polygon = _polygon(points)
    return polygon if not polygon.is_empty and polygon.area > 1.0e-6 else None


def _valid_line(points: np.ndarray):
    from shapely.geometry import LineString

    line = LineString(np.asarray(points, dtype=np.float64))
    return line if not line.is_empty and line.length > 1.0e-6 else None


def _point(point: np.ndarray):
    from shapely.geometry import Point

    return Point(float(point[0]), float(point[1]))


__all__ = ["ReplayGeometry", "compute_replay_geometry"]
