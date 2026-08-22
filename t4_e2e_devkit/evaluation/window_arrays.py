"""Everything a metric needs from one window, as numpy.

Device-independent by construction, which is why it lives outside
``evaluation.gpu``: the CPU scorer reads the track association and the
lane-change exemption straight from here, and the extraction is
proposal-independent, so a dataloader worker can run it while the GPU is busy
with the previous batch.

Duplicating any of this on the torch side is how the two backends drifted apart
on every metric that had to be re-derived from upstream -- one implementation,
two callers.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from t4_e2e_devkit.common.actor_state.tracked_objects_types import (
    AGENT_TYPES,
    TrackedObjectType,
)
from t4_e2e_devkit.common.constants import T4_INTERVAL_LENGTH
from t4_e2e_devkit.common.maps.maps_datatypes import SemanticMapLayer
from t4_e2e_devkit.dataset.tracks import T4_LABEL_TO_TRACKED_OBJECT_TYPE

_AGENT_INSTANCE_TYPES = {
    TrackedObjectType.VEHICLE,
    TrackedObjectType.BICYCLE,
    TrackedObjectType.PEDESTRIAN,
}
_ASSOCIATION_MAX_GAP_FRAMES = 3
_ASSOCIATION_BASE_GATE_M = 4.0
_TURN_INDICATOR_SIGNALLED = (2, 3)  # left, right; 0/1 straight, 4 keep


def pack_rings_numpy(rings: list) -> np.ndarray:
    """Pad variable-length exterior rings into one [L, E+1, 2] buffer.

    Padding repeats the closing vertex; a zero-length edge can never satisfy
    the crossing predicate, so padded edges contribute nothing.
    """

    arrays = [np.asarray(ring, dtype=np.float64) for ring in rings]
    max_vertices = max(a.shape[0] + (a[0] != a[-1]).any() for a in arrays) + 0
    padded = np.empty((len(arrays), int(max_vertices), 2), dtype=np.float64)
    for index, ring in enumerate(arrays):
        length = ring.shape[0]
        padded[index, :length] = ring
        if (ring[0] != ring[-1]).any():
            padded[index, length] = ring[0]
            length += 1
        padded[index, length:] = padded[index, length - 1]
    return padded


def _valid_segment_points(row: np.ndarray, *, min_points: int = 2) -> np.ndarray | None:
    values = np.asarray(row, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] < 8:
        return None
    points = values[np.abs(values[:, :8]).sum(axis=-1) > 0.0]
    return points if points.shape[0] >= min_points else None


def _safe_polygon(ring: np.ndarray, *, min_points: int = 3) -> Polygon | None:
    values = np.asarray(ring, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != 2:
        return None
    values = values[np.isfinite(values).all(axis=-1)]
    if values.shape[0] < min_points:
        return None
    keep = np.ones(values.shape[0], dtype=bool)
    if values.shape[0] > 1:
        keep[1:] = np.linalg.norm(np.diff(values, axis=0), axis=-1) > 1.0e-7
    values = values[keep]
    if values.shape[0] < min_points:
        return None
    polygon = Polygon(values)
    if polygon.is_empty or polygon.area <= 1.0e-8:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 1.0e-8:
        return None
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda item: item.area)
    return polygon


def _lane_polygon(row: np.ndarray) -> Polygon | None:
    points = _valid_segment_points(row)
    if points is None:
        return None
    center = points[:, :2]
    left = center + points[:, 4:6]
    right = center + points[:, 6:8]
    return _safe_polygon(np.concatenate((left, right[::-1]), axis=0))


def _match_route_lanes(
    route_rows: Iterable[np.ndarray], lane_rows: Iterable[np.ndarray]
) -> list[int | None]:
    lane_points = [_valid_segment_points(row) for row in lane_rows]
    result: list[int | None] = []
    used: set[int] = set()
    for route_row in route_rows:
        route = _valid_segment_points(route_row)
        if route is None:
            result.append(None)
            continue
        best_index: int | None = None
        best_score = float("inf")
        for index, lane in enumerate(lane_points):
            if lane is None or index in used:
                continue
            score = float(
                np.linalg.norm(route[0, :2] - lane[0, :2])
                + np.linalg.norm(route[-1, :2] - lane[-1, :2])
            )
            if score < best_score:
                best_index, best_score = index, score
        if best_index is not None and best_score <= 3.0:
            used.add(best_index)
            result.append(best_index)
        else:
            result.append(None)
    return result


def _route_centerline(route: np.ndarray) -> np.ndarray:
    states: list[np.ndarray] = []
    for row in np.asarray(route):
        points = _valid_segment_points(row)
        if points is None:
            continue
        for point in points[:, :2]:
            value = np.asarray(point, dtype=np.float64)
            if states and np.linalg.norm(value - states[-1]) <= 1.0e-6:
                continue
            states.append(value)
    if len(states) < 2:
        raise ValueError("T4 route has fewer than two usable centerline points")
    return np.ascontiguousarray(np.stack(states))


def _map_geometries(
    lanes: np.ndarray, route: np.ndarray, polygons: np.ndarray
) -> tuple[list[Polygon], list[SemanticMapLayer], list[int], list[int], list[int]]:
    lane_entries = [
        (index, polygon)
        for index, row in enumerate(np.asarray(lanes))
        if (polygon := _lane_polygon(row)) is not None
    ]
    if not lane_entries:
        raise ValueError("T4 frame contains no usable lane polygons")

    geometries: list[Polygon] = []
    map_types: list[SemanticMapLayer] = []
    lane_indices: list[int] = []
    route_indices: list[int] = []
    intersection_indices: list[int] = []
    source_to_lane_index: dict[int, int] = {}
    for source, polygon in lane_entries:
        geometries.append(polygon)
        map_types.append(SemanticMapLayer.ROADBLOCK)
        geometries.append(polygon)
        map_types.append(SemanticMapLayer.LANE)
        lane_index = len(geometries) - 1
        lane_indices.append(lane_index)
        source_to_lane_index[source] = lane_index

    # A distinct name: the entry above is a lane's own row index, this one is a
    # route row's match and may be missing.
    for route_index, matched in enumerate(_match_route_lanes(route, lanes)):
        if matched is not None and matched in source_to_lane_index:
            route_indices.append(source_to_lane_index[matched])
            continue
        if route_index >= len(route):
            continue
        polygon = _lane_polygon(route[route_index])
        if polygon is None:
            continue
        geometries.extend((polygon, polygon))
        map_types.extend((SemanticMapLayer.ROADBLOCK, SemanticMapLayer.LANE))
        lane_index = len(geometries) - 1
        lane_indices.append(lane_index)
        route_indices.append(lane_index)

    for row in np.asarray(polygons):
        values = np.asarray(row, dtype=np.float64)
        valid = np.abs(values).sum(axis=-1) > 0.0
        polygon = _safe_polygon(values[valid, :2])
        if polygon is None:
            continue
        geometries.append(polygon)
        map_types.append(SemanticMapLayer.INTERSECTION)
        intersection_indices.append(len(geometries) - 1)

    if not route_indices:
        raise ValueError("T4 route could not be matched to any lane polygon")
    return geometries, map_types, lane_indices, route_indices, intersection_indices


def _red_light_rings(route: np.ndarray) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    for row in np.asarray(route):
        points = _valid_segment_points(row)
        if points is None or not bool((points[:, 10] > 0.5).any()):
            continue
        polygon = _lane_polygon(row)
        if polygon is not None:
            rings.append(np.asarray(polygon.exterior.coords, dtype=np.float64))
    return rings


def lane_change_exempt(
    indicators: np.ndarray,
    interval_s: float,
    pre_s: float,
    post_s: float,
) -> np.ndarray:
    """Samples the analyzer's lane-change exemption covers, as a ``[T]`` mask.

    ``trajectory_metrics.cpp`` collects the windows where the turn indicator is
    active, widens each by ``lane_change_pre_grace_time`` before and
    ``lane_change_post_grace_time`` after, merges them, and exempts any lane
    keeping sample inside one -- deviating from the current lane's centerline is
    the point of a signalled manoeuvre.

    PARTIAL PORT: upstream also collects hazard-lights windows, and T4 carries
    no hazard channel (``derived/scalars.npz`` has ``turn`` and nothing else for
    this), so only the turn-indicator half exists here.

    Both backends read this one implementation. The mask is index-aligned with
    the scored poses, and being off by a sample silently exempts or convicts a
    frame, so the alignment is pinned by test rather than by inspection.
    """

    values = np.asarray(indicators).reshape(-1)
    active = np.isin(values, _TURN_INDICATOR_SIGNALLED)
    if not active.any():
        return active
    before = max(0, int(round(float(pre_s) / float(interval_s))))
    after = max(0, int(round(float(post_s) / float(interval_s))))
    exempt = active.copy()
    for offset in range(1, before + 1):
        exempt[:-offset] |= active[offset:]
    for offset in range(1, after + 1):
        exempt[offset:] |= active[:-offset]
    return exempt


def associate_boxes(
    boxes_per_frame: Sequence[np.ndarray], labels_per_frame: Sequence[np.ndarray]
) -> list[list[str]]:
    """Greedy nearest-neighbour tracking over a window's annotation frames.

    Track state lives in parallel arrays rather than a dict of per-box dicts:
    the frame loop is inherently sequential, so its cost was the Python it did
    per box per frame (a dict each) plus a rescan of every token ever seen.
    Tracks past the gap can never match again -- ``frame_index`` only grows --
    so they are dropped instead of refiltered every frame.

    Order matters and is preserved: candidate rows keep first-appearance order,
    which is what decides how ``linear_sum_assignment`` breaks ties.
    """

    if len(boxes_per_frame) != len(labels_per_frame):
        raise ValueError("T4 box/label frame counts differ")
    next_token = 0
    tokens: list[str] = []
    states = np.empty((0, 9), dtype=np.float64)
    seen = np.empty(0, dtype=np.float64)
    kinds = np.empty(0, dtype=np.int64)
    output: list[list[str]] = []
    for frame_index, (raw_boxes, raw_labels) in enumerate(
        zip(boxes_per_frame, labels_per_frame, strict=True)
    ):
        boxes = np.asarray(raw_boxes, dtype=np.float64).reshape(-1, 9)
        labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
        if boxes.shape[0] != labels.shape[0]:
            raise ValueError(f"T4 frame {frame_index} box/label count mismatch")
        live = np.flatnonzero(frame_index - seen <= _ASSOCIATION_MAX_GAP_FRAMES)
        if live.shape[0] != len(tokens):
            tokens = [tokens[index] for index in live]
            states, seen, kinds = states[live], seen[live], kinds[live]
        # ``-1`` marks a box that matched no track and so opens one.
        assigned = np.full(boxes.shape[0], -1, dtype=np.int64)
        if len(tokens) and boxes.shape[0]:
            elapsed = frame_index - seen
            predicted = states[:, :2] + states[:, 7:9] * (elapsed * T4_INTERVAL_LENGTH)[:, None]
            speed = np.hypot(states[:, 7], states[:, 8])
            gate = np.maximum(
                _ASSOCIATION_BASE_GATE_M,
                2.0 + speed * elapsed * T4_INTERVAL_LENGTH * 2.0,
            )
            distance = np.linalg.norm(boxes[None, :, :2] - predicted[:, None, :], axis=-1)
            eligible = (kinds[:, None] == labels[None, :]) & (distance <= gate[:, None])
            rows, columns = linear_sum_assignment(np.where(eligible, distance, 1.0e6))
            matched = distance[rows, columns] <= gate[rows]
            assigned[columns[matched]] = rows[matched]

        hit = assigned >= 0
        continuing = assigned[hit]
        states[continuing] = boxes[hit]
        seen[continuing] = frame_index
        kinds[continuing] = labels[hit]

        frame_tokens: list[str] = []
        opened = np.flatnonzero(~hit)
        for box_index in range(boxes.shape[0]):
            row = int(assigned[box_index])
            if row >= 0:
                frame_tokens.append(tokens[row])
            else:
                frame_tokens.append(f"t4-agent-{next_token:08d}")
                next_token += 1
        if opened.shape[0]:
            tokens = tokens + [frame_tokens[index] for index in opened]
            states = np.concatenate((states, boxes[opened]))
            seen = np.concatenate((seen, np.full(opened.shape[0], frame_index, dtype=np.float64)))
            kinds = np.concatenate((kinds, labels[opened]))
        output.append(frame_tokens)
    return output


def _t4_box_corners(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 9)
    x, y, heading = values[:, 0], values[:, 1], values[:, 6]
    half_length = np.maximum(values[:, 4], 1.0e-3) / 2.0
    half_width = np.maximum(values[:, 3], 1.0e-3) / 2.0
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    cos_l, sin_l = np.cos(heading + np.pi / 2.0), np.sin(heading + np.pi / 2.0)
    corners = np.empty((values.shape[0], 4, 2), dtype=np.float64)
    for index, (longitudinal, lateral) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        lon, lat = longitudinal * half_length, lateral * half_width
        corners[:, index, 0] = x + lat * cos_l + lon * cos_h
        corners[:, index, 1] = y + lat * sin_l + lon * sin_h
    return corners


def extract_track_arrays(
    boxes: list[np.ndarray],
    labels: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-token box corners over the observation horizon; loader-worker friendly.

    Uses the same association as the CPU observation; the per-token snapshot
    (velocity, class) is taken at first appearance, exactly like
    ``unique_objects``.
    """

    tokens_per_frame = associate_boxes(boxes, labels)

    # One pass to assign columns and record first-appearance snapshots.
    token_index: dict[str, int] = {}
    snapshots: list[tuple[np.ndarray, int]] = []
    frame_rows: list[np.ndarray] = []
    frame_labels_list: list[np.ndarray] = []
    for frame_boxes, frame_labels, tokens in zip(boxes, labels, tokens_per_frame, strict=True):
        rows = np.asarray(frame_boxes, dtype=np.float64).reshape(-1, 9)
        values = np.asarray(frame_labels, dtype=np.int64).reshape(-1)
        frame_rows.append(rows)
        frame_labels_list.append(values)
        for row_idx, token in enumerate(tokens):
            if token not in token_index:
                token_index[token] = len(snapshots)
                snapshots.append((rows[row_idx].copy(), int(values[row_idx])))

    n_frames = len(boxes)
    n_tracks = len(snapshots)
    corners = np.zeros((n_frames, n_tracks, 4, 2), dtype=np.float64)
    valid = np.zeros((n_frames, n_tracks), dtype=bool)
    for frame_idx, (rows, tokens) in enumerate(zip(frame_rows, tokens_per_frame, strict=True)):
        if not rows.shape[0]:
            continue
        columns = np.fromiter(
            (token_index[token] for token in tokens), dtype=np.int64, count=len(tokens)
        )
        corners[frame_idx, columns] = _t4_box_corners(rows)
        valid[frame_idx, columns] = True

    snapshot_rows = np.stack([row for row, _ in snapshots]) if snapshots else np.zeros((0, 9))
    snapshot_labels = [label for _, label in snapshots]
    velocity = np.hypot(snapshot_rows[:, 7], snapshot_rows[:, 8]) if n_tracks else np.zeros(0)
    object_types = [
        T4_LABEL_TO_TRACKED_OBJECT_TYPE.get(label, TrackedObjectType.GENERIC_OBJECT)
        for label in snapshot_labels
    ]
    is_agent_instance = np.array([t in _AGENT_INSTANCE_TYPES for t in object_types], dtype=bool)
    is_agent_type = np.array([t in AGENT_TYPES for t in object_types], dtype=bool)

    masked = np.where(valid[..., None, None], corners, np.nan)
    with np.errstate(invalid="ignore"):
        bboxes = (
            np.stack(
                [
                    np.nanmin(masked[..., 0], axis=(0, 2)),
                    np.nanmin(masked[..., 1], axis=(0, 2)),
                    np.nanmax(masked[..., 0], axis=(0, 2)),
                    np.nanmax(masked[..., 1], axis=(0, 2)),
                ],
                axis=-1,
            )
            if n_tracks
            else np.zeros((0, 4))
        )

    return {
        "track_corners": corners,
        "track_valid": valid,
        "track_is_agent_instance": is_agent_instance,
        "track_velocity": velocity,
        "track_is_agent_type": is_agent_type,
        "track_bboxes": np.ascontiguousarray(bboxes),
    }


def extract_window_scene_arrays(
    lanes: np.ndarray,
    route: np.ndarray,
    polygons: np.ndarray,
    boxes: list,
    labels: list,
) -> dict[str, np.ndarray]:
    """Everything the GPU metric stage needs from one window, as numpy.

    Proposal-independent by construction, so a dataloader worker can run it
    while the GPU trains on the previous batch.
    """

    geometries, map_types, lane_indices, route_indices, intersection_indices = _map_geometries(
        lanes, route, polygons
    )
    rings = [np.asarray(geometry.exterior.coords, dtype=np.float64) for geometry in geometries]
    padded = pack_rings_numpy(rings)
    starts = padded[:, :-1]
    bboxes = np.concatenate([starts.min(axis=1), starts.max(axis=1)], axis=-1)
    drivable_idcs = [
        index
        for index, layer in enumerate(map_types)
        if layer in {SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION}
    ]

    arrays = {
        "map_rings": padded,
        "map_bboxes": np.ascontiguousarray(bboxes),
        "map_drivable_idx": np.asarray(drivable_idcs, dtype=np.int64),
        "map_lane_idx": np.asarray(lane_indices, dtype=np.int64),
        "map_on_route_idx": np.asarray(route_indices, dtype=np.int64),
        "map_intersection_idx": np.asarray(intersection_indices, dtype=np.int64),
        "centerline": _route_centerline(route),
    }
    red_light_rings = _red_light_rings(route)
    if red_light_rings:
        arrays["red_light_rings"] = np.ascontiguousarray(
            pack_rings_numpy(red_light_rings), dtype=np.float64
        )
        arrays["red_light_ring_lengths"] = np.asarray(
            [ring.shape[0] for ring in red_light_rings], dtype=np.int64
        )
    else:
        arrays["red_light_rings"] = np.empty((0, 0, 2), dtype=np.float64)
        arrays["red_light_ring_lengths"] = np.empty((0,), dtype=np.int64)
    arrays.update(extract_track_arrays(boxes, labels))
    return arrays
