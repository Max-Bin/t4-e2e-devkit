"""Bird's-eye-view rendering from T4 tensors.

A T4 scene ships its vector map slice per frame, already cropped to the window
and expressed in the trajectory frame. The map layer is drawn directly from
:class:`~t4_e2e_devkit.common.dataclasses.MapTensors`.

Two facts about that tensor are load-bearing and were verified against real
scenes rather than assumed:

* **Boundary columns are offsets, not absolute coordinates.** ``LB_X/LB_Y`` and
  ``RB_X/RB_Y`` are displacements from the centerline point in the same row, so a
  boundary is ``center + offset``. Measured on a real window: both offsets have
  magnitude 1.497 m, they point to opposite sides in 100% of rows, and the
  implied lane width has a median of exactly 3.00 m. Reading them as absolute
  coordinates would collapse every lane to a 3 m blob at the origin.
* **Unused rows are all-zero padding.** A lane slot is present but empty when its
  first eight columns sum to zero, which is the same validity test the reference
  judge uses. Drawing padding produces spurious geometry at the origin.

Everything is in the ego frame of the window's current frame: ``x`` forward,
``y`` left, ego at the origin facing ``+x``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    EgoShape,
    Lidar,
    MapTensors,
    Trajectory,
)
from t4_e2e_devkit.common.enums import T4BoxIndex
from t4_e2e_devkit.visualization.config import (
    BEV_AGENT_COLORS,
    EGO_COLOR,
    MAP_CONFIG,
    TRACK_CONFIG,
    TRAFFIC_LIGHT_COLORS,
    TRAJECTORY_CONFIG,
)
from t4_e2e_devkit.visualization.lidar import prepare_lidar_pc

#: Traffic-light one-hot column -> state name, in the segment layout's order.
_TRAFFIC_LIGHT_STATES: Sequence[str] = ("green", "yellow", "red", "white", "none")

# Turn indicators stored by the T4 converter.
_TURN_INDICATOR_LABELS = {
    0: "Straight",
    1: "Straight",
    2: "Left",
    3: "Right",
    4: "Keep",
}


# --------------------------------------------------------------------------- #
# Polyline extraction
# --------------------------------------------------------------------------- #


def valid_segment_mask(segments: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    """Which points of a lane/route tensor carry geometry.

    Uses the reference judge's test -- the first eight (geometry) columns summing
    to a non-zero absolute value -- so a plot and a score agree about which rows
    exist.

    :param segments: ``[N, P, >=8]`` lane or route tensor.
    :return: ``[N, P]`` boolean mask.
    """
    values = np.asarray(segments, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < 8:
        raise ValueError(f"expected a [N, P, >=8] segment tensor; got shape {values.shape}")
    return np.abs(values[..., :8]).sum(axis=-1) > 0.0


def centerlines_from_segments(segments: npt.NDArray[np.floating]) -> List[npt.NDArray[np.float64]]:
    """
    Centerline polylines of a lane or route tensor.
    :param segments: ``[N, P, >=8]`` lane or route tensor.
    :return: one ``[V, 2]`` array per non-empty segment.
    """
    values = np.asarray(segments, dtype=np.float64)
    mask = valid_segment_mask(values)
    polylines = []
    for index in range(values.shape[0]):
        points = values[index][mask[index]]
        if points.shape[0] >= 2:
            polylines.append(points[:, [C.X, C.Y]])
    return polylines


def boundaries_from_segments(
    segments: npt.NDArray[np.floating],
) -> tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]]]:
    """Left and right boundary polylines, reconstructed from the offsets.

    :param segments: ``[N, P, >=8]`` lane or route tensor.
    :return: ``(left, right)`` lists of ``[V, 2]`` arrays.
    """
    values = np.asarray(segments, dtype=np.float64)
    mask = valid_segment_mask(values)
    left, right = [], []
    for index in range(values.shape[0]):
        points = values[index][mask[index]]
        if points.shape[0] < 2:
            continue
        center = points[:, [C.X, C.Y]]
        # boundary = center + offset; see the module docstring for why.
        left.append(center + points[:, [C.LB_X, C.LB_Y]])
        right.append(center + points[:, [C.RB_X, C.RB_Y]])
    return left, right


def traffic_light_states(segments: npt.NDArray[np.floating]) -> List[str]:
    """The dominant traffic-light state of each non-empty segment.

    T4's segment layout carries a five-way one-hot per point, which is why this
    exists at all because the segment layout carries a traffic-light one-hot.

    Only ``route_lanes`` states describe the ego's own signal. ``lanes`` carries
    cross-traffic lights, which are red whenever the ego has green, so colouring
    the full lane set by signal would paint an intersection red on a green phase.

    :param segments: ``[N, P, 33]`` lane or route tensor.
    :return: one state name per non-empty segment, aligned with
        :func:`centerlines_from_segments`.
    """
    values = np.asarray(segments, dtype=np.float64)
    if values.shape[-1] < C.TRAFFIC_LIGHT + C.TRAFFIC_LIGHT_ONE_HOT_DIM:
        return []
    mask = valid_segment_mask(values)
    states = []
    for index in range(values.shape[0]):
        points = values[index][mask[index]]
        if points.shape[0] < 2:
            continue
        one_hot = points[:, C.TRAFFIC_LIGHT : C.TRAFFIC_LIGHT + C.TRAFFIC_LIGHT_ONE_HOT_DIM].sum(
            axis=0
        )
        states.append(_TRAFFIC_LIGHT_STATES[int(np.argmax(one_hot))])
    return states


def _traffic_light_state(points: npt.NDArray[np.floating]) -> str:
    """Return the upstream renderer's state for one valid segment."""

    start = C.TRAFFIC_LIGHT
    stop = start + C.TRAFFIC_LIGHT_ONE_HOT_DIM
    if points.shape[-1] < stop or points.shape[0] == 0:
        return "white"
    # ``draw_lanes`` in the source reads the first point of each segment.
    one_hot = np.asarray(points[0, start:stop], dtype=np.float64)
    if not np.any(one_hot):
        return "white"
    return _TRAFFIC_LIGHT_STATES[int(np.argmax(one_hot))]


def _lane_boundaries_by_traffic_light(
    segments: npt.NDArray[np.floating],
) -> Dict[str, List[npt.NDArray[np.float64]]]:
    """Extract lane edges grouped by traffic-light state.

    The reference function draws only the two lane boundaries.  It colours both
    sides from the segment's first traffic-light one-hot, with a low alpha, so
    the route and GT remain visually dominant.
    """

    values = np.asarray(segments, dtype=np.float64)
    mask = valid_segment_mask(values)
    grouped: Dict[str, List[npt.NDArray[np.float64]]] = {}
    for index in range(values.shape[0]):
        points = values[index][mask[index]]
        if points.shape[0] < 2:
            continue
        centre = points[:, [C.X, C.Y]]
        lines = (
            centre + points[:, [C.LB_X, C.LB_Y]],
            centre + points[:, [C.RB_X, C.RB_Y]],
        )
        grouped.setdefault(_traffic_light_state(points), []).extend(lines)
    return grouped


def _line_string_groups(
    line_strings: npt.NDArray[np.floating],
) -> Dict[str, List[npt.NDArray[np.float64]]]:
    """Split line strings into the reference renderer's red/orange groups."""

    values = np.asarray(line_strings, dtype=np.float64)
    grouped: Dict[str, List[npt.NDArray[np.float64]]] = {
        "road_borders": [],
        "line_strings": [],
    }
    for line in values:
        valid = np.abs(line[:, :2]).sum(axis=-1) > 0.0
        if valid.sum() < 2:
            continue
        key = "road_borders" if line.shape[-1] > 3 and np.any(line[:, 3] > 0.5) else "line_strings"
        grouped[key].append(line[valid][:, :2])
    return grouped


def polygon_rings(polygons: npt.NDArray[np.floating]) -> List[npt.NDArray[np.float64]]:
    """
    Non-empty rings of the polygon tensor.
    :param polygons: ``[N, P, 3]`` polygon tensor.
    :return: one ``[V, 2]`` ring per non-empty polygon.
    """
    values = np.asarray(polygons, dtype=np.float64)
    rings = []
    for index in range(values.shape[0]):
        points = values[index]
        valid = np.abs(points[:, :2]).sum(axis=-1) > 0.0
        if valid.sum() >= 3:
            rings.append(points[valid][:, :2])
    return rings


def line_string_polylines(line_strings: npt.NDArray[np.floating]) -> List[npt.NDArray[np.float64]]:
    """
    Non-empty polylines of the line-string tensor (road borders and markings).
    :param line_strings: ``[N, P, 4]`` line-string tensor.
    :return: one ``[V, 2]`` polyline per non-empty entry.
    """
    values = np.asarray(line_strings, dtype=np.float64)
    polylines = []
    for index in range(values.shape[0]):
        points = values[index]
        valid = np.abs(points[:, :2]).sum(axis=-1) > 0.0
        if valid.sum() >= 2:
            polylines.append(points[valid][:, :2])
    return polylines


# --------------------------------------------------------------------------- #
# Axis primitives
# --------------------------------------------------------------------------- #


def add_reference_bounding_box_to_bev_ax(
    ax,
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    color: str,
    alpha: float,
    linewidth: float = 1.0,
):
    """Draw a wireframe oriented box.

    The first edge is red, making the heading visible even when the object is
    stationary. The box is not filled, so it does not obscure the map.
    """

    from matplotlib.lines import Line2D

    dx_coeff = (+1, +1, -1, -1)
    dy_coeff = (+1, -1, -1, +1)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    for edge in range(4):
        current_dx = dx_coeff[edge] * length / 2.0
        current_dy = dy_coeff[edge] * width / 2.0
        next_edge = (edge + 1) % 4
        next_dx = dx_coeff[next_edge] * length / 2.0
        next_dy = dy_coeff[next_edge] * width / 2.0
        current = (
            x + current_dx * cos_h - current_dy * sin_h,
            y + current_dx * sin_h + current_dy * cos_h,
        )
        following = (
            x + next_dx * cos_h - next_dy * sin_h,
            y + next_dx * sin_h + next_dy * cos_h,
        )
        ax.add_line(
            Line2D(
                [current[0], following[0]],
                [current[1], following[1]],
                color="red" if edge == 0 else color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=6,
            )
        )
    return ax


def add_polylines_to_bev_ax(ax, polylines: Sequence[npt.NDArray], config: Dict[str, Any]):
    """
    Draw a set of polylines as one collection.
    :param ax: target axes.
    :param polylines: ``[V, 2]`` arrays.
    :param config: entry from :data:`MAP_CONFIG` or an equivalent.
    :return: the axes.
    """
    from matplotlib.collections import LineCollection

    if not len(polylines):
        return ax
    ax.add_collection(
        LineCollection(
            [np.asarray(line)[:, :2] for line in polylines],
            colors=config["color"],
            alpha=config["alpha"],
            linewidths=config["line_width"],
            linestyles=config.get("line_style", "-"),
            zorder=config.get("zorder", 1),
            label=config.get("label"),
        )
    )
    return ax


def add_map_to_bev_ax(
    ax,
    map_tensors: MapTensors,
    layers: Optional[Sequence[str]] = None,
    color_route_by_traffic_light: bool = False,
    draw_lane_boundaries: bool = True,
    draw_lane_centerlines: bool = False,
):
    """Draw the vector map.

    :param ax: target axes.
    :param map_tensors: the current frame's map.
    :param layers: which layers to draw; all of them by default.
    :param color_route_by_traffic_light: colour each route lane by its own signal
        state. Only route lanes are coloured this way -- see
        :func:`traffic_light_states`.
    :param draw_lane_boundaries: also draw the reconstructed lane edges.
    :param draw_lane_centerlines: draw lane centrelines as an additional layer.
        Centerlines are hidden by default to keep boundaries readable.
    :return: the axes.
    """
    layers = layers or ("polygons", "line_strings", "lanes", "route_lanes")

    # Draw boundaries and route before static map geometry so the route remains
    # readable.
    if "lanes" in layers:
        if draw_lane_boundaries:
            for state, polylines in _lane_boundaries_by_traffic_light(map_tensors.lanes).items():
                add_polylines_to_bev_ax(
                    ax,
                    polylines,
                    {
                        **MAP_CONFIG["lane_boundaries"],
                        "color": TRAFFIC_LIGHT_COLORS[state],
                    },
                )
        if draw_lane_centerlines:
            add_polylines_to_bev_ax(
                ax, centerlines_from_segments(map_tensors.lanes), MAP_CONFIG["lanes"]
            )

    if "route_lanes" in layers:
        route = centerlines_from_segments(map_tensors.route_lanes)
        if color_route_by_traffic_light:
            states = traffic_light_states(map_tensors.route_lanes)
            base = MAP_CONFIG["route_lanes"]
            grouped: Dict[str, List[npt.NDArray]] = {}
            for index, polyline in enumerate(route):
                state = states[index] if index < len(states) else "none"
                grouped.setdefault(state, []).append(polyline)
            for state, lines in grouped.items():
                add_polylines_to_bev_ax(ax, lines, {**base, "color": TRAFFIC_LIGHT_COLORS[state]})
        else:
            add_polylines_to_bev_ax(ax, route, MAP_CONFIG["route_lanes"])

    if "polygons" in layers:
        from matplotlib.collections import PolyCollection

        rings = polygon_rings(map_tensors.polygons)
        if rings:
            config = MAP_CONFIG["polygons"]
            ax.add_collection(
                PolyCollection(
                    rings,
                    facecolors=config["color"],
                    edgecolors=config["color"],
                    alpha=config["alpha"],
                    linewidths=config["line_width"],
                    zorder=config["zorder"],
                )
            )

    if "line_strings" in layers:
        # The source renderer distinguishes road borders from the other line
        # strings. In particular, do not let every map line inherit one neutral
        # colour: the red/orange split is one of the useful visual cues in its
        # frame output.
        for kind, polylines in _line_string_groups(map_tensors.line_strings).items():
            add_polylines_to_bev_ax(ax, polylines, MAP_CONFIG[kind])
    return ax


def add_box_to_bev_ax(
    ax,
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    color: str,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Draw one oriented rectangle, centred on ``(x, y)``.
    :param ax: target axes.
    :param x: centre x, metres.
    :param y: centre y, metres.
    :param heading: heading in radians.
    :param length: longitudinal extent.
    :param width: lateral extent.
    :param color: fill colour.
    :param config: overrides for :data:`TRACK_CONFIG`.
    :return: the axes.
    """
    from matplotlib.patches import Polygon as MplPolygon

    config = {**TRACK_CONFIG, **(config or {})}
    half_length, half_width = length / 2.0, width / 2.0
    corners = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ]
    )
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ rotation.T + np.array([x, y])

    ax.add_patch(
        MplPolygon(
            corners,
            closed=True,
            facecolor=color,
            edgecolor=config["edge_color"],
            alpha=config["fill_alpha"],
            linewidth=config["line_width"],
            zorder=config["zorder"],
        )
    )
    if config.get("heading_tick"):
        # A stationary box has no trajectory to read its orientation from.
        tick = config["heading_tick_length"] * length
        ax.plot(
            [x, x + tick * cos_h],
            [y, y + tick * sin_h],
            color=config["edge_color"],
            linewidth=config["line_width"],
            zorder=config["zorder"] + 1,
        )
    return ax


def add_annotations_to_bev_ax(
    ax,
    annotations: Annotations,
    config: Optional[Dict[str, Any]] = None,
    label_classes: bool = False,
    history: Optional[Sequence[Annotations]] = None,
):
    """Draw tracked objects and their past trajectories when available.

    Box columns follow :class:`~t4_e2e_devkit.common.enums.T4BoxIndex` --
    ``width`` before ``length``. The named indices keep this order explicit.

    :param ax: target axes.
    :param annotations: the frame's annotations.
    :param config: overrides for :data:`TRACK_CONFIG`.
    :param label_classes: add one legend entry per class present.
    :param history: annotations ordered oldest-first through the current frame;
        the renderer draws these as dashed neighbour traces.
    :return: the axes.
    """
    boxes = np.asarray(annotations.boxes, dtype=np.float64)
    labels = np.asarray(annotations.labels, dtype=np.int64)
    velocities = (
        np.asarray(annotations.velocities, dtype=np.float64)
        if annotations.velocities is not None
        else boxes[:, T4BoxIndex.VELOCITY_2D]
    )
    seen: set[int] = set()

    if history is not None and len(history) > 1:
        from matplotlib.collections import LineCollection

        past_lines = []
        past_colors = []
        current_tokens = annotations.track_tokens
        for object_index, label in enumerate(labels):
            token = (
                current_tokens[object_index]
                if current_tokens is not None and object_index < len(current_tokens)
                else None
            )
            points = []
            for past in history:
                if past is None or len(past) == 0:
                    continue
                past_index = object_index
                if token is not None and past.track_tokens is not None:
                    matching = [
                        i for i, past_token in enumerate(past.track_tokens) if past_token == token
                    ]
                    if not matching:
                        continue
                    past_index = matching[0]
                if past_index >= len(past) or int(past.labels[past_index]) != int(label):
                    continue
                points.append(past.boxes[past_index, T4BoxIndex.POINT2D])
            if len(points) > 1:
                past_lines.append(np.asarray(points, dtype=np.float64))
                past_colors.append(BEV_AGENT_COLORS.get(int(label), "blue"))
        if past_lines:
            ax.add_collection(
                LineCollection(
                    past_lines,
                    colors=past_colors,
                    alpha=0.6,
                    linewidths=1.0,
                    linestyles="--",
                    zorder=2,
                )
            )
            ax.plot(
                [],
                [],
                color="#4B5563",
                alpha=0.6,
                linewidth=1.0,
                linestyle="--",
                label="other-agent history",
            )

    for box, label, velocity in zip(boxes, labels, velocities, strict=True):
        color = BEV_AGENT_COLORS.get(int(label), "blue")
        add_reference_bounding_box_to_bev_ax(
            ax,
            x=float(box[T4BoxIndex.X]),
            y=float(box[T4BoxIndex.Y]),
            heading=float(box[T4BoxIndex.HEADING]),
            length=float(box[T4BoxIndex.LENGTH]),
            width=float(box[T4BoxIndex.WIDTH]),
            color=color,
            alpha=float((config or {}).get("alpha", 0.5)),
            linewidth=float((config or {}).get("line_width", 1.0)),
        )
        speed = float(np.hypot(velocity[0], velocity[1]))
        if speed > 0.1:
            arrow_color = BEV_AGENT_COLORS.get(int(label), "blue")
            ax.arrow(
                float(box[T4BoxIndex.X]),
                float(box[T4BoxIndex.Y]),
                float(velocity[0]) / 2.0,
                float(velocity[1]) / 2.0,
                width=0.2,
                head_width=0.5,
                head_length=0.3,
                fc=arrow_color,
                ec=arrow_color,
                alpha=0.6,
                zorder=7,
            )
        if label_classes and int(label) not in seen:
            seen.add(int(label))
            ax.plot([], [], color=color, linestyle="-", label=_class_name(int(label)))
    return ax


def add_ego_to_bev_ax(
    ax,
    ego_shape: EgoShape,
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    color: str = EGO_COLOR,
    label: Optional[str] = "ego",
):
    """Draw the reference forward arrow and a clearly visible ego footprint.

    The source renderer uses a red forward arrow. T4 poses are rear-axle poses,
    so the footprint is shifted to the body centre using the same vehicle
    geometry that the scorer uses. The outline is retained in the single-frame
    devkit view so the ego remains visible beneath overlapping trajectories.

    :param ax: target axes.
    :param ego_shape: the scene's own footprint.
    :param x: rear-axle x, metres.
    :param y: rear-axle y, metres.
    :param heading: heading in radians.
    :param color: fill colour.
    :param label: legend label, or ``None``.
    :return: the axes.
    """
    dx = ego_shape.length / 2.0 * np.cos(heading)
    dy = ego_shape.length / 2.0 * np.sin(heading)
    ax.arrow(
        x,
        y,
        dx,
        dy,
        width=ego_shape.width / 2.0,
        head_width=ego_shape.width,
        head_length=ego_shape.length / 3.0,
        fc=color,
        ec=color,
        alpha=0.7,
        zorder=11,
    )

    offset = ego_shape.rear_axle_to_center
    add_box_to_bev_ax(
        ax,
        x=x + offset * np.cos(heading),
        y=y + offset * np.sin(heading),
        heading=heading,
        length=ego_shape.length,
        width=ego_shape.width,
        color=color,
        config={
            "fill_alpha": 0.55,
            "edge_color": "black",
            "line_width": 2.2,
            "zorder": 12,
            "heading_tick": False,
        },
    )
    if label:
        ax.plot([], [], color=color, linestyle="-", label=label)
    return ax


def _future_gradient_colors(count: int) -> List[List[float]]:
    """Blue-to-red colours used for future points in the source renderer."""

    if count <= 0:
        return []
    values = np.linspace(0.0, 1.0, count)
    return [[float(value), 0.0, float(1.0 - value)] for value in values]


def add_ego_future_to_bev_ax(
    ax,
    future_poses: npt.NDArray[np.floating],
    ego_shape: EgoShape,
    *,
    draw_footprints: bool = True,
):
    """Draw the ego GT future as gradient points and horizon wireframes.

    The scene may retain a longer future than the trajectory supplied to a
    scorer. Draw the recorded future independently so both horizons remain
    inspectable.
    """

    poses = np.asarray(future_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] < 3:
        return ax
    valid = np.abs(poses[:, :2]).sum(axis=-1) > 1e-6
    valid_poses = poses[valid]
    if len(valid_poses):
        ax.scatter(
            valid_poses[:, 0],
            valid_poses[:, 1],
            c=_future_gradient_colors(len(valid_poses)),
            alpha=0.5,
            s=20,
            zorder=5,
            label="ego GT future",
        )

    if draw_footprints:
        footprint_label_added = False
        for index in (40 - 1, 80 - 1):
            if index >= len(poses) or not valid[index]:
                continue
            add_reference_bounding_box_to_bev_ax(
                ax,
                x=float(poses[index, 0]),
                y=float(poses[index, 1]),
                heading=float(poses[index, 2]),
                length=ego_shape.length,
                width=ego_shape.width,
                color="orange",
                alpha=0.1,
            )
            if not footprint_label_added:
                ax.plot(
                    [],
                    [],
                    color="orange",
                    alpha=0.35,
                    linewidth=1.0,
                    label="ego GT horizon footprint",
                )
                footprint_label_added = True
    return ax


def _match_future_annotation_positions(
    current: Annotations,
    future: Sequence[Annotations],
) -> List[npt.NDArray[np.float64]]:
    """Associate future GT boxes to current objects for gradient point traces."""

    current_boxes = np.asarray(current.boxes, dtype=np.float64)
    current_labels = np.asarray(current.labels, dtype=np.int64)
    current_velocities = (
        np.asarray(current.velocities, dtype=np.float64)
        if current.velocities is not None
        else current_boxes[:, T4BoxIndex.VELOCITY_2D]
    )
    tracks: List[npt.NDArray[np.float64]] = []
    for box, label, velocity in zip(current_boxes, current_labels, current_velocities, strict=True):
        points: List[npt.NDArray[np.float64]] = []
        for step, future_annotations in enumerate(future, start=1):
            candidates = np.asarray(future_annotations.boxes, dtype=np.float64)
            labels = np.asarray(future_annotations.labels, dtype=np.int64)
            same_class = np.flatnonzero(labels == label)
            if not len(same_class):
                continue
            predicted = box[:2] + velocity * (step * C.T4_INTERVAL_LENGTH)
            distances = np.linalg.norm(candidates[same_class, :2] - predicted, axis=1)
            candidate_index = int(same_class[int(np.argmin(distances))])
            if float(np.min(distances)) > 8.0:
                continue
            points.append(candidates[candidate_index, :2].copy())
        if points:
            tracks.append(np.stack(points))
    return tracks


def add_future_annotations_to_bev_ax(
    ax,
    current: Optional[Annotations],
    future: Optional[Sequence[Annotations]],
):
    """Draw neighbour GT future points when the privileged scene supplies them."""

    if current is None or not future:
        return ax
    for index, points in enumerate(_match_future_annotation_positions(current, future)):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c="#9467BD",
            alpha=0.5,
            s=8,
            zorder=4,
            label="other-agent future" if index == 0 else None,
        )
    return ax


def add_trajectory_to_bev_ax(
    ax,
    trajectory: Trajectory | npt.NDArray[np.floating],
    kind: str = "prediction",
    config: Optional[Dict[str, Any]] = None,
    include_origin: bool = True,
):
    """Draw a trajectory.

    :param ax: target axes.
    :param trajectory: a :class:`Trajectory` or an ``[T, >=2]`` pose array.
    :param kind: key into :data:`TRAJECTORY_CONFIG`.
    :param config: overrides for that entry.
    :param include_origin: prepend the ego origin, so the path starts at the
        vehicle rather than floating at the first predicted pose.
    :return: the axes.
    """
    poses = np.asarray(
        trajectory.poses if isinstance(trajectory, Trajectory) else trajectory, dtype=np.float64
    )
    if poses.ndim != 2 or poses.shape[0] == 0:
        return ax
    if kind not in TRAJECTORY_CONFIG:
        raise ValueError(
            f"unknown trajectory kind {kind!r}; expected one of {sorted(TRAJECTORY_CONFIG)}"
        )
    settings = {**TRAJECTORY_CONFIG[kind], **(config or {})}

    # GT is a time-coloured scatter, not another solid line. This keeps it
    # distinct from route, history and prediction layers.
    if kind == "ground_truth":
        valid = np.abs(poses[:, :2]).sum(axis=-1) > 1e-6
        valid_poses = poses[valid]
        if len(valid_poses):
            ax.scatter(
                valid_poses[:, 0],
                valid_poses[:, 1],
                c=_future_gradient_colors(len(valid_poses)),
                alpha=settings["alpha"],
                s=settings["marker_size"],
                zorder=settings["zorder"],
                label=settings.get("label"),
            )
        return ax

    xy = poses[:, :2]
    if include_origin:
        xy = np.vstack([np.zeros((1, 2)), xy])

    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=settings["color"],
        alpha=settings["alpha"],
        linewidth=settings["line_width"],
        linestyle=settings["line_style"],
        zorder=settings["zorder"],
        label=settings.get("label"),
    )
    if settings.get("marker"):
        # Mark the model's own pose spacing, so a reader can see the time
        # structure rather than only the shape.
        ax.scatter(
            poses[:, 0],
            poses[:, 1],
            s=settings["marker_size"],
            c=settings["color"],
            marker=settings["marker"],
            edgecolors="white",
            linewidths=0.5,
            zorder=settings["zorder"] + 1,
        )
    return ax


def add_lidar_to_bev_ax(ax, lidar: Lidar, config: Optional[Dict[str, Any]] = None, seed: int = 0):
    """
    Scatter a point cloud.
    :param ax: target axes.
    :param lidar: the frame's sweep.
    :param config: overrides for :data:`LIDAR_CONFIG`.
    :param seed: subsample seed.
    :return: the axes.
    """
    from t4_e2e_devkit.visualization.config import LIDAR_CONFIG

    if lidar is None or lidar.lidar_pc is None:
        return ax
    settings = {**LIDAR_CONFIG, **(config or {})}
    points, colors, _ = prepare_lidar_pc(lidar.lidar_pc, settings, seed=seed)
    if points.shape[0]:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=settings["point_size"],
            c=colors,
            alpha=settings["alpha"],
            linewidths=0,
            zorder=settings["zorder"],
        )
    return ax


def add_goal_to_bev_ax(ax, goal_pose: npt.NDArray[np.floating]):
    """Draw the blue goal-pose arrow used by the reference renderer.

    :param ax: target axes.
    :param goal_pose: ``[4]`` of ``(x, y, cos, sin)``.
    :return: the axes.
    """
    if goal_pose is None:
        return ax
    goal = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    if goal.shape[0] < 4:
        return ax
    goal_x, goal_y, goal_cos, goal_sin = goal[:4]
    ax.arrow(
        goal_x,
        goal_y,
        2.0 * goal_cos,
        2.0 * goal_sin,
        width=0.5,
        head_width=1.0,
        head_length=1.0,
        fc="blue",
        ec="blue",
        alpha=0.7,
        zorder=7,
        label="Goal Pose",
    )
    return ax


def add_bev_status_text(
    ax,
    ego_status,
    view_range: float,
    previous_turn_indicator: Optional[int] = None,
):
    """Add the upper-right kinematic text block from ``setup_axis`` upstream."""

    velocity = np.asarray(ego_status.ego_velocity, dtype=np.float64).reshape(-1)
    acceleration = np.asarray(ego_status.ego_acceleration, dtype=np.float64).reshape(-1)
    control = ego_status.control_state or {}

    def _control_value(name: str) -> float:
        value = control.get(name, 0.0)
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        return float(values[0]) if len(values) else 0.0

    turn_value = ego_status.turn_indicator
    turn_text = (
        "There is no turn command"
        if turn_value is None
        else _TURN_INDICATOR_LABELS.get(int(turn_value), str(int(turn_value)))
    )
    if (
        turn_value is not None
        and previous_turn_indicator is not None
        and int(turn_value) == int(previous_turn_indicator)
    ):
        turn_label = _TURN_INDICATOR_LABELS.get(int(turn_value), str(int(turn_value)))
        turn_text = turn_label if turn_label == "Keep" else f"Keep {turn_label}"
    ax.text(
        view_range - 1.0,
        view_range - 1.0,
        f"VelocityX: {velocity[0]:.2f} m/s\n"
        f"VelocityY: {velocity[1]:.2f} m/s\n"
        f"AccelerationX: {acceleration[0]:.2f} m/s²\n"
        f"AccelerationY: {acceleration[1]:.2f} m/s²\n"
        f"Steering: {_control_value('steering'):.2f} rad\n"
        f"Yaw Rate: {_control_value('yaw_rate'):.2f} rad/s\n"
        f"Turn Command GT: {turn_text}\n"
        "Turn Command PR: There is no predicted turn command",
        fontsize=8,
        color="red",
        ha="right",
        va="top",
        zorder=20,
    )
    return ax


def _class_name(label: int) -> str:
    from t4_e2e_devkit.common.enums import T4TrackLabel

    try:
        return T4TrackLabel(label).name.lower()
    except ValueError:
        return f"class {label}"
