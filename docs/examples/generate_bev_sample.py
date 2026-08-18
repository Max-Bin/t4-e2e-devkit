"""Generate the rich, deterministic BEV example used by the documentation.

Run from the repository root with::

    MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python \
        docs/examples/generate_bev_sample.py

The scene is synthetic on purpose.  It exercises the same ``T4Scene`` and
``plot_bev_frame`` interfaces as a real window while keeping the documentation
build independent of a locally mounted T4 dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from t4_e2e_devkit.common.constants import (  # noqa: E402
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    SEGMENT_POINT_DIM,
    T4_INTERVAL_LENGTH,
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_WHITE,
    TRAFFIC_LIGHT_YELLOW,
)
from t4_e2e_devkit.common.dataclasses import (  # noqa: E402
    Annotations,
    EgoShape,
    EgoStatus,
    MapTensors,
    SceneMetadata,
    T4Frame,
    T4Scene,
    Trajectory,
)
from t4_e2e_devkit.common.enums import T4BoxIndex, T4TrackLabel, TurnIndicator  # noqa: E402
from t4_e2e_devkit.visualization import plot_bev_frame, save_figure  # noqa: E402

_TRAFFIC_LIGHT_COLUMNS = {
    "green": TRAFFIC_LIGHT_GREEN,
    "yellow": TRAFFIC_LIGHT_YELLOW,
    "red": TRAFFIC_LIGHT_RED,
    "white": TRAFFIC_LIGHT_WHITE,
    "none": TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
}


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a 2-D polyline by arc length."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError(f"expected a non-empty [N, 2] polyline, got {values.shape}")
    if len(values) == 1:
        return np.repeat(values, count, axis=0)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
    if distances[-1] <= 1e-9:
        return np.repeat(values[:1], count, axis=0)
    query = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [np.interp(query, distances, values[:, dimension]) for dimension in range(2)]
    )


def _segment(points: np.ndarray, traffic_light: str, width: float = 3.2) -> np.ndarray:
    """Build one T4 lane segment, including reconstructed boundary offsets."""

    geometry = _resample(points, POINTS_PER_LANELET)
    tangent = np.gradient(geometry, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) * (width / 2.0)

    segment = np.zeros((POINTS_PER_LANELET, SEGMENT_POINT_DIM), dtype=np.float32)
    segment[:, 0:2] = geometry
    segment[:, 2:4] = tangent
    segment[:, 4:6] = normal
    segment[:, 6:8] = -normal
    segment[:, _TRAFFIC_LIGHT_COLUMNS[traffic_light]] = 1.0
    return segment


def _rectangle(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Return a closed rectangle in counter-clockwise order."""

    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])


def _put_ring(target: np.ndarray, index: int, points: np.ndarray) -> None:
    ring = _resample(points, POINTS_PER_POLYGON)
    target[index, :, :2] = ring
    target[index, :, 2] = 1.0


def _put_line(target: np.ndarray, index: int, points: np.ndarray, road_border: bool) -> None:
    line = _resample(points, POINTS_PER_LINE_STRING)
    target[index, :, :2] = line
    target[index, :, 3] = 1.0 if road_border else 0.0


def _make_map() -> MapTensors:
    """Create a small urban intersection with all renderer map layers populated."""

    lanes = np.zeros(
        (NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM), dtype=np.float32
    )
    route_lanes = np.zeros(
        (NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM), dtype=np.float32
    )

    x_main = np.linspace(-55.0, 55.0, 80)
    lane_specs = [
        (-6.4, "none"),
        (-3.2, "yellow"),
        (0.0, "green"),
        (3.2, "red"),
        (6.4, "white"),
    ]
    lane_paths = [
        (np.column_stack((x_main, np.full_like(x_main, y))), state)
        for y, state in lane_specs
    ]

    y_cross = np.linspace(-52.0, 52.0, 80)
    lane_paths.extend(
        [
            (np.column_stack((np.full_like(y_cross, x), y_cross)), state)
            for x, state in ((20.4, "red"), (23.6, "green"), (27.0, "yellow"))
        ]
    )
    for index, (points, state) in enumerate(lane_paths):
        lanes[index] = _segment(points, state)

    route_paths = [
        np.column_stack((np.linspace(-53.0, 18.0, 80), np.zeros(80))),
        np.array([[18.0, 0.0], [24.0, 1.0], [29.0, 6.0], [33.0, 14.0], [33.0, 49.0]]),
        np.column_stack((np.linspace(18.0, 53.0, 80), np.zeros(80))),
    ]
    for index, points in enumerate(route_paths):
        route_lanes[index] = _segment(points, "none")

    polygons = np.zeros((NUM_POLYGONS, POINTS_PER_POLYGON, 3), dtype=np.float32)
    polygon_shapes = [
        _rectangle(18.0, 28.5, -10.0, 10.0),  # intersection box
        _rectangle(-45.0, -34.0, -11.5, -9.5),
        _rectangle(-2.0, 6.0, 9.5, 12.0),
        _rectangle(35.0, 42.0, -12.0, -9.5),
        _rectangle(36.0, 41.0, 12.0, 18.0),
        _rectangle(14.5, 17.0, 14.0, 22.0),
    ]
    for index, shape in enumerate(polygon_shapes):
        _put_ring(polygons, index, shape)

    line_strings = np.zeros(
        (NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 4), dtype=np.float32
    )
    line_specs: list[tuple[np.ndarray, bool]] = [
        (np.column_stack((x_main, np.full_like(x_main, -10.4))), True),
        (np.column_stack((x_main, np.full_like(x_main, 10.4))), True),
        (np.column_stack((np.full_like(y_cross, 18.2), y_cross)), True),
        (np.column_stack((np.full_like(y_cross, 28.8), y_cross)), True),
    ]
    # Broken orange strings make lane markings and a crosswalk visible without
    # adding another renderer-specific primitive.
    for y in (-4.8, 4.8):
        for start in np.arange(-52.0, 15.0, 9.0):
            line_specs.append((np.array([[start, y], [start + 5.0, y]]), False))
    for x in np.arange(19.0, 28.0, 1.5):
        line_specs.append((np.array([[x, -9.0], [x, -4.0]]), False))
    for index, (points, road_border) in enumerate(line_specs[:NUM_LINE_STRINGS]):
        _put_line(line_strings, index, points, road_border)

    return MapTensors(
        lanes=lanes,
        lanes_speed_limit=np.full((NUM_SEGMENTS_IN_LANE, 1), 13.9, dtype=np.float32),
        lanes_has_speed_limit=np.ones((NUM_SEGMENTS_IN_LANE, 1), dtype=bool),
        route_lanes=route_lanes,
        route_lanes_speed_limit=np.full((NUM_SEGMENTS_IN_ROUTE, 1), 13.9, dtype=np.float32),
        route_lanes_has_speed_limit=np.ones((NUM_SEGMENTS_IN_ROUTE, 1), dtype=bool),
        polygons=polygons,
        line_strings=line_strings,
    )


def _make_annotations() -> Annotations:
    """Place every T4 tracked class, with varied headings and velocities."""

    rows = [
        (17.0, -3.2, 1.8, 4.5, 1.6, 0.02, 4.4, 0.0),  # car
        (-18.0, 3.2, 2.0, 5.8, 2.3, np.pi, -3.2, 0.0),  # truck
        (34.0, 6.4, 2.3, 7.2, 3.0, np.pi, -2.0, 0.0),  # bus
        (7.0, -6.5, 0.8, 1.8, 1.5, 0.35, 1.3, 0.2),  # bicycle
        (18.0, 7.8, 0.7, 0.7, 1.7, -np.pi / 2.0, 0.0, -0.7),  # pedestrian
        (-8.0, -3.2, 1.8, 4.3, 1.6, 0.0, 2.0, 0.0),  # second car
        (25.0, 2.0, 0.7, 0.7, 1.7, np.pi / 2.0, 0.0, -0.9),  # crossing pedestrian
        (-30.0, 6.4, 0.9, 1.9, 1.5, 0.1, 1.0, 0.0),  # second bicycle
        (42.0, -3.2, 1.9, 4.6, 1.7, np.pi, -4.0, 0.0),  # distant car
    ]
    labels = [
        T4TrackLabel.CAR,
        T4TrackLabel.TRUCK,
        T4TrackLabel.BUS,
        T4TrackLabel.BICYCLE,
        T4TrackLabel.PEDESTRIAN,
        T4TrackLabel.CAR,
        T4TrackLabel.PEDESTRIAN,
        T4TrackLabel.BICYCLE,
        T4TrackLabel.CAR,
    ]
    boxes = np.zeros((len(rows), 9), dtype=np.float32)
    for index, (x, y, width, length, height, heading, velocity_x, velocity_y) in enumerate(rows):
        boxes[index] = [
            x,
            y,
            0.0,
            width,
            length,
            height,
            heading,
            velocity_x,
            velocity_y,
        ]
    velocities = boxes[:, T4BoxIndex.VELOCITY_2D].copy()
    return Annotations(
        boxes=boxes,
        labels=np.asarray([int(label) for label in labels], dtype=np.int64),
        track_tokens=[f"sample-track-{index:02d}" for index in range(len(rows))],
        velocities=velocities,
    )


def _make_future_annotations(current: Annotations, count: int = 80) -> list[Annotations]:
    """Move every neighbour with its current velocity for the privileged trace."""

    future = [current]
    velocities = np.asarray(current.velocities, dtype=np.float32)
    for step in range(1, count + 1):
        boxes = current.boxes.copy()
        boxes[:, T4BoxIndex.X] += velocities[:, 0] * (step * T4_INTERVAL_LENGTH)
        boxes[:, T4BoxIndex.Y] += velocities[:, 1] * (step * T4_INTERVAL_LENGTH)
        future.append(
            Annotations(
                boxes=boxes,
                labels=current.labels.copy(),
                track_tokens=current.track_tokens,
                velocities=velocities.copy(),
            )
        )
    return future


def _make_scene() -> T4Scene:
    shape = EgoShape(wheel_base=2.8, length=4.8, width=1.9)
    map_tensors = _make_map()
    annotations = _make_annotations()
    future_annotations = _make_future_annotations(annotations)
    velocities = np.asarray(annotations.velocities, dtype=np.float32)

    frames: list[T4Frame] = []
    for index in range(31):
        relative_time = (index - 30) * T4_INTERVAL_LENGTH
        x = 6.0 * relative_time
        y = 0.8 * np.sin(0.75 * relative_time)
        heading = np.arctan2(0.8 * 0.75 * np.cos(0.75 * relative_time), 6.0)
        history_boxes = annotations.boxes.copy()
        history_boxes[:, T4BoxIndex.X] += velocities[:, 0] * relative_time
        history_boxes[:, T4BoxIndex.Y] += velocities[:, 1] * relative_time
        history_annotations = Annotations(
            boxes=history_boxes,
            labels=annotations.labels.copy(),
            track_tokens=annotations.track_tokens,
            velocities=velocities.copy(),
        )
        frames.append(
            T4Frame(
                frame_index=index,
                timestamp_us=index * 100_000,
                ego_status=EgoStatus(
                    ego_pose=np.array([x, y, heading], dtype=np.float32),
                    ego_velocity=np.array([6.0, 0.25], dtype=np.float32),
                    ego_acceleration=np.array([0.35, 0.08], dtype=np.float32),
                    ego_shape=shape,
                    turn_indicator=int(TurnIndicator.ENABLE_LEFT),
                    control_state={"steering": 0.08, "yaw_rate": 0.03},
                ),
                map_tensors=map_tensors if index == 30 else None,
                annotations=history_annotations,
            )
        )

    future_time = np.arange(1, 81, dtype=np.float64) * T4_INTERVAL_LENGTH
    gt_x = 6.0 * future_time
    gt_y = 1.2 * np.sin(0.45 * future_time) + 0.055 * future_time**2
    gt_dy = 1.2 * 0.45 * np.cos(0.45 * future_time) + 0.11 * future_time
    future_ego_poses = np.column_stack((gt_x, gt_y, np.arctan2(gt_dy, 6.0))).astype(np.float32)

    metadata = SceneMetadata(
        scene_dir="docs/examples/synthetic_bev",
        scene_id="rich_bev_sample",
        center_frame=30,
        num_history_frames=31,
        num_future_frames=80,
        vehicle="synthetic-demo-vehicle",
        date="2026-01-01",
    )
    return T4Scene(
        scene_metadata=metadata,
        frames=frames,
        future_ego_poses=future_ego_poses,
        future_annotations=future_annotations,
        goal_pose=np.array([49.0, 7.0, 0.96, 0.28], dtype=np.float32),
    )


def build_figure():
    """Build the documentation figure with every available single-frame layer."""

    scene = _make_scene()
    prediction_time = np.arange(1, 9, dtype=np.float64) * 0.5
    prediction_x = 6.15 * prediction_time
    prediction_y = 0.15 * prediction_time**2 + 0.25 * np.sin(prediction_time)
    prediction_dy = 0.30 * prediction_time + 0.25 * np.cos(prediction_time)
    prediction = Trajectory(
        poses=np.column_stack(
            (prediction_x, prediction_y, np.arctan2(prediction_dy, 6.15))
        ).astype(np.float32)
    )
    trajectories = {
        "history": scene.get_history_poses(),
        "ground_truth": scene.get_future_trajectory(),
        "prediction": prediction,
    }
    return plot_bev_frame(
        scene,
        trajectories=trajectories,
        config={
            "figure_size": (10.0, 8.0),
            "dpi": 100,
                "view_range": 50.0,
            "layers": [
                "polygons",
                "line_strings",
                "lanes",
                "route_lanes",
                "annotations",
            ],
            "legend": True,
            "legend_loc": "lower left",
            "show_history": True,
            "show_neighbor_future": True,
            "show_ego_future_footprints": True,
            "status_text": True,
        },
        title="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs/assets/bev_sample.png",
        help="PNG destination (default: docs/assets/bev_sample.png)",
    )
    args = parser.parse_args()
    figure, _ = build_figure()
    save_figure(figure, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
