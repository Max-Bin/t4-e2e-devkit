"""Plotting entry points.

The module also provides score overlays and multi-agent comparison plots:
* :func:`plot_bev_with_score` puts a trajectory next to the PDM components
  that scored it, with the multiplicative gates marked.
* :func:`plot_agent_comparison` renders several plans on one window.

Everything renders through Agg and returns a figure, so these work headless and
compose into GIFs, image logs or a report without a display.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.constants import (
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    SEGMENT_POINT_DIM,
)
from t4_e2e_devkit.common.dataclasses import EgoShape, MapTensors, T4Scene, Trajectory
from t4_e2e_devkit.common.enums import T4TrackLabel
from t4_e2e_devkit.evaluation.navsim_score import NAVSIM_METRICS
from t4_e2e_devkit.visualization.bev import (
    add_annotations_to_bev_ax,
    add_bev_status_text,
    add_ego_future_to_bev_ax,
    add_ego_to_bev_ax,
    add_future_annotations_to_bev_ax,
    add_goal_to_bev_ax,
    add_lidar_to_bev_ax,
    add_map_to_bev_ax,
    add_trajectory_to_bev_ax,
)
from t4_e2e_devkit.visualization.camera import (
    add_annotations_to_camera_ax,
    add_camera_ax,
    add_lidar_to_camera_ax,
    add_trajectory_to_camera_ax,
    camera_grid_layout,
)
from t4_e2e_devkit.visualization.config import (
    BEV_AGENT_COLORS,
    BEV_PLOT_CONFIG,
    CAMERAS_PLOT_CONFIG,
    EGO_COLOR,
    GOAL_COLOR,
    SCORE_PANEL_CONFIG,
    TRAJECTORY_CONFIG,
)


def _pyplot():
    """Import pyplot with a headless backend, without disturbing an existing one."""
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def configure_bev_ax(ax, view_range: float, config: Optional[Dict[str, Any]] = None):
    """
    Apply the shared BEV axis styling.
    :param ax: target axes.
    :param view_range: half-extent in metres.
    :param config: overrides for :data:`BEV_PLOT_CONFIG`.
    :return: the axes.
    """
    settings = {**BEV_PLOT_CONFIG, **(config or {})}
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    if settings["grid"]:
        ax.grid(True, alpha=settings["grid_alpha"])
    if settings["axis_labels"]:
        ax.set_xlabel("X [m]", fontsize=9)
        ax.set_ylabel("Y [m]", fontsize=9)
    return ax


def add_fixed_bev_legend(
    ax,
    *,
    trajectory_roles: Sequence[str] = (
        "history",
        "ground_truth",
        "prediction",
    ),
    include_ego: bool = True,
    include_agents: bool = True,
    include_goal: bool = True,
    loc: str = "lower right",
    fontsize: float = 8,
    framealpha: float = 0.85,
    ncol: int = 2,
):
    """Attach a deterministic BEV legend with a fixed semantic vocabulary.

    A frame-by-frame legend built from ``ax.get_legend_handles_labels()``
    changes size whenever an agent class disappears from view.  That is
    especially distracting in a video.  This helper creates proxy artists for
    the requested roles and all five T4 agent classes, so callers can keep the
    legend content and geometry constant across frames.

    ``trajectory_roles`` is deliberately an explicit argument rather than
    inferred from the artists on ``ax``.  A video renderer can pass one stable
    tuple for every frame, while a one-off plot can request only the roles it
    actually displays.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    labels = []
    for role in trajectory_roles:
        if role not in TRAJECTORY_CONFIG:
            raise ValueError(
                f"unknown trajectory role {role!r}; expected one of "
                f"{sorted(TRAJECTORY_CONFIG)}"
            )
        settings = TRAJECTORY_CONFIG[role]
        marker = settings.get("marker") or None
        handles.append(
            Line2D(
                [0],
                [0],
                color=settings["color"],
                alpha=settings["alpha"],
                linewidth=settings["line_width"],
                linestyle=settings["line_style"],
                marker=marker,
                markersize=5 if marker else 0,
            )
        )
        labels.append(str(settings["label"]))

    if include_ego:
        handles.append(
            Patch(facecolor=EGO_COLOR, edgecolor="black", alpha=0.55, linewidth=1.2)
        )
        labels.append("ego")

    if include_agents:
        for track_label in T4TrackLabel:
            handles.append(
                Patch(
                    facecolor=BEV_AGENT_COLORS[int(track_label)],
                    edgecolor="black",
                    alpha=0.5,
                    linewidth=1.0,
                )
            )
            labels.append(track_label.name.lower())

    if include_goal:
        handles.append(
            Line2D(
                [0],
                [0],
                color=GOAL_COLOR,
                linewidth=2.0,
                marker=">",
                markersize=6,
            )
        )
        labels.append("Goal Pose")

    return ax.legend(
        handles,
        labels,
        loc=loc,
        fontsize=fontsize,
        framealpha=framealpha,
        ncol=ncol,
        borderpad=0.8,
        labelspacing=0.45,
        columnspacing=1.0,
        handletextpad=0.5,
    )


def render_prediction_bev(
    ground_truth: npt.NDArray[np.floating],
    prediction: npt.NDArray[np.floating],
    lanes: Optional[npt.NDArray[np.floating]] = None,
    route: Optional[npt.NDArray[np.floating]] = None,
    *,
    view_range: float = 60.0,
    title: str = "",
    ego_shape: Optional[npt.NDArray[np.floating]] = None,
) -> npt.NDArray[np.uint8]:
    """Render a trajectory against raw T4 map tensors.

    Inputs are ego-frame ``[T, >=2]`` trajectories and raw, unnormalised
    lane/route tensors. The renderer owns colors, map interpretation, footprint
    and rasterisation.

    :return: RGB ``[H, W, 3]`` image, ready for an image logger or media sink.
    """
    def _as_map(value, shape):
        """Normalize full T4 segments and compact XY polylines alike.

        The canonical T4 representation is ``[N, P, >=8]`` and carries lane
        boundaries and traffic-light attributes. Callers may retain only
        geometry as ``[N, P, 2]`` after slicing the batch tensor.
        Geometry-only input is a useful, lossless contract for a trajectory
        preview, so pad its attributes instead of rejecting it at the shared
        renderer boundary.
        """
        if value is None:
            return np.zeros(shape, dtype=np.float32), False
        values = np.asarray(value, dtype=np.float32)
        if values.ndim != 3 or values.shape[:2] != shape[:2]:
            raise ValueError(
                f"expected map geometry with shape {shape[:2]} and 2 or more "
                f"features; got {values.shape}"
            )
        if values.shape[-1] == 2:
            padded = np.zeros(shape, dtype=np.float32)
            padded[..., :2] = values
            return padded, False
        if values.shape[-1] < 8:
            raise ValueError(
                f"expected map geometry with 2 or >=8 features; got {values.shape}"
            )
        padded = np.zeros(shape, dtype=np.float32)
        width = min(values.shape[-1], shape[-1])
        padded[..., :width] = values[..., :width]
        return padded, True

    lanes_value, lanes_have_attributes = _as_map(
        lanes, (NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM)
    )
    route_value, _route_have_attributes = _as_map(
        route, (NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM)
    )
    map_tensors = MapTensors(
        lanes=lanes_value,
        lanes_speed_limit=np.zeros((lanes_value.shape[0], 1), dtype=np.float32),
        lanes_has_speed_limit=np.zeros((lanes_value.shape[0], 1), dtype=bool),
        route_lanes=route_value,
        route_lanes_speed_limit=np.zeros((route_value.shape[0], 1), dtype=np.float32),
        route_lanes_has_speed_limit=np.zeros((route_value.shape[0], 1), dtype=bool),
        polygons=np.zeros((NUM_POLYGONS, POINTS_PER_POLYGON, 3), dtype=np.float32),
        line_strings=np.zeros((NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 4), dtype=np.float32),
    )
    plt = _pyplot()
    settings = {**BEV_PLOT_CONFIG, "view_range": view_range}
    figure, ax = plt.subplots(figsize=settings["figure_size"], dpi=settings["dpi"])
    add_map_to_bev_ax(
        ax,
        map_tensors,
        layers=("lanes", "route_lanes"),
        # Geometry-only inputs have no boundary offsets.  Do not draw two
        # coincident fake boundaries on top of the centerline.
        draw_lane_boundaries=lanes_have_attributes,
    )
    add_trajectory_to_bev_ax(ax, ground_truth, kind="ground_truth")
    add_trajectory_to_bev_ax(ax, prediction, kind="prediction")
    shape = EgoShape.from_array(ego_shape if ego_shape is not None else [2.7, 4.8, 1.9])
    add_ego_to_bev_ax(ax, shape)
    configure_bev_ax(ax, view_range, settings)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(loc=settings["legend_loc"], fontsize=8, framealpha=0.75)
    figure.tight_layout()
    image = figure_to_rgb(figure)
    plt.close(figure)
    return image


# --------------------------------------------------------------------------- #
# BEV
# --------------------------------------------------------------------------- #


def plot_bev_frame(
    scene: T4Scene,
    trajectories: Optional[Dict[str, Trajectory | npt.NDArray]] = None,
    config: Optional[Dict[str, Any]] = None,
    ax=None,
    title: Optional[str] = None,
):
    """Bird's-eye view of one window.

    :param scene: the window to draw.
    :param trajectories: kind -> trajectory, where kind keys
        :data:`~t4_e2e_devkit.visualization.config.TRAJECTORY_CONFIG`.
    :param config: overrides for :data:`BEV_PLOT_CONFIG`.
    :param ax: draw into these axes rather than a new figure.
    :param title: axis title; a scene summary by default.
    :return: ``(figure, axes)``.
    """
    plt = _pyplot()
    settings = {**BEV_PLOT_CONFIG, **(config or {})}
    if ax is None:
        figure, ax = plt.subplots(figsize=settings["figure_size"], dpi=settings["dpi"])
    else:
        figure = ax.get_figure()

    frame = scene.current_frame
    layers = settings["layers"]
    supplied_trajectories = trajectories or {}

    # Draw context first and trajectories last so plans remain visible above the
    # road graphics.
    if "lidar" in layers and frame.lidar is not None:
        add_lidar_to_bev_ax(ax, frame.lidar)

    ego_pose = np.asarray(frame.ego_status.ego_pose, dtype=np.float64).reshape(-1)
    add_ego_to_bev_ax(
        ax,
        frame.ego_status.ego_shape,
        x=float(ego_pose[0]),
        y=float(ego_pose[1]),
        heading=float(ego_pose[2]),
        label=None,
    )

    if settings.get("show_history", True):
        history = supplied_trajectories.get("history")
        if history is None:
            history = scene.get_history_poses()
        add_trajectory_to_bev_ax(ax, history, kind="history", include_origin=False)

    if "ground_truth" in supplied_trajectories:
        future = (
            scene.future_ego_poses
            if scene.future_ego_poses is not None
            else supplied_trajectories["ground_truth"]
        )
        add_ego_future_to_bev_ax(
            ax,
            getattr(future, "poses", future),
            frame.ego_status.ego_shape,
            draw_footprints=settings.get("show_ego_future_footprints", True),
        )

    if "annotations" in layers and frame.annotations is not None:
        history_annotations = [
            history_frame.annotations
            for history_frame in scene.frames[: scene.current_frame_index + 1]
            if history_frame.annotations is not None
        ]
        add_annotations_to_bev_ax(
            ax,
            frame.annotations,
            label_classes=bool(settings.get("legend", False)),
            history=history_annotations,
        )
        if settings.get("show_neighbor_future", True):
            future_annotations = scene.future_annotations
            add_future_annotations_to_bev_ax(
                ax,
                frame.annotations,
                future_annotations[1:] if future_annotations else None,
            )

    if frame.map_tensors is not None:
        add_map_to_bev_ax(ax, frame.map_tensors, layers=layers)

    if scene.goal_pose is not None:
        add_goal_to_bev_ax(ax, scene.goal_pose)

    for kind, trajectory in supplied_trajectories.items():
        if kind in {"history", "ground_truth"}:
            continue
        add_trajectory_to_bev_ax(ax, trajectory, kind=kind, include_origin=True)

    configure_bev_ax(ax, settings["view_range"], settings)
    if settings.get("status_text", True):
        previous_turn_indicator = None
        if scene.current_frame_index > 0:
            previous_turn_indicator = scene.frames[
                scene.current_frame_index - 1
            ].ego_status.turn_indicator
        add_bev_status_text(
            ax,
            frame.ego_status,
            settings["view_range"],
            previous_turn_indicator=previous_turn_indicator,
        )
    ax.set_title(title if title is not None else describe_scene(scene), fontsize=10)
    if settings["legend"]:
        legend_roles = [
            role
            for role in ("history", "ground_truth", "prediction")
            if role in supplied_trajectories
        ]
        if settings.get("show_history", True) and "history" not in legend_roles:
            legend_roles.insert(0, "history")
        add_fixed_bev_legend(
            ax,
            trajectory_roles=legend_roles,
            include_agents=frame.annotations is not None,
            include_goal=scene.goal_pose is not None,
            loc=settings["legend_loc"],
            fontsize=8,
            framealpha=0.75,
            ncol=int(settings.get("legend_ncol", 2)),
        )
    figure.tight_layout()
    return figure, ax


def reference_trajectories(
    scene: T4Scene,
    include_human: bool = True,
    include_history: bool = True,
) -> Dict[str, Any]:
    """Every trajectory a window can supply on its own, ready to plot.

    Each is skipped when the window does not carry it, so this never fabricates a
    line.

    :param scene: the window.
    :param include_human: the recorded future.
    :param include_history: the recorded past.
    :return: kind -> trajectory, keyed for
        :data:`~t4_e2e_devkit.visualization.config.TRAJECTORY_CONFIG`.
    """
    trajectories: Dict[str, Any] = {}
    if include_history:
        trajectories["history"] = scene.get_history_poses()
    if include_human and scene.future_ego_poses is not None:
        trajectories["ground_truth"] = scene.get_future_trajectory()
    return trajectories


def plot_bev_with_agent(
    scene: T4Scene,
    agent,
    include_human: bool = True,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Bird's-eye view with one agent's plan, and everything the window itself supplies.
    :param scene: the window.
    :param agent: an :class:`~t4_e2e_devkit.agents.AbstractT4Agent`.
    :param include_human: also draw the recorded trajectory.
    :param config: overrides for :data:`BEV_PLOT_CONFIG`.
    :return: ``(figure, axes)``.
    """
    trajectories = reference_trajectories(scene, include_human=include_human)
    trajectories["prediction"] = _plan(agent, scene)
    return plot_bev_frame(scene, trajectories, config)


def plot_bev_with_score(
    scene: T4Scene,
    trajectory: Trajectory,
    results: Mapping[str, Any] | Any,
    config: Optional[Dict[str, Any]] = None,
):
    """Bird's-eye view beside the score that trajectory earned.

    The component panel marks the multiplicative PDM gates separately from the
    weighted terms.

    :param scene: the window.
    :param trajectory: the scored trajectory.
    :param results: its components and aggregate.
    :param config: overrides for :data:`BEV_PLOT_CONFIG`.
    :return: ``(figure, axes)``.
    """
    plt = _pyplot()
    settings = {**BEV_PLOT_CONFIG, **(config or {})}
    figure, (bev_ax, score_ax) = plt.subplots(
        1, 2, figsize=(settings["figure_size"][0] * 1.55, settings["figure_size"][1]),
        dpi=settings["dpi"], gridspec_kw={"width_ratios": [2.4, 1.0]},
    )

    trajectories: Dict[str, Any] = {"prediction": trajectory}
    if scene.future_ego_poses is not None:
        trajectories["ground_truth"] = scene.get_future_trajectory()
    plot_bev_frame(scene, trajectories, settings, ax=bev_ax)

    add_score_panel(score_ax, results)
    figure.tight_layout()
    return figure, (bev_ax, score_ax)


def add_score_panel(
    ax,
    results: Mapping[str, Any] | Any,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Draw the available PDM components as a horizontal bar chart.
    :param ax: target axes.
    :param results: the score to draw.
    :param config: overrides for :data:`SCORE_PANEL_CONFIG`.
    :return: the axes.
    """
    settings = {**SCORE_PANEL_CONFIG, **(config or {})}
    components = results if isinstance(results, Mapping) else results.values
    names = [
        name for name in NAVSIM_METRICS
        if name not in {"score", "extended_comfort_available"} and name in components
    ]
    values = [float(components[name]) for name in names]
    gates = {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
    }

    colors = [
        settings["zero_color"]
        if value <= 0.0
        else (settings["gate_color"] if name in gates else settings["weighted_color"])
        for name, value in zip(names, values, strict=True)
    ]
    positions = np.arange(len(names))
    ax.barh(positions, values, height=settings["bar_height"], color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{name.upper()}*" if name in gates else name.upper() for name in names],
        fontsize=settings["text_size"],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.axvline(1.0, color="grey", linewidth=0.6, linestyle=":")
    for position, value in zip(positions, values, strict=True):
        ax.text(min(value + 0.03, 0.97), position, f"{value:.2f}",
                va="center", fontsize=settings["text_size"] - 1)

    score = float(components.get("score", getattr(results, "score", float("nan"))))
    ax.set_title(f"PDM {score:.4f}", fontsize=settings["text_size"] + 2)
    ax.set_xlabel("* multiplicative gate", fontsize=settings["text_size"] - 1)
    return ax


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def plot_cameras_frame(
    scene: T4Scene,
    with_annotations: bool = False,
    with_lidar: bool = False,
    trajectory: Optional[Trajectory] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """The camera register of one window, laid out by where the views point.

    :param scene: the window; its cameras must have been decoded.
    :param with_annotations: overlay projected 3D boxes.
    :param with_lidar: overlay the projected point cloud.
    :param trajectory: overlay a planned path on the road surface.
    :param config: overrides for :data:`CAMERAS_PLOT_CONFIG`.
    :return: ``(figure, axes)``.
    :raises ValueError: when no cameras were decoded, which means the scene was
        read with a sensor config that did not ask for them.
    """
    plt = _pyplot()
    settings = {**CAMERAS_PLOT_CONFIG, **(config or {})}
    frame = scene.current_frame
    if frame.cameras is None or len(frame.cameras) == 0:
        raise ValueError(
            "this window carries no decoded cameras; build it with a SensorConfig "
            "that requests them, e.g. rigs.sensor_config_for_scene(scene_dir)"
        )

    grid = camera_grid_layout(frame.cameras.names)
    rows, columns = len(grid), max(len(row) for row in grid)
    figure, axes = plt.subplots(
        rows, columns, figsize=settings["figure_size"], dpi=settings["dpi"], squeeze=False
    )

    for row_index, row in enumerate(grid):
        for column_index in range(columns):
            ax = axes[row_index][column_index]
            name = row[column_index] if column_index < len(row) else None
            if name is None:
                ax.axis("off")
                continue
            camera = frame.cameras[name]
            add_camera_ax(ax, camera)
            if with_lidar and frame.lidar is not None:
                add_lidar_to_camera_ax(ax, camera, frame.lidar)
            if with_annotations and frame.annotations is not None:
                add_annotations_to_camera_ax(ax, camera, frame.annotations)
            if trajectory is not None:
                add_trajectory_to_camera_ax(
                    ax, camera, trajectory.poses, TRAJECTORY_CONFIG["prediction"]["color"]
                )

    figure.suptitle(describe_scene(scene), fontsize=settings["title_size"] + 1)
    figure.tight_layout()
    return figure, axes


def plot_scene_summary(
    scene: T4Scene,
    trajectories: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """One figure showing the BEV and the camera register together.

    The most useful single view for inspecting a window, because the two halves
    fail differently: a plan that looks fine in BEV can be driving at something
    only the cameras show, and a map defect is invisible in the images.

    :param scene: the window.
    :param trajectories: kind -> trajectory, as in :func:`plot_bev_frame`.
    :param config: overrides for :data:`BEV_PLOT_CONFIG`.
    :return: ``(figure, axes)``.
    """
    plt = _pyplot()
    frame = scene.current_frame
    has_cameras = frame.cameras is not None and len(frame.cameras) > 0
    if not has_cameras:
        return plot_bev_frame(scene, trajectories, config)

    grid = camera_grid_layout(frame.cameras.names)
    columns = max(len(row) for row in grid)
    figure = plt.figure(figsize=(6.0 + 3.6 * columns, 8.4), dpi=100)
    spec = figure.add_gridspec(len(grid), columns + 2)

    bev_ax = figure.add_subplot(spec[:, :2])
    plot_bev_frame(scene, trajectories, config, ax=bev_ax, title="")

    prediction = (trajectories or {}).get("prediction")
    for row_index, row in enumerate(grid):
        for column_index in range(columns):
            ax = figure.add_subplot(spec[row_index, column_index + 2])
            name = row[column_index] if column_index < len(row) else None
            if name is None:
                ax.axis("off")
                continue
            camera = frame.cameras[name]
            add_camera_ax(ax, camera)
            if frame.annotations is not None:
                add_annotations_to_camera_ax(ax, camera, frame.annotations)
            if prediction is not None:
                poses = getattr(prediction, "poses", prediction)
                add_trajectory_to_camera_ax(
                    ax, camera, poses, TRAJECTORY_CONFIG["prediction"]["color"]
                )
    figure.suptitle(describe_scene(scene), fontsize=11)
    figure.tight_layout()
    return figure, bev_ax


def plot_agent_comparison(
    scene: T4Scene,
    agents: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
):
    """Several agents' plans on one window.

    :param scene: the window.
    :param agents: label -> agent.
    :param config: overrides for :data:`BEV_PLOT_CONFIG`.
    :return: ``(figure, axes)``.
    """
    plt = _pyplot()
    settings = {**BEV_PLOT_CONFIG, **(config or {})}
    figure, ax = plt.subplots(figsize=settings["figure_size"], dpi=settings["dpi"])

    base_trajectories: Dict[str, Any] = {}
    if scene.future_ego_poses is not None:
        base_trajectories["ground_truth"] = scene.get_future_trajectory()
    plot_bev_frame(scene, base_trajectories, settings, ax=ax, title="")

    from t4_e2e_devkit.visualization.config import TAB_10

    for index, (label, agent) in enumerate(agents.items()):
        add_trajectory_to_bev_ax(
            ax, _plan(agent, scene), kind="prediction",
            config={"color": TAB_10[index % len(TAB_10)], "label": label},
        )
    configure_bev_ax(ax, settings["view_range"], settings)
    ax.set_title(describe_scene(scene), fontsize=10)
    ax.legend(loc=settings["legend_loc"], fontsize=8, framealpha=0.75)
    figure.tight_layout()
    return figure, ax


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def figure_to_rgb(figure) -> npt.NDArray[np.uint8]:
    """
    Rasterise a figure.
    :param figure: the figure.
    :return: ``[H, W, 3]`` uint8 RGB for any image logging backend.
    """
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()


def save_figure(figure, path, close: bool = True):
    """
    Write a figure to disk.
    :param figure: the figure.
    :param path: destination; the suffix picks the format.
    :param close: close the figure afterwards.
    :return: the path.
    """
    from pathlib import Path

    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    if close:
        plt.close(figure)
    return path


def frames_to_gif(frames: Sequence[npt.NDArray[np.uint8]], path, duration_ms: int = 200):
    """
    Write rasterised frames as an animated GIF.
    :param frames: ``[H, W, 3]`` uint8 arrays.
    :param path: destination.
    :param duration_ms: milliseconds per frame.
    :return: the path.
    """
    from pathlib import Path

    from PIL import Image

    if not frames:
        raise ValueError("no frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0
    )
    return path


def describe_scene(scene: T4Scene) -> str:
    """
    A one-line caption identifying a window.
    :param scene: the window.
    :return: the caption.
    """
    metadata = scene.scene_metadata
    status = scene.current_frame.ego_status
    parts = [
        metadata.token,
        f"{status.speed:.1f} m/s",
    ]
    if scene.current_frame.annotations is not None:
        parts.append(f"{len(scene.current_frame.annotations)} agents")
    return "  |  ".join(parts)


def _plan(agent, scene: T4Scene) -> Trajectory:
    """Plan with an agent, honouring whether it is an oracle."""
    if getattr(agent, "requires_scene", False):
        return agent.compute_trajectory_from_scene(scene)
    return agent.compute_trajectory(scene.get_agent_input())
