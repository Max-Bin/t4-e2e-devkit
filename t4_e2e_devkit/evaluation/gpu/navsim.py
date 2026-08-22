"""Batched CUDA implementation of the T4 PDM metrics.

The scene preparation is deliberately kept at the batch boundary.  Once map,
track and trajectory tensors are on the selected device, geometry, simulation,
comfort and profile aggregation stay there until one final result copy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np
import torch

from t4_e2e_devkit.common.dataclasses import T4Scene, Trajectory
from t4_e2e_devkit.evaluation.gpu.areas import points_in_polygons_torch
from t4_e2e_devkit.evaluation.gpu.comfort import (
    _apply_savgol,
    _phase_unwrap,
    ego_is_comfortable_torch,
)
from t4_e2e_devkit.evaluation.gpu.geometry import TorchPolyline
from t4_e2e_devkit.evaluation.gpu.geometry_ops import (
    oriented_box_corners,
    segment_intersects_quad,
)
from t4_e2e_devkit.evaluation.gpu.oracle import (
    VehicleTensors,
    WindowScene,
    score_simulated_window,
)
from t4_e2e_devkit.evaluation.gpu.scene import (
    extract_window_scene_arrays,
    window_scene_from_arrays,
    lane_change_exempt,
)
from t4_e2e_devkit.evaluation.gpu.precision import simulate_proposals_fp32
from t4_e2e_devkit.evaluation.gpu.simulate import TorchSimulatorConfig
from t4_e2e_devkit.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex,
)


@dataclass
class _PreparedScene:
    source: T4Scene
    gpu_scene: Optional[WindowScene]
    red_starts: torch.Tensor
    red_ends: torch.Tensor
    route_centerlines: tuple[torch.Tensor, ...]
    intersection_starts: torch.Tensor
    intersection_ends: torch.Tensor
    border_starts: torch.Tensor
    border_ends: torch.Tensor
    lane_change: Optional[torch.Tensor]
    vehicle: VehicleTensors
    metadata: dict[str, float]


def score_navsim_batch(
    trajectories: Sequence[Trajectory],
    scenes: Sequence[T4Scene],
    config: Any,
    *,
    previous_trajectories: Optional[Sequence[Trajectory | None]] = None,
    previous_scenes: Optional[Sequence[T4Scene | None]] = None,
    human_trajectories: Optional[Sequence[Trajectory | None]] = None,
) -> list[dict[str, Any]]:
    """Score one batch on CUDA and return host-side scalar dictionaries.

    The returned dictionaries contain the model and human-filter reference
    components. Applying the filter and version-specific weighted
    aggregate remains in :mod:`evaluation.navsim_score`, so CPU and GPU share
    exactly the same final aggregation code.
    """

    if len(trajectories) != len(scenes) or not scenes:
        raise ValueError("trajectories and scenes must have the same non-zero length")
    from t4_e2e_devkit.evaluation.navsim_score import (
        NAVSIM_COMPONENT_METRICS,
        required_navsim_metric_names,
        resolve_navsim_metric_names,
    )

    selected_metrics = resolve_navsim_metric_names(
        config.version, getattr(config, "metric_names", None)
    )
    required_metrics = required_navsim_metric_names(config.version, selected_metrics)
    needs_human_filter = bool(
        config.version == "v2"
        and config.use_human_filter
        and required_metrics.intersection(NAVSIM_COMPONENT_METRICS)
    )
    needs_states = bool(
        required_metrics.intersection(
            {
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
                "history_comfort",
            }
        )
        or (
            config.version == "v1"
            and "history_comfort" in required_metrics
        )
        or (
            "extended_comfort" in required_metrics
            and any(value is not None for value in previous_trajectories or ())
        )
    )
    batch_size = len(scenes)
    if previous_trajectories is None:
        previous_trajectories = [None] * batch_size
    if previous_scenes is None:
        previous_scenes = [None] * batch_size
    if len(previous_trajectories) != batch_size or len(previous_scenes) != batch_size:
        raise ValueError("previous trajectory and scene batches must match scenes")
    if human_trajectories is None:
        human_trajectories = [None] * batch_size
    if len(human_trajectories) != batch_size:
        raise ValueError("human trajectory batch must match scenes")

    device = torch.device(config.device or "cuda")
    dtype = torch.float64 if str(getattr(config, "gpu_dtype", "float32")) == "float64" else torch.float32
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the PDM GPU backend requires an available CUDA device")

    prepared = [
        _prepare_scene(scene, config, device, dtype, required_metrics)
        for scene in scenes
    ]
    plan_np = [_dense_poses(item, config) for item in trajectories]
    plan = torch.as_tensor(np.stack(plan_np), device=device, dtype=dtype)
    human = None
    if "ego_progress" in required_metrics or needs_human_filter:
        human_np = [
            _dense_poses(
                human_trajectory
                if human_trajectory is not None
                else scene.get_future_trajectory(trajectory_sampling=_sampling(config)),
                config,
            )
            for human_trajectory, scene in zip(human_trajectories, scenes, strict=True)
        ]
        human = torch.as_tensor(np.stack(human_np), device=device, dtype=dtype)
    speeds = torch.as_tensor(
        [float(scene.current_frame.ego_status.speed) for scene in scenes],
        device=device,
        dtype=dtype,
    )
    if needs_states:
        if human is None:
            human = torch.zeros_like(plan)
        model_states, human_states = _simulate_pair(plan, human, speeds, prepared, config, dtype)
    else:
        model_states, human_states = None, None

    previous_states: list[torch.Tensor | None] = [None] * batch_size
    previous_indices = [index for index, value in enumerate(previous_trajectories) if value is not None]
    if previous_indices and "extended_comfort" in required_metrics:
        if any(previous_scenes[index] is None for index in previous_indices):
            raise ValueError("each previous trajectory needs a previous scene")
        previous_plan = torch.as_tensor(
            np.stack([_dense_poses(previous_trajectories[index], config) for index in previous_indices]),
            device=device,
            dtype=dtype,
        )
        previous_speed = torch.as_tensor(
            [float(previous_scenes[index].current_frame.ego_status.speed) for index in previous_indices],
            device=device,
            dtype=dtype,
        )
        previous_states_batch = _simulate(
            previous_plan,
            previous_speed,
            [
                float(previous_scenes[index].current_frame.ego_status.ego_shape.wheel_base)
                for index in previous_indices
            ],
            config,
            dtype,
        )
        for local, index in enumerate(previous_indices):
            previous_states[index] = previous_states_batch[local]

    model_values: list[dict[str, float]] = []
    human_values: list[dict[str, float]] = []
    extended_values = torch.full(
        (batch_size,), float("nan"), device=device, dtype=dtype
    )
    for index, item in enumerate(prepared):
        model = _components(
            plan[index],
            None if model_states is None else model_states[index],
            item,
            config,
            None if human is None else human[index],
            required_metrics,
        )
        model_values.append({key: float(value) for key, value in model.items()})
        if needs_human_filter:
            human_result = _components(
                poses=human[index] if human is not None else plan[index],
                states=None if human_states is None else human_states[index],
                prepared=item,
                config=config,
                human_poses=None if human is None else human[index],
                metric_names=required_metrics,
            )
            human_values.append(
                {key: float(value) for key, value in human_result.items()}
            )
        if (
            config.version == "v2"
            and "extended_comfort" in required_metrics
            and model_states is not None
        ):
            previous_scene = previous_scenes[index]
            if previous_states[index] is not None and previous_scene is not None:
                delta_s = (
                    float(
                        item.source.current_frame.timestamp_us
                        - previous_scene.current_frame.timestamp_us
                    )
                    / 1e6
                )
                if delta_s <= 0.0:
                    delta_s = float(config.observation_interval_s)
                previous_global = _states_to_global(previous_states[index], previous_scene)
                current_global = _states_to_global(model_states[index], item.source)
                extended_values[index] = _extended_comfort_gpu(
                    previous_global,
                    current_global,
                    config.interval_s,
                    delta_s,
                )
    extended_host = extended_values.detach().cpu().numpy()
    outputs: list[dict[str, Any]] = []
    for index, item in enumerate(prepared):
        human_result = human_values[index] if needs_human_filter else None
        extended = float(extended_host[index])
        outputs.append(
            {
                "components": model_values[index],
                "human_components": human_result,
                "extended_comfort": extended if np.isfinite(extended) else None,
                "metadata": item.metadata,
            }
        )
    return outputs


def _sampling(config: Any):
    from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    return TrajectorySampling(
        num_poses=int(config.num_steps), interval_length=float(config.interval_s)
    )


def _dense_poses(trajectory: Trajectory, config: Any) -> np.ndarray:
    try:
        dense = trajectory.resample(_sampling(config))
    except ValueError as error:
        raise ValueError(f"trajectory must cover {config.horizon_s:g}s for PDM scoring") from error
    values = np.asarray(dense.poses, dtype=np.float64)
    return np.vstack((np.zeros((1, 3), dtype=np.float64), values))


def _prepare_scene(
    scene: T4Scene,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    required_metrics: Sequence[str],
) -> _PreparedScene:
    frame = scene.current_frame
    map_tensors = frame.map_tensors
    shape = frame.ego_status.ego_shape
    needs_map = bool(
        set(required_metrics).intersection(
            {
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "driving_direction_compliance",
                "traffic_light_compliance",
                "time_to_collision_within_bound",
                "lane_keeping",
            }
        )
        or (config.version == "v1" and "history_comfort" in required_metrics)
    )
    gpu_scene = None
    red_starts = torch.empty((0, 0, 2), device=device, dtype=dtype)
    red_ends = torch.empty((0, 0, 2), device=device, dtype=dtype)
    route_centerlines: tuple[torch.Tensor, ...] = ()
    intersection_starts = torch.empty((0, 0, 2), device=device, dtype=dtype)
    intersection_ends = torch.empty((0, 0, 2), device=device, dtype=dtype)
    border_starts = torch.empty((0, 2), device=device, dtype=dtype)
    border_ends = torch.empty((0, 2), device=device, dtype=dtype)
    coverage: dict[str, float] = {}

    # One numpy implementation feeds both backends; see ``lane_change_exempt``.
    lane_change: Optional[torch.Tensor] = None
    steps = int(config.num_steps) + 1
    indicators = scene.future_turn_indicators
    if indicators is not None and len(indicators) >= steps:
        lane_change = torch.as_tensor(
            lane_change_exempt(
                indicators[:steps],
                float(config.interval_s),
                float(config.lane_keeping_lane_change_pre_s),
                float(config.lane_keeping_lane_change_post_s),
            ),
            device=device,
        )

    if needs_map:
        if map_tensors is None:
            raise ValueError("PDM GPU scoring requires current-frame map tensors")
        annotations = scene.future_annotations
        needs_annotations = bool(
            set(required_metrics).intersection(
                {"no_at_fault_collisions", "time_to_collision_within_bound"}
            )
        )
        required_annotations = int(config.num_steps) + 10
        if needs_annotations and (
            annotations is None or len(annotations) < required_annotations
        ):
            raise ValueError(
                "PDM GPU scoring requires "
                f"at least {required_annotations} current/future annotation frames"
            )
        if annotations is None:
            empty_boxes = [np.zeros((0, 9), dtype=np.float64)] * required_annotations
            empty_labels = [np.zeros((0,), dtype=np.int64)] * required_annotations
            boxes, labels = empty_boxes, empty_labels
        else:
            boxes = [np.asarray(annotation.boxes) for annotation in annotations]
            labels = [np.asarray(annotation.labels) for annotation in annotations]
        try:
            arrays = extract_window_scene_arrays(
                np.asarray(map_tensors.lanes),
                np.asarray(map_tensors.route_lanes),
                np.asarray(map_tensors.polygons),
                boxes,
                labels,
            )
        except ValueError as error:
            raise ValueError(
                "PDM GPU scoring requires usable lane and route geometry"
            ) from error
        tracks, gpu_map, centerline = window_scene_from_arrays(arrays, device, dtype)
        navsim_drivable = torch.cat(
            (gpu_map.lane_indices, gpu_map.intersection_indices), dim=0
        ).unique()
        gpu_map = replace(gpu_map, drivable_area_indices=navsim_drivable)
        gpu_scene = WindowScene(
            tracks=tracks,
            map_tensors=gpu_map,
            centerline=centerline,
        )

        red = torch.as_tensor(arrays["red_light_rings"], device=device, dtype=dtype)
        if red.ndim == 3 and red.shape[0]:
            red_starts, red_ends = red[:, :-1], red[:, 1:]

        route = np.asarray(map_tensors.route_lanes, dtype=np.float64)
        route_rows = _valid_rows(route, min_columns=8)
        route_centerlines = tuple(
            torch.as_tensor(row[:, :2], device=device, dtype=dtype)
            for row in route_rows
            if row.shape[0] >= 2
        )
        intersections = gpu_map.intersection_indices
        intersection_starts = gpu_map.edge_starts[intersections]
        intersection_ends = gpu_map.edge_ends[intersections]
        border_starts, border_ends, border_count = _border_segments(
            map_tensors.line_strings, device, dtype
        )
        line_rows = _valid_rows(np.asarray(map_tensors.line_strings), min_columns=2)
        red_count = sum(
            row.shape[1] > 10 and bool((row[:, 10] > 0.5).any())
            for row in route_rows
        )
        coverage = {
            "dac_border_frac": float(border_count / max(len(line_rows), 1)),
            "ddc_route_frac": float(len(route_rows) / max(len(route_rows), 1)),
            "lane_centerline_frac": float(len(route_centerlines) / max(len(route_rows), 1)),
            "traffic_light_route_frac": float(red_count / max(len(route_rows), 1)),
        }
    return _PreparedScene(
        source=scene,
        gpu_scene=gpu_scene,
        red_starts=red_starts,
        red_ends=red_ends,
        route_centerlines=route_centerlines,
        intersection_starts=intersection_starts,
        intersection_ends=intersection_ends,
        border_starts=border_starts,
        border_ends=border_ends,
        lane_change=lane_change,
        vehicle=VehicleTensors(
            half_length=float(shape.length) / 2.0,
            half_width=float(shape.width) / 2.0,
            rear_axle_to_center=float(shape.rear_axle_to_center),
        ),
        metadata=coverage,
    )


def _valid_rows(values: np.ndarray, *, min_columns: int) -> list[np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] < min_columns:
        return []
    rows: list[np.ndarray] = []
    for row in array:
        valid = np.linalg.norm(row[:, :2], axis=-1) > 1.0e-6
        if int(valid.sum()) >= 2:
            rows.append(np.ascontiguousarray(row[valid]))
    return rows


def _border_segments(
    values: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Every drivable-area border segment as one ``[S, 2]`` start/end pair.

    Concatenated across line-strings rather than kept per line-string:
    ``_drivable_area`` reduces the whole set with ``any()``, so one vectorised
    kernel call over all segments is equivalent to a call per border and costs
    about twenty times fewer launches -- and launch overhead, not arithmetic,
    dominated the metric.  The third element is the number of contributing
    line-strings, which the caller reports as ``dac_border_frac``.
    """

    starts: list[torch.Tensor] = []
    ends: list[torch.Tensor] = []
    for row in _valid_rows(values, min_columns=2):
        if row.shape[1] < 4 or not bool((row[:, 3] > 0.5).any()):
            continue
        points = torch.as_tensor(row[:, :2], device=device, dtype=dtype)
        starts.append(points[:-1])
        ends.append(points[1:])
    if not starts:
        empty = torch.empty((0, 2), device=device, dtype=dtype)
        return empty, empty, 0
    return torch.cat(starts), torch.cat(ends), len(starts)


def _simulate_pair(
    plan: torch.Tensor,
    human: torch.Tensor,
    speeds: torch.Tensor,
    prepared: Sequence[_PreparedScene],
    config: Any,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_poses = torch.cat((plan, human), dim=0)
    all_speeds = torch.cat((speeds, speeds), dim=0)
    wheelbases = [
        float(item.source.current_frame.ego_status.ego_shape.wheel_base)
        for item in prepared
    ]
    states = _simulate(all_poses, all_speeds, wheelbases * 2, config, dtype)
    batch = plan.shape[0]
    return states[:batch], states[batch:]


def _simulate(
    poses: torch.Tensor,
    speeds: torch.Tensor,
    wheelbases: Sequence[float],
    config: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not config.use_simulator:
        return _states_from_poses(poses, float(config.interval_s))
    batch, time = poses.shape[:2]
    proposal_states = torch.zeros((batch, time, 11), device=poses.device, dtype=dtype)
    proposal_states[..., :3] = poses
    initial_states = torch.zeros((batch, 11), device=poses.device, dtype=dtype)
    initial_states[:, StateIndex.VELOCITY_X] = speeds
    if len(wheelbases) != batch:
        raise ValueError("wheelbase metadata must match the simulation batch")
    groups: dict[float, list[int]] = {}
    for index, wheel_base in enumerate(wheelbases):
        value = float(wheel_base)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"wheel base must be finite and positive, got {value}")
        groups.setdefault(round(value, 8), []).append(index)
    simulated = torch.empty_like(proposal_states)
    for wheel_base, indices in groups.items():
        index_tensor = torch.as_tensor(indices, device=poses.device, dtype=torch.long)
        simulated.index_copy_(
            0,
            index_tensor,
            simulate_proposals_fp32(
                proposal_states.index_select(0, index_tensor),
                initial_states.index_select(0, index_tensor),
                TorchSimulatorConfig(
                    wheel_base=wheel_base,
                    discretization_time=float(config.interval_s),
                ),
                compile_rollout=bool(getattr(config, "compile_rollout", False)),
            ),
        )
    return simulated


def _states_from_poses(poses: torch.Tensor, dt: float) -> torch.Tensor:
    if poses.shape[-2] == 0:
        return torch.zeros(
            (*poses.shape[:-2], 0, 11), device=poses.device, dtype=poses.dtype
        )
    x, y, heading = poses.unbind(dim=-1)
    vx = _gradient(x, dt)
    vy = _gradient(y, dt)
    ax = _gradient(vx, dt)
    ay = _gradient(vy, dt)
    cos_h, sin_h = heading.cos(), heading.sin()
    states = torch.zeros((*poses.shape[:-1], 11), device=poses.device, dtype=poses.dtype)
    states[..., StateIndex.X] = x
    states[..., StateIndex.Y] = y
    states[..., StateIndex.HEADING] = heading
    states[..., StateIndex.VELOCITY_X] = vx * cos_h + vy * sin_h
    states[..., StateIndex.VELOCITY_Y] = -vx * sin_h + vy * cos_h
    states[..., StateIndex.ACCELERATION_X] = ax * cos_h + ay * sin_h
    states[..., StateIndex.ACCELERATION_Y] = -ax * sin_h + ay * cos_h
    return states


def _states_to_global(states: torch.Tensor, scene: T4Scene) -> torch.Tensor:
    """Move simulated local-frame states into the scene's global frame.

    The two plans used by extended comfort come from consecutive windows, so
    their local origins and headings are different.  Positions and heading
    must therefore be transformed before comparing their kinematic features.
    Velocity and acceleration remain expressed in the vehicle frame and are
    intentionally unchanged.
    """

    center = scene.scene_metadata.global_center_pose
    if center is None:
        return states
    values = np.asarray(center, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(
            "global_center_pose must contain finite [x, y, cos_heading, sin_heading]"
        )
    transform = torch.as_tensor(values, device=states.device, dtype=states.dtype)
    cosine, sine = transform[2], transform[3]
    local_x = states[..., StateIndex.X]
    local_y = states[..., StateIndex.Y]
    states_global = states.clone()
    states_global[..., StateIndex.X] = transform[0] + cosine * local_x - sine * local_y
    states_global[..., StateIndex.Y] = transform[1] + sine * local_x + cosine * local_y
    states_global[..., StateIndex.HEADING] = states[..., StateIndex.HEADING] + torch.atan2(
        sine, cosine
    )
    return states_global


def _gradient(values: torch.Tensor, dt: float) -> torch.Tensor:
    output = torch.empty_like(values)
    if values.shape[-1] == 1:
        output.zero_()
        return output
    output[..., 0] = (values[..., 1] - values[..., 0]) / dt
    output[..., -1] = (values[..., -1] - values[..., -2]) / dt
    if values.shape[-1] > 2:
        output[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / (2.0 * dt)
    return output


def _components(
    poses: torch.Tensor,
    states: Optional[torch.Tensor],
    prepared: _PreparedScene,
    config: Any,
    human_poses: Optional[torch.Tensor],
    metric_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    required = set(metric_names)
    needs_oracle = bool(
        required.intersection(
            {
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
            }
        )
        or (config.version == "v1" and "history_comfort" in required)
    )
    components = None
    if needs_oracle:
        if states is None or prepared.gpu_scene is None:
            raise ValueError("selected GPU metrics require simulated scene tensors")
        total, components = score_simulated_window(
            states.unsqueeze(0),
            prepared.gpu_scene,
            prepared.vehicle,
            interval_length=float(config.interval_s),
        )
        del total

    values: dict[str, torch.Tensor] = {}
    if "no_at_fault_collisions" in required:
        assert components is not None
        values["no_at_fault_collisions"] = components[0, 0]
    if "drivable_area_compliance" in required:
        assert components is not None
        values["drivable_area_compliance"] = _drivable_area(
            poses, prepared, components[0, 1]
        )
    if "driving_direction_compliance" in required:
        values["driving_direction_compliance"] = _driving_direction(
            poses, prepared, config
        )
    if "traffic_light_compliance" in required:
        values["traffic_light_compliance"] = _traffic_light(poses, prepared)
    if "ego_progress" in required:
        if human_poses is None:
            raise ValueError("ego progress requires a human reference trajectory")
        values["ego_progress"] = _ego_progress(poses, human_poses, config)
    if "time_to_collision_within_bound" in required:
        assert components is not None
        values["time_to_collision_within_bound"] = components[0, 3]
    if "lane_keeping" in required:
        values["lane_keeping"] = _lane_keeping(poses, prepared, config)
    if "history_comfort" in required:
        if config.version == "v1":
            assert components is not None
            values["history_comfort"] = components[0, 5]
        else:
            if states is None:
                raise ValueError("history comfort requires simulated states")
            values["history_comfort"] = _history_comfort(
                states, prepared.source, config
            )
    return values


def _ego_progress(poses: torch.Tensor, reference: torch.Tensor, config: Any) -> float:
    line = TorchPolyline(reference[:, :2])
    endpoints = torch.stack((poses[0, :2], poses[-1, :2]))
    progress = line.project(endpoints)
    raw = (progress[1] - progress[0]).clamp(min=0.0)
    reference_end = line.project(torch.stack((reference[0, :2], reference[-1, :2])))
    denominator = (reference_end[1] - reference_end[0]).clamp(min=0.0)
    maximum = torch.maximum(raw, denominator)
    return torch.where(
        maximum > float(config.progress_distance_threshold),
        (raw / maximum.clamp_min(1.0e-12)).clamp(0.0, 1.0),
        torch.ones_like(maximum),
    )


def _traffic_light(poses: torch.Tensor, prepared: _PreparedScene) -> torch.Tensor:
    if not prepared.red_starts.shape[0]:
        return torch.ones((), device=poses.device, dtype=poses.dtype)
    shape = prepared.source.current_frame.ego_status.ego_shape
    headings = poses[:, 2]
    centers = poses[:, :2] + float(shape.rear_axle_to_center) * torch.stack(
        (headings.cos(), headings.sin()), dim=-1
    )
    corners = oriented_box_corners(
        centers,
        headings,
        torch.full_like(headings, float(shape.length) / 2.0),
        torch.full_like(headings, float(shape.width) / 2.0),
    )
    inside_red = points_in_polygons_torch(
        corners.reshape(-1, 2), prepared.red_starts, prepared.red_ends
    ).any(dim=0).reshape(-1, 4).any(dim=-1)
    red_vertices_inside = points_in_polygons_torch(
        prepared.red_starts.reshape(-1, 2),
        corners,
        torch.roll(corners, shifts=-1, dims=-2),
    ).reshape(poses.shape[0], -1).any(dim=-1)
    ego_starts = corners
    ego_ends = torch.roll(corners, shifts=-1, dims=-2)
    red_starts = prepared.red_starts[None, None, :, :, :]
    red_ends = prepared.red_ends[None, None, :, :, :]
    ego_start = ego_starts[:, :, None, None, :]
    ego_end = ego_ends[:, :, None, None, :]
    edge_hit = _segments_intersect(
        ego_start,
        ego_end,
        red_starts,
        red_ends,
    ).any(dim=(1, 2, 3))
    return (~(inside_red | red_vertices_inside | edge_hit)).all().to(poses.dtype)


def _drivable_area(
    poses: torch.Tensor,
    prepared: _PreparedScene,
    semantic_score: torch.Tensor,
) -> torch.Tensor:
    starts, ends = prepared.border_starts, prepared.border_ends
    if not starts.numel():
        return semantic_score
    corners = _ego_corners(poses, prepared)
    hit = segment_intersects_quad(
        starts[:, None, :], ends[:, None, :], corners[None, :, :, :]
    ).any()
    return (~hit).to(poses.dtype)


def _driving_direction(
    poses: torch.Tensor, prepared: _PreparedScene, config: Any
) -> torch.Tensor:
    route_indices = prepared.gpu_scene.map_tensors.on_route_lane_indices
    if not route_indices.numel():
        return torch.ones((), device=poses.device, dtype=poses.dtype)
    route_hits = points_in_polygons_torch(
        poses[:, :2],
        prepared.gpu_scene.map_tensors.edge_starts[route_indices],
        prepared.gpu_scene.map_tensors.edge_ends[route_indices],
    ).any(dim=0)
    step = torch.zeros(poses.shape[0], device=poses.device, dtype=poses.dtype)
    if poses.shape[0] > 1:
        step[1:] = torch.linalg.vector_norm(torch.diff(poses[:, :2], dim=0), dim=-1)
    displacement = torch.where(~route_hits, step, torch.zeros_like(step))
    cumulative = torch.cat(
        (torch.zeros(1, device=poses.device, dtype=poses.dtype),
         torch.cumsum(displacement, dim=0))
    )
    horizon = int(1.0 / float(config.interval_s))
    indices = torch.arange(poses.shape[0], device=poses.device)
    starts = (indices - horizon).clamp(min=0)
    windowed = cumulative[indices + 1] - cumulative[starts]
    worst = windowed.max()
    return torch.where(
        worst < 2.0,
        torch.ones_like(worst),
        torch.where(worst < 6.0, torch.full_like(worst, 0.5), torch.zeros_like(worst)),
    )


def _ego_corners(poses: torch.Tensor, prepared: _PreparedScene) -> torch.Tensor:
    shape = prepared.source.current_frame.ego_status.ego_shape
    headings = poses[:, 2]
    centers = poses[:, :2] + float(shape.rear_axle_to_center) * torch.stack(
        (headings.cos(), headings.sin()), dim=-1
    )
    return oriented_box_corners(
        centers,
        headings,
        torch.full_like(headings, float(shape.length) / 2.0),
        torch.full_like(headings, float(shape.width) / 2.0),
    )


def _segments_intersect(
    left_start: torch.Tensor,
    left_end: torch.Tensor,
    right_start: torch.Tensor,
    right_end: torch.Tensor,
) -> torch.Tensor:
    def orient(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    d1 = orient(right_start, right_end, left_start)
    d2 = orient(right_start, right_end, left_end)
    d3 = orient(left_start, left_end, right_start)
    d4 = orient(left_start, left_end, right_end)
    proper = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))

    def on_segment(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, cross: torch.Tensor):
        return (
            (cross == 0)
            & (c[..., 0] <= torch.maximum(a[..., 0], b[..., 0]))
            & (c[..., 0] >= torch.minimum(a[..., 0], b[..., 0]))
            & (c[..., 1] <= torch.maximum(a[..., 1], b[..., 1]))
            & (c[..., 1] >= torch.minimum(a[..., 1], b[..., 1]))
        )

    return proper | on_segment(right_start, right_end, left_start, d1) | on_segment(
        right_start, right_end, left_end, d2
    ) | on_segment(left_start, left_end, right_start, d3) | on_segment(
        left_start, left_end, right_end, d4
    )


def _lane_keeping(poses: torch.Tensor, prepared: _PreparedScene, config: Any) -> torch.Tensor:
    if not prepared.route_centerlines:
        return torch.ones((), device=poses.device, dtype=poses.dtype)
    points = poses[:, :2]
    distances = torch.full((points.shape[0],), float("inf"), device=points.device, dtype=points.dtype)
    for line in prepared.route_centerlines:
        starts, segments = line[:-1], torch.diff(line, dim=0)
        fraction = (((points[:, None] - starts) * segments).sum(-1) / (segments.square().sum(-1).clamp_min(1e-12))).clamp(0.0, 1.0)
        nearest = starts + fraction[..., None] * segments
        distances = torch.minimum(
            distances,
            torch.linalg.vector_norm(points[:, None] - nearest, dim=-1).amin(dim=-1),
        )
    if prepared.intersection_starts.shape[0]:
        in_intersection = points_in_polygons_torch(
            points, prepared.intersection_starts, prepared.intersection_ends
        ).any(dim=0)
    else:
        in_intersection = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    step = torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    if points.shape[0] > 1:
        step[1:] = torch.linalg.vector_norm(torch.diff(points, dim=0), dim=-1)
        step[0] = step[1]
    interval = float(config.interval_s)
    speed = step / interval
    # The analyzer's window is the trailing ``queue_window_s`` of travel, which
    # is ``window`` inter-sample steps ending here -- so the subtrahend is
    # ``window - 1`` back, not ``window``, which took in one step too many.
    window = max(1, int(round(float(config.lane_keeping_queue_window_s) / interval)))
    cumulative = torch.cat((torch.zeros(1, device=points.device, dtype=points.dtype), torch.cumsum(step, dim=0)))
    indices = torch.arange(points.shape[0], device=points.device, dtype=torch.long)
    progress = cumulative[1:] - cumulative[indices.sub(window - 1).clamp(min=0)]
    queue = (speed <= float(config.lane_keeping_queue_speed_mps)) & (
        progress <= float(config.lane_keeping_queue_progress_m)
    )
    last_queue = torch.cummax(
        torch.where(queue, indices, torch.full_like(indices, -10**9)), dim=0
    ).values
    release = (~queue) & (
        indices - last_queue <= int(round(float(config.lane_keeping_queue_release_s) / interval))
    )
    changing = (
        prepared.lane_change
        if prepared.lane_change is not None
        else torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    )
    violation = (
        (distances > float(config.lane_keeping_deviation_m))
        & ~in_intersection
        & ~changing
        & ~queue
        & ~release
    )
    last_clear = torch.cummax(
        torch.where(violation, torch.full_like(indices, -1), indices), dim=0
    ).values
    run = indices - last_clear
    # ``violation_duration >= max_continuous_violation_time`` measured from the
    # first violating sample, which is one sample more than the horizon's worth.
    failed = ((run - 1) * interval >= float(config.lane_keeping_horizon_s)).any()
    return (~failed).to(poses.dtype)


def _history_comfort(
    states: torch.Tensor, scene: T4Scene, config: Any
) -> torch.Tensor:
    history = torch.as_tensor(
        np.asarray(scene.get_history_poses()[:-1], dtype=np.float64),
        device=states.device,
        dtype=states.dtype,
    ).unsqueeze(0)
    history_states = _states_from_poses(history, float(config.interval_s))
    combined = torch.cat((history_states, states.unsqueeze(0)), dim=1)
    times = torch.arange(combined.shape[1], device=states.device, dtype=states.dtype) * float(config.interval_s)
    return ego_is_comfortable_torch(combined, times).all(dim=-1)[0].to(states.dtype)


def _extended_comfort_gpu(
    previous: torch.Tensor,
    current: torch.Tensor,
    dt: float,
    observation_interval: float,
) -> torch.Tensor:
    shift = int(round((observation_interval if observation_interval > 0.0 else dt) / dt))
    if shift <= 0 or shift >= previous.shape[0]:
        return torch.full((), float("nan"), device=current.device, dtype=current.dtype)
    length = min(current.shape[0], previous.shape[0] - shift)
    if length < 3:
        return torch.full((), float("nan"), device=current.device, dtype=current.dtype)
    previous = previous[shift : shift + length].unsqueeze(0)
    current = current[:length].unsqueeze(0)
    previous_features = _comfort_features(previous, dt)
    current_features = _comfort_features(current, dt)
    thresholds = current.new_tensor((0.7, 0.5, 0.1, 0.1))
    rms = torch.stack(
        [
            (current_features[key] - previous_features[key]).square().mean().sqrt()
            for key in ("a", "j", "yr", "ya")
        ]
    )
    return (rms <= thresholds).all().to(current.dtype)


def _comfort_features(states: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
    acceleration = torch.hypot(
        states[..., StateIndex.ACCELERATION_X], states[..., StateIndex.ACCELERATION_Y]
    )
    filtered_acceleration = _apply_savgol(acceleration, 8, 2)
    heading = _phase_unwrap(states[..., StateIndex.HEADING])
    return {
        "a": filtered_acceleration,
        "j": _apply_savgol(filtered_acceleration, 15, 2, 1, dt),
        "yr": _apply_savgol(heading, 15, 2, 1, dt),
        "ya": _apply_savgol(heading, 15, 3, 2, dt),
    }


__all__ = ["score_navsim_batch"]
