"""Online GPU PDM-Closed reference generation.

The reference denominator used by EP is not a learned target and must not be
filled from the demonstrated endpoint.  This module generates the same 3 x 5
PDM candidate family online, simulates it with the Torch PDM simulator, and
scores it with the GPU metric kernels.  It deliberately has no cache or CPU
scoring fallback.

The T4 bundle is decoded as numpy because that is the dataset boundary.  Once
the window enters this module, candidate generation, simulation, metric
selection, and the EP denominator stay on the requested CUDA device.  The
small CPU scene-packing helper is used only to turn variable-length T4 map and
track records into fixed GPU buffers; it never runs the CPU PDM judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from t4_e2e_devkit.common.dataclasses import T4Scene
from t4_e2e_devkit.evaluation.gpu.collisions import quads_intersect
from t4_e2e_devkit.evaluation.gpu.geometry import TorchPolyline, raw_progress_torch
from t4_e2e_devkit.evaluation.gpu.geometry_ops import (
    SublineGeometry,
    _subline_segments,
    oriented_box_corners,
    polygon_polygon_distance,
    polyline_polygon_buffer_intersects,
    polyline_ring_buffer_intersects,
    quad_ring_distance,
    ring_centroids,
)
from t4_e2e_devkit.evaluation.gpu.oracle import (
    VehicleTensors,
    WindowScene,
    score_simulated_window,
)
from t4_e2e_devkit.evaluation.gpu.scene import (
    extract_track_arrays,
    extract_window_scene_arrays,
    window_scene_from_arrays,
)
from t4_e2e_devkit.evaluation.gpu.simulate import (
    TorchSimulatorConfig,
    simulate_proposals_torch,
)
from t4_e2e_devkit.evaluation.reference.pdm_closed import (
    PDM_FALLBACK_TARGET_VELOCITY_MS,
    PDM_HEADWAY_S,
    PDM_INTERVAL_S,
    PDM_LATERAL_OFFSETS_M,
    PDM_MAP_RADIUS_M,
    PDM_MIN_GAP_M,
    PDM_SPEED_LIMIT_FRACTIONS,
)
from t4_e2e_devkit.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex,
)

PDM_REFERENCE_PROPOSALS = len(PDM_LATERAL_OFFSETS_M) + 1
PDM_REFERENCE_POLICIES = len(PDM_SPEED_LIMIT_FRACTIONS)
PDM_REFERENCE_COUNT = PDM_REFERENCE_PROPOSALS * PDM_REFERENCE_POLICIES
PDM_REFERENCE_POSES = 40
PDM_REFERENCE_TRAJECTORY_POSES = 50
PDM_REFERENCE_OBSERVATION_FRAMES = 50


@dataclass
class OnlinePDMReference:
    """Device-resident PDM-Closed labels for a batch of windows."""

    pdm_progress: torch.Tensor  # [B]
    reference_trajectory: torch.Tensor  # [B, 51, 3]
    selected_proposal: torch.Tensor  # [B]
    proposal_scores: torch.Tensor  # [B, 15]
    reference_nc: torch.Tensor  # [B]
    reference_dac: torch.Tensor  # [B]
    reference_raw_progress: torch.Tensor  # [B], gated progress audit field


@dataclass(frozen=True)
class OnlinePDMReferenceInput:
    """One loader-prepared input for online reference generation.

    ``scene_arrays`` contains the packed map and recorded-track geometry from
    :func:`extract_window_scene_arrays`.  The remaining values are the small
    raw fields needed to generate the PDM-Closed candidate family.  Keeping
    this boundary array-based lets a data-loader worker prepare variable-size
    geometry without constructing a cache or a CPU PDM scorer.
    """

    scene_arrays: Mapping[str, Any]
    current_boxes: Any
    current_labels: Any
    ego_shape: Any
    control: Any
    route_speed: Any
    route_has_speed: Any
    token: str = "scene"


@dataclass
class _PreparedPDMReference:
    """Proposal-independent and simulator inputs for one reference window."""

    token: str
    selected_boxes: np.ndarray
    selected_labels: np.ndarray
    shape: np.ndarray
    route_speed: np.ndarray
    route_has_speed: np.ndarray
    ground_truth_scene: WindowScene
    forecast_scene: WindowScene
    centerline: TorchPolyline
    red_light_rings: torch.Tensor | None
    red_light_ring_lengths: torch.Tensor | None
    proposal_states: torch.Tensor
    idm_progress: torch.Tensor
    idm_velocity: torch.Tensor
    leading_agent: torch.Tensor
    initial_states: torch.Tensor
    vehicle: VehicleTensors
    wheel_base: float


def compute_online_pdm_references(
    scenes: Sequence[T4Scene],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    roadblock_buffer_m: float = 0.0,
) -> OnlinePDMReference:
    """Generate PDM-Closed references for ``scenes`` on ``device``.

    ``device`` must be CUDA.  Refusing a CPU device here is intentional: an
    accidental configuration change must fail at the boundary instead of
    turning a validation epoch into a long serial CPU job.
    """

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError(
            "online PDM-Closed references require CUDA; the CPU implementation "
            "is available only through the explicit offline reference API"
        )
    if not torch.cuda.is_available():
        raise ValueError("online PDM-Closed references require an available CUDA device")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"online PDM reference dtype must be float32 or float64, got {dtype}")
    if not scenes:
        empty = torch.empty(0, device=target, dtype=dtype)
        return OnlinePDMReference(
            pdm_progress=empty,
            reference_trajectory=torch.empty((0, 51, 3), device=target, dtype=dtype),
            selected_proposal=torch.empty(0, device=target, dtype=torch.long),
            proposal_scores=torch.empty((0, PDM_REFERENCE_COUNT), device=target, dtype=dtype),
            reference_nc=empty,
            reference_dac=empty,
            reference_raw_progress=empty,
        )

    inputs = [
        _scene_to_reference_input(scene, roadblock_buffer_m=roadblock_buffer_m)
        for scene in scenes
    ]
    return compute_online_pdm_references_from_arrays(
        inputs,
        device=target,
        dtype=dtype,
        roadblock_buffer_m=roadblock_buffer_m,
    )


def compute_online_pdm_references_from_arrays(
    inputs: Sequence[OnlinePDMReferenceInput | Mapping[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    roadblock_buffer_m: float = 0.0,
) -> OnlinePDMReference:
    """Generate PDM-Closed references from loader-prepared arrays.

    This is the training-facing entry point.  It accepts the packed
    ``oracle_scene`` payload emitted by the T4 reader and never calls the
    reference cache or the CPU PDM implementation.  Map packing and token
    association may have happened in a loader worker; candidate generation,
    rollout, metric selection, and the EP denominator all run on CUDA here.
    """

    target = torch.device(device)
    _validate_online_device(target, dtype)
    if not inputs:
        empty = torch.empty(0, device=target, dtype=dtype)
        return OnlinePDMReference(
            pdm_progress=empty,
            reference_trajectory=torch.empty((0, 51, 3), device=target, dtype=dtype),
            selected_proposal=torch.empty(0, device=target, dtype=torch.long),
            proposal_scores=torch.empty((0, PDM_REFERENCE_COUNT), device=target, dtype=dtype),
            reference_nc=empty,
            reference_dac=empty,
            reference_raw_progress=empty,
        )

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        normalized = [
            value
            if isinstance(value, OnlinePDMReferenceInput)
            else OnlinePDMReferenceInput(**value)
            for value in inputs
        ]
        prepared = [
            _prepare_one_arrays(
                value,
                device=target,
                dtype=dtype,
                roadblock_buffer_m=roadblock_buffer_m,
            )
            for value in normalized
        ]
        simulated = _simulate_prepared_batch(
            prepared,
        )
        results = [
            _finish_prepared_reference(value, rollout, dtype=dtype)
            for value, rollout in zip(prepared, simulated, strict=True)
        ]

    return OnlinePDMReference(
        pdm_progress=torch.stack([value.pdm_progress for value in results]),
        reference_trajectory=torch.stack([value.reference_trajectory for value in results]),
        selected_proposal=torch.stack([value.selected_proposal for value in results]),
        proposal_scores=torch.stack([value.proposal_scores for value in results]),
        reference_nc=torch.stack([value.reference_nc for value in results]),
        reference_dac=torch.stack([value.reference_dac for value in results]),
        reference_raw_progress=torch.stack([value.reference_raw_progress for value in results]),
    )


def _compute_one(
    scene: OnlinePDMReferenceInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
    roadblock_buffer_m: float,
) -> OnlinePDMReference:
    return _compute_one_arrays(
        scene,
        device=device,
        dtype=dtype,
        roadblock_buffer_m=roadblock_buffer_m,
    )


def _prepare_one_arrays(
    scene: OnlinePDMReferenceInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
    roadblock_buffer_m: float,
) -> _PreparedPDMReference:
    """Prepare one window without running the simulator or metric judge."""

    current_boxes_np = _to_numpy(scene.current_boxes, dtype=np.float64).reshape(-1, 9)
    current_labels_np = _to_numpy(scene.current_labels, dtype=np.int64).reshape(-1)
    if current_boxes_np.shape[0] != current_labels_np.shape[0]:
        raise ValueError(f"scene {scene.token} has mismatched current box and label counts")
    shape = _to_numpy(scene.ego_shape, dtype=np.float64).reshape(-1)
    if shape.shape != (3,):
        raise ValueError(f"scene {scene.token} ego_shape must have shape [3]")
    control = _to_numpy(scene.control, dtype=np.float64).reshape(-1)
    if control.shape != (6,):
        raise ValueError(f"scene {scene.token} control state must have shape [6]")

    ground_truth_tracks, ground_truth_map, centerline = window_scene_from_arrays(
        dict(scene.scene_arrays), device, dtype
    )
    ground_truth_scene = WindowScene(
        tracks=ground_truth_tracks,
        map_tensors=ground_truth_map,
        centerline=centerline,
        pdm_progress=torch.zeros((), device=device, dtype=dtype),
    )

    forecast_indices = _forecast_object_indices(
        current_boxes_np,
        current_labels_np,
        shape,
        device=device,
        dtype=dtype,
    )
    selected_boxes = current_boxes_np[forecast_indices]
    selected_labels = current_labels_np[forecast_indices]
    forecast_boxes = _forecast_boxes(
        selected_boxes,
        selected_labels,
        PDM_REFERENCE_OBSERVATION_FRAMES,
    )
    forecast_labels = [selected_labels.copy() for _ in forecast_boxes]
    forecast_track_arrays = extract_track_arrays(forecast_boxes, forecast_labels)
    forecast_scene_arrays = dict(scene.scene_arrays)
    forecast_scene_arrays.update(forecast_track_arrays)
    forecast_tracks, _, _ = window_scene_from_arrays(
        forecast_scene_arrays, device, dtype
    )
    forecast_scene = WindowScene(
        tracks=forecast_tracks,
        map_tensors=ground_truth_map,
        centerline=centerline,
        pdm_progress=torch.zeros((), device=device, dtype=dtype),
    )

    route_speed = _to_numpy(scene.route_speed, dtype=np.float32)
    route_has_speed = _to_numpy(scene.route_has_speed, dtype=bool)
    red_light_rings = _optional_scene_tensor(
        scene.scene_arrays.get("red_light_rings"), device, dtype
    )
    red_light_ring_lengths = _optional_scene_tensor(
        scene.scene_arrays.get("red_light_ring_lengths"), device, torch.long
    )
    proposal_states, idm_progress, idm_velocity, leading_agent = _generate_proposals(
        centerline,
        torch.as_tensor(selected_boxes, device=device, dtype=dtype),
        torch.as_tensor(selected_labels, device=device, dtype=torch.long),
        control,
        shape,
        route_speed,
        route_has_speed,
        red_light_rings=red_light_rings,
        red_light_ring_lengths=red_light_ring_lengths,
        device=device,
        dtype=dtype,
        num_poses=PDM_REFERENCE_POSES,
    )
    initial_state = _initial_state(control, device=device, dtype=dtype)
    initial_states = initial_state.expand(PDM_REFERENCE_COUNT, -1).contiguous()
    vehicle = VehicleTensors(
        half_length=float(shape[1]) / 2.0,
        half_width=float(shape[2]) / 2.0,
        rear_axle_to_center=float(shape[0]) / 2.0,
    )
    return _PreparedPDMReference(
        token=scene.token,
        selected_boxes=selected_boxes,
        selected_labels=selected_labels,
        shape=shape,
        route_speed=route_speed,
        route_has_speed=route_has_speed,
        ground_truth_scene=ground_truth_scene,
        forecast_scene=forecast_scene,
        centerline=centerline,
        red_light_rings=red_light_rings,
        red_light_ring_lengths=red_light_ring_lengths,
        proposal_states=proposal_states,
        idm_progress=idm_progress,
        idm_velocity=idm_velocity,
        leading_agent=leading_agent,
        initial_states=initial_states,
        vehicle=vehicle,
        wheel_base=float(shape[0]),
    )


def _simulate_prepared_batch(
    prepared: Sequence[_PreparedPDMReference],
) -> list[torch.Tensor]:
    """Simulate all prepared windows in grouped GPU batches.

    Proposal generation and metric geometry remain window-local.  The
    simulator is independent across windows, so concatenating its inputs
    removes one Python/kernel-launch sequence per window without changing any
    simulator expression or metric definition.  Wheel-base groups preserve
    the exact per-window simulator configuration.
    """

    outputs: dict[int, torch.Tensor] = {}
    groups: dict[float, list[int]] = {}
    for index, value in enumerate(prepared):
        groups.setdefault(value.wheel_base, []).append(index)
    for wheel_base, indices in groups.items():
        proposal_states = torch.cat(
            [prepared[index].proposal_states for index in indices], dim=0
        )
        initial_states = torch.cat(
            [prepared[index].initial_states for index in indices], dim=0
        )
        simulated = simulate_proposals_torch(
            proposal_states,
            initial_states,
            TorchSimulatorConfig(
                wheel_base=wheel_base,
                discretization_time=PDM_INTERVAL_S,
            ),
            compile_rollout=False,
        )
        offset = 0
        for index in indices:
            count = prepared[index].proposal_states.shape[0]
            outputs[index] = simulated[offset : offset + count]
            offset += count
    return [outputs[index] for index in range(len(prepared))]


def _finish_prepared_reference(
    prepared: _PreparedPDMReference,
    simulated: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> OnlinePDMReference:
    """Select and extend one already-simulated reference window."""

    shape = prepared.shape
    proposal_states = prepared.proposal_states
    centerline = prepared.centerline
    red_light_rings = prepared.red_light_rings
    red_light_ring_lengths = prepared.red_light_ring_lengths
    proposal_totals, _ = score_simulated_window(
        simulated,
        prepared.forecast_scene,
        prepared.vehicle,
        interval_length=PDM_INTERVAL_S,
        normalize_progress_across_proposals=True,
    )
    selected = torch.argmax(proposal_totals)

    proposal_states_full = torch.zeros(
        (
            PDM_REFERENCE_COUNT,
            PDM_REFERENCE_TRAJECTORY_POSES + 1,
            proposal_states.shape[-1],
        ),
        device=simulated.device,
        dtype=dtype,
    )
    proposal_states_full[:, : PDM_REFERENCE_POSES + 1] = proposal_states
    selected_boxes = torch.as_tensor(
        prepared.selected_boxes, device=simulated.device, dtype=dtype
    )
    selected_labels = torch.as_tensor(
        prepared.selected_labels, device=simulated.device, dtype=torch.long
    )
    object_positions = _forecast_object_positions(
        selected_boxes,
        selected_labels,
        PDM_REFERENCE_TRAJECTORY_POSES,
        dtype=dtype,
    )
    selected_path_index = selected // PDM_REFERENCE_POLICIES
    selected_path = [centerline]
    selected_path.extend(
        centerline.parallel(offset) for offset in PDM_LATERAL_OFFSETS_M
    )
    selected_path = selected_path[selected_path_index]
    target_velocities = (
        torch.as_tensor(PDM_SPEED_LIMIT_FRACTIONS, device=simulated.device, dtype=dtype)
        .repeat(PDM_REFERENCE_PROPOSALS)
        * _route_speed(
            torch.as_tensor(prepared.route_speed, device=simulated.device),
            torch.as_tensor(prepared.route_has_speed, device=simulated.device),
            device=simulated.device,
            dtype=dtype,
        )
    ).clamp(min=1.0e-3)
    selected_corridor_start = selected_path.project(
        torch.zeros((2,), device=simulated.device, dtype=dtype)
    )
    selected_corridor_end = (
        selected_corridor_start
        + target_velocities.max() * PDM_REFERENCE_TRAJECTORY_POSES * PDM_INTERVAL_S
    )
    selected_corridor = _subline_segments(
        selected_path, selected_corridor_start, selected_corridor_end
    )
    selected_red_progress = None
    selected_red_in_corridor = None
    if red_light_rings is not None and red_light_ring_lengths is not None and red_light_rings.shape[0]:
        red_centroids = ring_centroids(red_light_rings, red_light_ring_lengths)
        selected_red_progress = selected_path.project(red_centroids)
        selected_red_in_corridor = polyline_ring_buffer_intersects(
            selected_path,
            red_light_rings,
            red_light_ring_lengths,
            float(shape[2]) / 2.0,
            selected_corridor_start,
            selected_corridor_end,
            subline=selected_corridor,
        )
    proposal_states_full[selected] = _continue_selected_proposal(
        proposal_states_full[selected],
        selected_path,
        prepared.idm_progress[selected],
        prepared.idm_velocity[selected],
        prepared.leading_agent[selected],
        target_velocities[selected],
        object_positions,
        selected_boxes[:, 7:9],
        selected_boxes[:, 3].clamp(min=1.0e-3) / 2.0,
        selected_boxes[:, 4].clamp(min=1.0e-3) / 2.0,
        selected_boxes[:, 6],
        torch.isin(
            selected_labels,
            torch.tensor((0, 1, 2, 3, 4), device=simulated.device),
        ),
        float(shape[2]) / 2.0,
        float(shape[1]) / 2.0,
        float(shape[1]) / 2.0,
        selected_corridor_end,
        float(shape[0]) / 2.0,
        red_light_rings,
        red_light_ring_lengths,
        selected_corridor,
        selected_red_progress,
        selected_red_in_corridor,
    )

    selected_simulated = simulated[selected]
    _, selected_components = score_simulated_window(
        selected_simulated[None],
        prepared.ground_truth_scene,
        prepared.vehicle,
        interval_length=PDM_INTERVAL_S,
    )
    raw_progress = raw_progress_torch(
        selected_simulated[None, :, :2]
        + _rear_axle_to_center(selected_simulated, prepared.vehicle)[None],
        prepared.ground_truth_scene.centerline,
    )[0]
    multiplicative = selected_components[0, 0] * selected_components[0, 1]
    pdm_progress = raw_progress * multiplicative

    values = (pdm_progress, raw_progress, selected_components[0, 0], selected_components[0, 1])
    if not all(bool(torch.isfinite(value).item()) for value in values) or any(
        bool((value < 0.0).item()) for value in values
    ):
        raise ValueError(f"GPU PDM-Closed produced invalid metrics for {prepared.token}")

    return OnlinePDMReference(
        pdm_progress=pdm_progress,
        reference_trajectory=proposal_states_full[selected, :, :3],
        selected_proposal=selected,
        proposal_scores=proposal_totals,
        reference_nc=selected_components[0, 0],
        reference_dac=selected_components[0, 1],
        reference_raw_progress=pdm_progress,
    )


def _compute_one_arrays(
    scene: OnlinePDMReferenceInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
    roadblock_buffer_m: float,
) -> OnlinePDMReference:
    """Run the shared preparation, batched simulator, and finish stages."""

    prepared = _prepare_one_arrays(
        scene,
        device=device,
        dtype=dtype,
        roadblock_buffer_m=roadblock_buffer_m,
    )
    simulated = _simulate_prepared_batch(
        [prepared],
    )[0]
    return _finish_prepared_reference(prepared, simulated, dtype=dtype)


def _scene_to_reference_input(
    scene: T4Scene, *, roadblock_buffer_m: float
) -> OnlinePDMReferenceInput:
    """Convert the public scene object to the array API without scoring."""

    if scene.future_annotations is None or len(scene.future_annotations) < PDM_REFERENCE_OBSERVATION_FRAMES + 1:
        raise ValueError(
            f"scene {scene.scene_metadata.token} needs current plus "
            f"{PDM_REFERENCE_OBSERVATION_FRAMES} future annotation frames"
        )
    frame = scene.current_frame
    if frame.map_tensors is None or frame.annotations is None:
        raise ValueError(f"scene {scene.scene_metadata.token} lacks map or current annotations")

    annotations = scene.future_annotations[: PDM_REFERENCE_OBSERVATION_FRAMES + 1]
    gt_boxes = [np.asarray(item.boxes, dtype=np.float64).reshape(-1, 9) for item in annotations]
    gt_labels = [np.asarray(item.labels, dtype=np.int64).reshape(-1) for item in annotations]
    lanes = np.asarray(frame.map_tensors.lanes, dtype=np.float32)
    route = np.asarray(frame.map_tensors.route_lanes, dtype=np.float32)
    route_speed = np.asarray(frame.map_tensors.route_lanes_speed_limit, dtype=np.float32)
    route_has_speed = np.asarray(frame.map_tensors.route_lanes_has_speed_limit, dtype=bool)
    polygons = np.asarray(frame.map_tensors.polygons, dtype=np.float32)

    # This packs variable-length map/track records once. It does not invoke the
    # CPU PDM reference or the CPU proposal scorer.
    ground_truth_arrays = extract_window_scene_arrays(
        lanes,
        route,
        polygons,
        gt_boxes,
        gt_labels,
        roadblock_buffer_m=float(roadblock_buffer_m),
    )
    current_boxes = gt_boxes[0]
    current_labels = gt_labels[0]
    control = _control_state(frame.ego_status.control_state)
    return OnlinePDMReferenceInput(
        scene_arrays=ground_truth_arrays,
        current_boxes=current_boxes,
        current_labels=current_labels,
        ego_shape=frame.ego_status.ego_shape.as_array(),
        control=control,
        route_speed=route_speed,
        route_has_speed=route_has_speed,
        token=scene.scene_metadata.token,
    )


def _validate_online_device(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cuda":
        raise ValueError(
            "online PDM-Closed references require CUDA; the CPU implementation "
            "is available only through the explicit offline reference API"
        )
    if not torch.cuda.is_available():
        raise ValueError("online PDM-Closed references require an available CUDA device")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"online PDM reference dtype must be float32 or float64, got {dtype}")


def _to_numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _control_state(state: Any) -> np.ndarray:
    state = state or {}
    velocity = np.asarray(state.get("velocity", (0.0, 0.0)), dtype=np.float64).reshape(-1)
    acceleration = np.asarray(state.get("acceleration", (0.0, 0.0)), dtype=np.float64).reshape(-1)
    if velocity.size != 2 or acceleration.size != 2:
        raise ValueError("T4 control state must contain two velocity and acceleration values")
    return np.array(
        [
            velocity[0],
            velocity[1],
            acceleration[0],
            acceleration[1],
            float(state.get("steering", 0.0)),
            float(state.get("yaw_rate", 0.0)),
        ],
        dtype=np.float64,
    )


def _initial_state(control: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    state = torch.zeros((StateIndex.size(),), device=device, dtype=dtype)
    values = torch.as_tensor(control, device=device, dtype=dtype)
    state[StateIndex.VELOCITY_X : StateIndex.VELOCITY_Y + 1] = values[:2]
    state[StateIndex.ACCELERATION_X : StateIndex.ACCELERATION_Y + 1] = values[2:4]
    state[StateIndex.STEERING_ANGLE] = values[4]
    state[StateIndex.ANGULAR_VELOCITY] = values[5]
    return state


def _rear_axle_to_center(states: torch.Tensor, vehicle: VehicleTensors) -> torch.Tensor:
    heading = states[..., StateIndex.HEADING]
    return vehicle.rear_axle_to_center * torch.stack(
        (torch.cos(heading), torch.sin(heading)), dim=-1
    )


def _forecast_boxes(
    current: np.ndarray, labels: np.ndarray, frames: int
) -> list[np.ndarray]:
    current = np.asarray(current, dtype=np.float64).reshape(-1, 9)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(current) != len(labels):
        raise ValueError("T4 current boxes and labels have different lengths")
    dynamic = np.isin(labels, np.asarray((0, 1, 2, 3, 4), dtype=np.int64))
    velocity = np.zeros((len(current), 2), dtype=np.float64)
    if np.any(dynamic):
        speed = np.linalg.norm(current[dynamic, 7:9], axis=-1)
        velocity_heading = np.arctan2(current[dynamic, 8], current[dynamic, 7])
        heading_delta = np.arctan2(
            np.sin(current[dynamic, 6] - velocity_heading),
            np.cos(current[dynamic, 6] - velocity_heading),
        )
        track_heading = np.where(
            np.abs(heading_delta) < (np.pi / 2.0),
            current[dynamic, 6],
            current[dynamic, 6] + np.pi,
        )
        velocity[dynamic] = speed[:, None] * np.stack(
            (np.cos(track_heading), np.sin(track_heading)), axis=-1
        )
    result = []
    for index in range(frames + 1):
        values = current.copy()
        if len(values):
            values[dynamic, :2] += float(index) * PDM_INTERVAL_S * velocity[dynamic]
        result.append(values)
    return result


def _forecast_object_indices(
    current_boxes: np.ndarray,
    current_labels: np.ndarray,
    shape: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Select the same current objects that ``PDMObjectManager`` forecasts.

    The CPU observation first removes objects outside its map radius or already
    intersecting the ego footprint, then keeps the nearest per-type caps.  The
    selection is geometry-only preprocessing; distance and intersection use
    the shared CUDA quad kernel, while the small index result is returned to
    the forecast packer.
    """

    boxes = torch.as_tensor(current_boxes, device=device, dtype=dtype).reshape(-1, 9)
    labels = torch.as_tensor(current_labels, device=device, dtype=torch.long).reshape(-1)
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    rear_axle_to_center = float(shape[0]) / 2.0
    ego_center = torch.tensor(
        [rear_axle_to_center, 0.0], device=device, dtype=dtype
    )
    object_centers = boxes[:, :2]
    distance = torch.linalg.vector_norm(object_centers - ego_center, dim=-1)
    ego_corners = oriented_box_corners(
        ego_center[None],
        torch.zeros((1,), device=device, dtype=dtype),
        torch.full((1,), float(shape[1]) / 2.0, device=device, dtype=dtype),
        torch.full((1,), float(shape[2]) / 2.0, device=device, dtype=dtype),
    )
    object_corners = oriented_box_corners(
        object_centers,
        boxes[:, 6],
        boxes[:, 4].clamp(min=1.0e-3) / 2.0,
        boxes[:, 3].clamp(min=1.0e-3) / 2.0,
    )
    valid = (distance <= float(PDM_MAP_RADIUS_M)) & ~quads_intersect(
        ego_corners, object_corners
    )

    # TrackedObjects/PDMObjectManager expose static objects first and then
    # dynamic VEHICLE, PEDESTRIAN, BICYCLE groups, each nearest-first.
    agent_labels = torch.tensor((0, 1, 2, 3, 4), device=device)
    static_mask = ~torch.isin(labels, agent_labels)
    masks = (
        static_mask,
        torch.isin(labels, torch.tensor((0, 1, 2), device=device)),
        labels == 4,
        labels == 3,
    )
    selected: list[torch.Tensor] = []
    for mask, limit in zip(masks, (50, 50, 25, 10), strict=True):
        candidates = torch.where(valid & mask)[0]
        if candidates.numel():
            order = torch.argsort(distance[candidates], stable=True)[:limit]
            selected.append(candidates[order])
    if not selected:
        return np.empty((0,), dtype=np.int64)
    return torch.cat(selected).detach().cpu().numpy().astype(np.int64, copy=False)


def _route_polyline(route: torch.Tensor) -> TorchPolyline:
    values = route.reshape(-1, route.shape[-1])
    valid = (values[:, :8].abs().sum(dim=-1) > 0.0) & torch.isfinite(values[:, :8]).all(dim=-1)
    values = values[valid]
    if values.shape[0] < 2:
        raise ValueError("T4 route has fewer than two usable points")
    keep = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    keep[1:] = torch.linalg.vector_norm(torch.diff(values[:, :2], dim=0), dim=-1) > 1.0e-6
    values = values[keep]
    if values.shape[0] < 2:
        raise ValueError("T4 route collapses to fewer than two distinct points")
    heading = torch.atan2(values[:, 3], values[:, 2])
    if heading.shape[0] > 1:
        delta = torch.atan2(torch.sin(torch.diff(heading)), torch.cos(torch.diff(heading)))
        heading = torch.cat((heading[:1], heading[:1] + torch.cumsum(delta, dim=0)), dim=0)
    return TorchPolyline(values[:, :2], heading)


def _route_speed(
    route_speed: torch.Tensor,
    route_has_speed: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    speed = torch.as_tensor(route_speed, device=device, dtype=dtype).reshape(-1)
    has_speed = torch.as_tensor(route_has_speed, device=device, dtype=torch.bool).reshape(-1)
    if speed.numel() == 0 or has_speed.numel() == 0:
        return torch.as_tensor(PDM_FALLBACK_TARGET_VELOCITY_MS, device=device, dtype=dtype)
    valid = has_speed & torch.isfinite(speed) & (speed > 1.0e-3)
    first_valid = torch.where(valid, speed, torch.zeros_like(speed))[0]
    return torch.where(
        first_valid > 0.0,
        first_valid,
        torch.as_tensor(PDM_FALLBACK_TARGET_VELOCITY_MS, device=device, dtype=dtype),
    )


def _generate_proposals(
    centerline: TorchPolyline,
    current_boxes: torch.Tensor,
    current_labels: torch.Tensor,
    control: np.ndarray,
    shape: np.ndarray,
    route_speed: np.ndarray,
    route_has_speed: np.ndarray,
    *,
    red_light_rings: torch.Tensor | None = None,
    red_light_ring_lengths: torch.Tensor | None = None,
    device: torch.device,
    dtype: torch.dtype,
    num_poses: int = PDM_REFERENCE_POSES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = [centerline]
    paths.extend(centerline.parallel(offset) for offset in PDM_LATERAL_OFFSETS_M)
    path_index = torch.arange(PDM_REFERENCE_PROPOSALS, device=device).repeat_interleave(
        PDM_REFERENCE_POLICIES
    )
    policy_index = torch.arange(PDM_REFERENCE_POLICIES, device=device).repeat(
        PDM_REFERENCE_PROPOSALS
    )
    path_progress = torch.stack([path.project(torch.zeros((2,), device=device, dtype=dtype)) for path in paths])
    progress = path_progress[path_index]
    velocity = torch.full_like(progress, float(control[0]))
    states = torch.zeros(
        (PDM_REFERENCE_COUNT, num_poses + 1, StateIndex.size()),
        device=device,
        dtype=dtype,
    )
    for lateral_index, path in enumerate(paths):
        rows = torch.where(path_index == lateral_index)[0]
        states[rows, 0, :3] = path.interpolate(progress[rows])
    lead = torch.zeros((PDM_REFERENCE_COUNT, 3), device=device, dtype=dtype)
    target_velocity = (
        torch.as_tensor(PDM_SPEED_LIMIT_FRACTIONS, device=device, dtype=dtype)[policy_index]
        * _route_speed(
            torch.as_tensor(route_speed, device=device),
            torch.as_tensor(route_has_speed, device=device),
            device=device,
            dtype=dtype,
        )
    ).clamp(min=1.0e-3)
    object_positions = _forecast_object_positions(
        current_boxes, current_labels, num_poses, dtype=dtype
    )
    object_half_width = current_boxes[:, 3].clamp(min=1.0e-3) / 2.0
    object_half_length = current_boxes[:, 4].clamp(min=1.0e-3) / 2.0
    object_headings = current_boxes[:, 6]
    object_is_agent = torch.isin(
        current_labels.reshape(-1),
        torch.tensor((0, 1, 2, 3, 4), device=current_boxes.device),
    )
    corridor_end = path_progress + target_velocity.max() * PDM_REFERENCE_TRAJECTORY_POSES * PDM_INTERVAL_S
    ego_half_width = float(shape[2]) / 2.0
    corridors = [
        _subline_segments(path, path_progress[index], corridor_end[index])
        for index, path in enumerate(paths)
    ]
    red_progresses = red_corridors = None
    if red_light_rings is not None and red_light_ring_lengths is not None and red_light_rings.shape[0]:
        red_centroids = ring_centroids(red_light_rings, red_light_ring_lengths)
        red_progresses = [path.project(red_centroids) for path in paths]
        red_corridors = [
            polyline_ring_buffer_intersects(
                path,
                red_light_rings,
                red_light_ring_lengths,
                ego_half_width,
                path_progress[index],
                corridor_end[index],
                subline=corridors[index],
            )
            for index, path in enumerate(paths)
        ]

    for time_index in range(1, num_poses + 1):
        if time_index % 2 == 0:
            for lateral_index, path in enumerate(paths):
                rows = torch.where(path_index == lateral_index)[0]
                path_lead = _find_leading_agent(
                    path,
                    progress[rows],
                    object_positions[time_index],
                    current_boxes[:, 7:9],
                    object_half_width,
                    object_half_length,
                    object_headings,
                    object_is_agent,
                    ego_half_width,
                    float(shape[1]) / 2.0,
                    float(shape[1]) / 2.0,
                    corridor_end[lateral_index],
                    float(shape[0]) / 2.0,
                    red_light_rings,
                    red_light_ring_lengths,
                    corridors[lateral_index],
                    None if red_progresses is None else red_progresses[lateral_index],
                    None if red_corridors is None else red_corridors[lateral_index],
                )
                lead[rows] = path_lead
        next_progress, next_velocity = _propagate_idm(
            progress,
            velocity,
            lead,
            target_velocity,
            dtype=dtype,
        )
        progress, velocity = next_progress, next_velocity
        for lateral_index, path in enumerate(paths):
            rows = torch.where(path_index == lateral_index)[0]
            states[rows, time_index, :3] = path.interpolate(progress[rows])
    return states, progress, velocity, lead


def _continue_selected_proposal(
    states: torch.Tensor,
    path: TorchPolyline,
    progress: torch.Tensor,
    velocity: torch.Tensor,
    lead: torch.Tensor,
    target_velocity: torch.Tensor,
    object_positions: torch.Tensor,
    object_velocities: torch.Tensor,
    object_half_width: torch.Tensor,
    object_half_length: torch.Tensor,
    object_headings: torch.Tensor,
    object_is_agent: torch.Tensor,
    ego_half_width: float,
    ego_half_length: float,
    ego_length_rear: float,
    corridor_end: torch.Tensor,
    rear_axle_to_center: float,
    red_light_rings: torch.Tensor | None = None,
    red_light_ring_lengths: torch.Tensor | None = None,
    subline: SublineGeometry | None = None,
    red_progress: torch.Tensor | None = None,
    red_in_corridor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extend the already generated selected proposal using CPU PDM ordering."""

    progress = progress.reshape(1)
    velocity = velocity.reshape(1)
    lead = lead.reshape(1, 3)
    for time_index in range(PDM_REFERENCE_POSES + 1, PDM_REFERENCE_TRAJECTORY_POSES + 1):
        if time_index % 2 == 0:
            lead = _find_leading_agent(
                path,
                progress,
                object_positions[time_index],
                object_velocities,
                object_half_width,
                object_half_length,
                object_headings,
                object_is_agent,
                ego_half_width,
                ego_half_length,
                ego_length_rear,
                corridor_end,
                rear_axle_to_center,
                red_light_rings,
                red_light_ring_lengths,
                subline,
                red_progress,
                red_in_corridor,
            )
        progress, velocity = _propagate_idm(
            progress,
            velocity,
            lead,
            target_velocity.reshape(1),
            dtype=states.dtype,
        )
        states[time_index, :3] = path.interpolate(progress)[0]
    return states


def _forecast_object_positions(
    current_boxes: torch.Tensor,
    current_labels: torch.Tensor,
    frames: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if current_boxes.shape[0] == 0:
        return torch.empty((frames + 1, 0, 2), device=current_boxes.device, dtype=dtype)
    times = torch.arange(frames + 1, device=current_boxes.device, dtype=dtype)[:, None, None]
    dynamic = torch.isin(
        current_labels.reshape(-1),
        torch.tensor((0, 1, 2, 3, 4), device=current_boxes.device),
    )
    speed = torch.linalg.vector_norm(current_boxes[:, 7:9], dim=-1)
    velocity_heading = torch.atan2(current_boxes[:, 8], current_boxes[:, 7])
    heading_delta = torch.atan2(
        torch.sin(current_boxes[:, 6] - velocity_heading),
        torch.cos(current_boxes[:, 6] - velocity_heading),
    )
    track_heading = torch.where(
        heading_delta.abs() < (np.pi / 2.0),
        current_boxes[:, 6],
        current_boxes[:, 6] + np.pi,
    )
    velocity = torch.where(
        dynamic[:, None],
        speed[:, None]
        * torch.stack((torch.cos(track_heading), torch.sin(track_heading)), dim=-1),
        torch.zeros_like(current_boxes[:, 7:9]),
    )
    return current_boxes[None, :, :2] + times * PDM_INTERVAL_S * velocity[None]


def _find_leading_agent(
    path: TorchPolyline,
    progress: torch.Tensor,
    object_positions: torch.Tensor,
    object_velocities: torch.Tensor,
    object_half_width: torch.Tensor,
    object_half_length: torch.Tensor,
    object_headings: torch.Tensor,
    object_is_agent: torch.Tensor,
    ego_half_width: float,
    ego_half_length: float,
    ego_length_rear: float,
    corridor_end: torch.Tensor,
    rear_axle_to_center: float,
    red_light_rings: torch.Tensor | None = None,
    red_light_ring_lengths: torch.Tensor | None = None,
    subline: SublineGeometry | None = None,
    red_progress: torch.Tensor | None = None,
    red_in_corridor: torch.Tensor | None = None,
) -> torch.Tensor:
    if object_positions.shape[0] == 0:
        return torch.tensor(
            [path.length, 0.0, ego_length_rear], device=progress.device, dtype=progress.dtype
        ).expand(progress.shape[0], -1)
    object_progress = path.project(object_positions)
    object_corners = oriented_box_corners(
        object_positions,
        object_headings,
        object_half_length,
        object_half_width,
    )
    ego_state = path.interpolate(progress)
    ego_corners = oriented_box_corners(
        ego_state[:, :2] + rear_axle_to_center * torch.stack(
            (torch.cos(ego_state[:, 2]), torch.sin(ego_state[:, 2])), dim=-1
        ),
        ego_state[:, 2],
        torch.full_like(progress, float(ego_half_length)),
        torch.full_like(progress, float(ego_half_width)),
    )
    relative = polygon_polygon_distance(ego_corners, object_corners)
    path_start = path.project(torch.zeros((2,), device=progress.device, dtype=progress.dtype))
    object_in_corridor = polyline_polygon_buffer_intersects(
        path,
        object_corners,
        float(ego_half_width),
        path_start,
        corridor_end,
        subline=subline,
    )
    candidate_progress = progress[:, None] + relative
    valid = (object_progress[None] > progress[:, None]) & object_in_corridor[None]
    candidate_progress = torch.where(valid, candidate_progress, torch.full_like(candidate_progress, float("inf")))
    best_progress, best_index = candidate_progress.min(dim=-1)
    has_agent = torch.isfinite(best_progress)
    best_index = best_index.clamp(max=max(object_positions.shape[0] - 1, 0))
    path_heading = ego_state[:, 2]
    object_speed = torch.linalg.vector_norm(object_velocities, dim=-1)
    lead_velocity_values = (
        object_speed[None, :]
        * torch.cos(object_headings[None, :] - path_heading[:, None])
        * object_is_agent[None, :].to(progress.dtype)
    )
    best_velocity = lead_velocity_values.gather(1, best_index[:, None]).squeeze(1)

    if (
        red_light_rings is not None
        and red_light_ring_lengths is not None
        and red_light_rings.shape[0] > 0
    ):
        if red_progress is None:
            red_progress = path.project(ring_centroids(red_light_rings, red_light_ring_lengths))
        if red_in_corridor is None:
            red_in_corridor = polyline_ring_buffer_intersects(
                path,
                red_light_rings,
                red_light_ring_lengths,
                float(ego_half_width),
                path_start,
                corridor_end,
                subline=subline,
            )
        red_relative = torch.stack(
            [
                quad_ring_distance(ego_corners, ring[: int(length_value)])
                for ring, length_value in zip(
                    red_light_rings,
                    red_light_ring_lengths.detach().cpu().tolist(),
                    strict=True,
                )
            ],
            dim=-1,
        )
        red_valid = (red_progress[None] > progress[:, None]) & red_in_corridor[None]
        red_candidate = torch.where(
            red_valid,
            progress[:, None] + red_relative,
            torch.full_like(red_relative, float("inf")),
        )
        red_best, _ = red_candidate.min(dim=-1)
        red_exists = torch.isfinite(red_best)
        use_red = red_exists & ((~has_agent) | (red_best < best_progress))
        best_progress = torch.where(use_red, red_best, best_progress)
        best_velocity = torch.where(use_red, torch.zeros_like(best_velocity), best_velocity)
        has_agent = has_agent | red_exists

    # PDMGenerator updates one mutable ``leading_agent_array`` for every
    # longitudinal policy sharing a path.  Its free-driving branch writes
    # progress and rear length but intentionally leaves velocity untouched;
    # consequently a later policy can inherit the previous policy's lead
    # velocity.  Preserve that exact stateful ordering on the device.  This
    # is observable in proposal selection, so a stateless vectorized default
    # would not be CPU-equivalent.
    output = torch.zeros((progress.shape[0], 3), device=progress.device, dtype=progress.dtype)
    carry_velocity = torch.zeros((), device=progress.device, dtype=progress.dtype)
    carry_rear = torch.zeros((), device=progress.device, dtype=progress.dtype)
    for index in range(progress.shape[0]):
        output[index, 0] = torch.where(has_agent[index], best_progress[index], path.length)
        output[index, 1] = torch.where(has_agent[index], best_velocity[index], carry_velocity)
        output[index, 2] = torch.where(
            has_agent[index], carry_rear, torch.as_tensor(ego_length_rear, device=progress.device, dtype=progress.dtype)
        )
        carry_velocity = output[index, 1]
        carry_rear = output[index, 2]
    return output


def _optional_scene_tensor(
    value: Any, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
    return tensor.to(device=device, dtype=dtype)


def _propagate_idm(
    progress: torch.Tensor,
    velocity: torch.Tensor,
    lead: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dtype
    lead_progress, lead_velocity, lead_length_rear = lead.unbind(dim=-1)
    s_star = (
        PDM_MIN_GAP_M
        + velocity * PDM_HEADWAY_S
        + velocity * (velocity - lead_velocity) / (2.0 * np.sqrt(1.5 * 3.0))
    )
    s_alpha = (lead_progress - progress - lead_length_rear).clamp(min=PDM_MIN_GAP_M)
    acceleration = 1.5 * (
        1.0
        - (velocity / target_velocity).pow(10)
        - (s_star / s_alpha).pow(2)
    )
    acceleration = acceleration.clamp(min=-3.0, max=1.5)
    return progress + PDM_INTERVAL_S * velocity, velocity + PDM_INTERVAL_S * acceleration


__all__ = [
    "OnlinePDMReference",
    "OnlinePDMReferenceInput",
    "compute_online_pdm_references",
    "compute_online_pdm_references_from_arrays",
]
