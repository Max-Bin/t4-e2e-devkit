"""PDM scoring entry point.

The backend is explicit:

``gpu``
    Batched device implementation for large runs.

``cpu``
    Reference judge over shapely geometry. Slower, but useful for audits.

``tier4``
    An optional T4 metric family. It is computed independently and is never
    folded into PDM-Score.

The aggregate is::

    PDMS = NC * DAC * (5*EP + 5*TTC + 2*Comfort + 0*DDC) / 12

with NC and DAC as multiplicative gates, so a collision or a drivable-area
violation zeroes the score no matter how well everything else scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from t4_e2e_devkit.common.constants import (
    DEFAULT_PDM_WEIGHTS,
    PDM_COMPONENT_ORDER,
    SCORER_FUTURE_FRAMES,
    TRAJECTORY_INTERVAL,
    TRAJECTORY_POSES,
)
from t4_e2e_devkit.common.dataclasses import PDMResults, T4Scene, Trajectory
from t4_e2e_devkit.evaluation import oracle_evaluator as _oracle
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

#: Frames of recorded future the PDM observation buffer consumes: 5 s at 10 Hz
#: plus the current frame.  Longer than the 4 s scoring horizon on purpose --
#: TTC projects ahead of the last scored step and would run off the end of a
#: window sized to the horizon alone.
PDM_OBSERVATION_FRAMES = _oracle.PDM_OBSERVATION_NUM_POSES

BACKENDS = ("gpu", "cpu")


class ScoringError(RuntimeError):
    """Scoring cannot proceed with the data it was given."""


@dataclass
class T4PDMScorerConfig:
    """Settings shared by every backend.

    ``num_poses`` and ``trajectory_interval`` describe raw proposal tensors
    passed to :meth:`T4PDMScorer.score_proposals`.  A :class:`Trajectory`
    passed to :meth:`T4PDMScorer.score_batch` carries its own sampling.
    """

    num_poses: int = TRAJECTORY_POSES
    trajectory_interval: float = TRAJECTORY_INTERVAL
    t4_oracle_num_poses: int = SCORER_FUTURE_FRAMES
    t4_pdm_observation_frames: int = PDM_OBSERVATION_FRAMES
    t4_oracle_device: str = "cpu"
    t4_drivable_area_buffer_m: float = _oracle.PDM_ROADBLOCK_BUFFER_M
    weights: Sequence[float] = DEFAULT_PDM_WEIGHTS

    @property
    def trajectory_sampling(self) -> TrajectorySampling:
        """Sampling of raw proposal tensors without per-trajectory metadata."""

        return TrajectorySampling(
            num_poses=self.num_poses,
            interval_length=self.trajectory_interval,
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        :param key: config field name.
        :param default: value when the field is absent.
        :return: the field value.
        """
        return getattr(self, key, default)


class T4PDMScorer:
    """Scores planned trajectories against recorded T4 windows."""

    def __init__(
        self,
        backend: str = "gpu",
        device: Optional[str] = None,
        config: Optional[T4PDMScorerConfig] = None,
        include_tier4_metrics: bool = False,
    ) -> None:
        """
        :param backend: ``"gpu"`` or ``"cpu"``; see the module docstring.
        :param device: CUDA device for the GPU backend.
        :param config: scoring settings.
        :param include_tier4_metrics: attach a compatibility copy of the
            independently computed T4 family to each result.
        :raises ValueError: on an unknown backend, or a GPU request without CUDA
            -- silently falling back to the CPU judge would turn a training step
            into a minutes-long stall rather than an error.
        """
        if backend not in BACKENDS:
            raise ValueError(f"unknown scoring backend {backend!r}; expected one of {BACKENDS}")
        if backend == "gpu" and not torch.cuda.is_available():
            raise ValueError(
                "backend='gpu' needs CUDA. The CPU judge is the audit implementation "
                "that is orders of magnitude slower, so it is never substituted silently; "
                "pass backend='cpu' to ask for it deliberately."
            )

        self.backend = backend
        self.device = torch.device(device or ("cuda" if backend == "gpu" else "cpu"))
        if backend == "gpu" and self.device.type != "cuda":
            raise ValueError(
                "backend='gpu' requires a CUDA device; the CPU judge is never "
                "substituted silently"
            )
        self.config = config or T4PDMScorerConfig()
        self.include_tier4_metrics = include_tier4_metrics

        self._evaluation_sampling = TrajectorySampling(
            num_poses=_oracle.MODEL_NUM_POSES,
            interval_length=_oracle.MODEL_DT,
        )

        evaluator_config = T4PDMScorerConfig(**{**self.config.__dict__})
        evaluator_config.num_poses = self._evaluation_sampling.num_poses
        evaluator_config.trajectory_interval = self._evaluation_sampling.interval_length
        evaluator_config.t4_oracle_device = str(self.device) if backend == "gpu" else "cpu"
        self._evaluator = _oracle.T4OracleEvaluator(config=evaluator_config)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(self, trajectory: Trajectory, scene: T4Scene) -> PDMResults:
        """
        Score one trajectory against one window.
        :param trajectory: the planned trajectory, in ego-frame local coordinates.
        :param scene: the recorded window it is scored against.
        :return: components and aggregate for this window.
        """
        return self.score_batch([trajectory], [scene])[0]

    def score_batch(
        self,
        trajectories: Sequence[Trajectory],
        scenes: Sequence[T4Scene],
    ) -> List[PDMResults]:
        """
        Score a batch of trajectories against their windows.
        :param trajectories: one planned trajectory per window.
        :param scenes: the recorded windows.
        :return: one result per window, in input order.
        :raises ScoringError: when the inputs disagree in length, or a window
            lacks the recorded future or the PDM-Closed progress denominator.
        """
        if len(trajectories) != len(scenes):
            raise ScoringError(
                f"got {len(trajectories)} trajectories for {len(scenes)} scenes"
            )
        if not scenes:
            return []

        proposals = self._stack_proposals(trajectories)
        progress_override = self._online_progress(scenes)
        targets = self._build_targets(scenes, progress_override=progress_override)
        features = self._build_features(scenes)

        result = self._evaluator.score(proposals, targets, features)
        components = result.components[:, 0].detach().cpu().numpy()  # [B, 6]

        results: List[PDMResults] = []
        for index, scene in enumerate(scenes):
            tier4 = (
                self.compute_tier4_metrics(trajectories[index], scene)
                if self.include_tier4_metrics
                else {}
            )
            results.append(
                PDMResults.from_components(
                    components[index],
                    token=scene.scene_metadata.token,
                    weights=self.config.weights,
                    tier4_metrics=tier4,
                )
            )
        return results

    def score_proposals(
        self,
        proposals: torch.Tensor,
        scenes: Sequence[T4Scene],
        trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> torch.Tensor:
        """Score many proposals per window, for training-time reporting.

        This is the training-time path: it keeps the tensor result rather than
        building :class:`PDMResults`, so the components stay on the device and
        The returned components are detached metric labels; this method is not a
        differentiable training objective.

        :param proposals: ``[B, N, num_poses, 3]``.
        :param scenes: one window per batch element.
        :param trajectory_sampling: sampling of ``proposals``.  When omitted,
            the sampling in :class:`T4PDMScorerConfig` is used.
        :return: ``[B, N, 6]`` components, in :data:`PDM_COMPONENT_ORDER`.
        """
        proposals = self._resample_proposals(
            proposals,
            trajectory_sampling or self.config.trajectory_sampling,
        )
        if proposals.shape[0] != len(scenes):
            raise ScoringError(
                f"got {proposals.shape[0]} proposal batches for {len(scenes)} scenes"
            )
        progress_override = self._online_progress(scenes)
        result = self._evaluator.score(
            proposals,
            self._build_targets(scenes, progress_override=progress_override),
            self._build_features(scenes),
        )
        return result.components

    def compute_tier4_metrics(self, trajectory: Trajectory, scene: T4Scene) -> Dict[str, float]:
        """TIER IV's metric family for one window.

        Reported next to the PDM score, never inside it.  These terms answer
        questions PDM-Score does not ask -- did the plan run a red light, is it
        kinematically executable, does it agree with the previous frame's plan
        -- and folding them into the aggregate would make the number
        incomparable with published PDM scores.

        :param trajectory: the planned trajectory.
        :param scene: the window it is scored against.
        :return: named metric values; empty when the extras are unavailable.
        """
        from t4_e2e_devkit.evaluation.tier4_metrics import compute_tier4_metrics

        return compute_tier4_metrics(trajectory, scene)

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _stack_proposals(self, trajectories: Sequence[Trajectory]) -> torch.Tensor:
        poses = []
        for index, trajectory in enumerate(trajectories):
            try:
                resampled = trajectory.resample(self._evaluation_sampling)
            except ValueError as error:
                raise ScoringError(
                    f"trajectory {index} cannot be adapted to the evaluation grid "
                    f"({self._evaluation_sampling.time_horizon:g}s at "
                    f"{self._evaluation_sampling.interval_length:g}s): {error}"
                ) from error
            poses.append(np.asarray(resampled.poses, dtype=np.float64))
        # [B, 1, num_poses, 3]: one proposal per window.
        return torch.from_numpy(np.stack(poses, axis=0)).unsqueeze(1)

    def _resample_proposals(
        self,
        proposals: torch.Tensor,
        source_sampling: TrajectorySampling,
    ) -> torch.Tensor:
        """Adapt raw proposal tensors to the scorer's internal time grid."""

        if (
            not torch.is_tensor(proposals)
            or proposals.ndim != 4
            or proposals.shape[-1] != 3
            or not proposals.is_floating_point()
        ):
            raise ScoringError(
                "proposals must be a floating-point tensor with shape [batch, count, poses, 3], "
                f"got {getattr(proposals, 'shape', None)}"
            )
        if proposals.shape[-2] != source_sampling.num_poses:
            raise ScoringError(
                f"proposal tensor has {proposals.shape[-2]} poses, but its sampling "
                f"declares {source_sampling.num_poses}"
            )
        target = self._evaluation_sampling
        if source_sampling == target:
            return proposals

        source_horizon = float(source_sampling.time_horizon)
        target_horizon = float(target.time_horizon)
        if target_horizon > source_horizon + 1e-9:
            raise ScoringError(
                "proposal horizon is shorter than the evaluation horizon: "
                f"source={source_horizon:g}s, target={target_horizon:g}s"
            )

        source = torch.cat(
            (torch.zeros_like(proposals[..., :1, :]), proposals), dim=-2
        )
        source_interval = float(source_sampling.interval_length)
        target_times = (
            torch.arange(
                1,
                target.num_poses + 1,
                device=proposals.device,
                dtype=proposals.dtype,
            )
            * target.interval_length
            / source_interval
        )
        lower = torch.floor(target_times).long().clamp(max=source.shape[-2] - 1)
        upper = torch.ceil(target_times).long().clamp(max=source.shape[-2] - 1)
        alpha = (target_times - lower.to(target_times.dtype)).reshape(1, 1, -1, 1)

        def interpolate(values: torch.Tensor) -> torch.Tensor:
            lower_values = torch.index_select(values, -2, lower)
            upper_values = torch.index_select(values, -2, upper)
            return lower_values + alpha * (upper_values - lower_values)

        xy = interpolate(source[..., :, :2])
        headings = source[..., 2]
        heading_delta = torch.atan2(
            torch.sin(torch.diff(headings, dim=-1)),
            torch.cos(torch.diff(headings, dim=-1)),
        )
        unwrapped = torch.cat(
            (headings[..., :1], headings[..., :1] + torch.cumsum(heading_delta, dim=-1)),
            dim=-1,
        )
        heading = interpolate(unwrapped.unsqueeze(-1))
        return torch.cat((xy, heading), dim=-1)

    def _online_progress(self, scenes: Sequence[T4Scene]) -> torch.Tensor | None:
        """Resolve missing EP denominators through the requested GPU path."""

        missing = [index for index, scene in enumerate(scenes) if scene.pdm_progress is None]
        if not missing:
            return None
        if self.backend != "gpu":
            return None

        from t4_e2e_devkit.evaluation.gpu.reference import compute_online_pdm_references

        online = compute_online_pdm_references(
            [scenes[index] for index in missing],
            device=self.device,
            dtype=torch.float64,
            roadblock_buffer_m=float(self.config.t4_drivable_area_buffer_m),
        )
        values: list[torch.Tensor] = []
        cursor = 0
        for scene in scenes:
            if scene.pdm_progress is None:
                values.append(online.pdm_progress[cursor])
                cursor += 1
            else:
                values.append(
                    torch.tensor(
                        float(scene.pdm_progress), device=self.device, dtype=torch.float32
                    )
                )
        return torch.stack(values)

    def _build_targets(
        self,
        scenes: Sequence[T4Scene],
        *,
        progress_override: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        horizon = self.config.t4_oracle_num_poses
        observation = self.config.t4_pdm_observation_frames

        oracle_future, progress = [], []
        current_boxes, current_labels = [], []
        future_boxes, future_labels = [], []

        for scene in scenes:
            token = scene.scene_metadata.token
            if scene.future_ego_poses is None or scene.future_annotations is None:
                raise ScoringError(
                    f"scene {token} carries no recorded future; scoring needs it, and its "
                    "absence is never interpreted as an empty traffic scene"
                )
            if scene.future_ego_poses.shape[0] < horizon:
                raise ScoringError(
                    f"scene {token} has {scene.future_ego_poses.shape[0]} future poses, "
                    f"fewer than the {horizon}-frame scoring horizon"
                )
            if len(scene.future_annotations) < observation + 1:
                raise ScoringError(
                    f"scene {token} has {len(scene.future_annotations)} future annotation "
                    f"frames, fewer than the {observation + 1} the PDM observation buffer "
                    "needs. The observation window is longer than the scoring horizon "
                    "because TTC projects past the last scored step."
                )
            if scene.pdm_progress is None and progress_override is None:
                raise ScoringError(
                    f"scene {token} has no pdm_progress. Build a PDM-Closed reference "
                    "through the GPU online reference path or use an explicit "
                    "offline reference; the demonstrated future endpoint is not a "
                    "substitute, because it changes what ego-progress measures."
                )

            oracle_future.append(np.asarray(scene.future_ego_poses[:horizon], dtype=np.float64))
            if progress_override is None:
                progress.append(float(scene.pdm_progress))
            else:
                progress.append(progress_override[len(progress)])

            current = scene.current_frame.annotations
            if current is None:
                raise ScoringError(f"scene {token} has no annotations at its current frame")
            current_boxes.append(np.asarray(current.boxes, dtype=np.float64).reshape(-1, 9))
            current_labels.append(np.asarray(current.labels, dtype=np.int64).reshape(-1))

            # future_annotations[0] is the current frame; the observation buffer
            # wants the frames after it.
            window = scene.future_annotations[1 : observation + 1]
            future_boxes.append(
                [np.asarray(a.boxes, dtype=np.float64).reshape(-1, 9) for a in window]
            )
            future_labels.append(
                [np.asarray(a.labels, dtype=np.int64).reshape(-1) for a in window]
            )

        if progress and torch.is_tensor(progress[0]):
            progress_tensor = torch.stack([value.reshape(()) for value in progress])
        else:
            progress_tensor = torch.tensor(progress, dtype=torch.float64)
        return {
            "oracle_future_trajectory": torch.from_numpy(np.stack(oracle_future, axis=0)),
            "pdm_progress": progress_tensor,
            "current_agent_boxes": current_boxes,
            "current_agent_labels": current_labels,
            "future_agent_boxes": future_boxes,
            "future_agent_labels": future_labels,
            "agent_gt_available": [True] * len(scenes),
        }

    def _build_features(self, scenes: Sequence[T4Scene]) -> Dict[str, Any]:
        ego_shapes, control_states = [], []
        lanes, routes, polygons = [], [], []
        route_speeds, route_has_speeds = [], []

        for scene in scenes:
            frame = scene.current_frame
            status = frame.ego_status
            ego_shapes.append(status.ego_shape.as_array())

            state = status.control_state or {}
            velocity = np.asarray(state.get("velocity", (0.0, 0.0)), dtype=np.float64).reshape(-1)
            acceleration = np.asarray(
                state.get("acceleration", (0.0, 0.0)), dtype=np.float64
            ).reshape(-1)
            control_states.append(
                np.array(
                    [
                        velocity[0], velocity[1],
                        acceleration[0], acceleration[1],
                        float(state.get("steering", 0.0)),
                        float(state.get("yaw_rate", 0.0)),
                    ],
                    dtype=np.float64,
                )
            )

            if frame.map_tensors is None:
                raise ScoringError(
                    f"scene {scene.scene_metadata.token} has no map tensors; the drivable "
                    "area, the route corridor and the progress centerline all come from "
                    "them, so scoring without a map is not a lenient case"
                )
            lanes.append(np.asarray(frame.map_tensors.lanes, dtype=np.float32))
            routes.append(np.asarray(frame.map_tensors.route_lanes, dtype=np.float32))
            polygons.append(np.asarray(frame.map_tensors.polygons, dtype=np.float32))
            route_speeds.append(
                np.asarray(frame.map_tensors.route_lanes_speed_limit, dtype=np.float32)
            )
            route_has_speeds.append(
                np.asarray(frame.map_tensors.route_lanes_has_speed_limit, dtype=bool)
            )

        return {
            "ego_shape": torch.from_numpy(np.stack(ego_shapes, axis=0).astype(np.float32)),
            "control_state": torch.from_numpy(np.stack(control_states, axis=0).astype(np.float32)),
            "lanes": torch.from_numpy(np.stack(lanes, axis=0)),
            "route_lanes": torch.from_numpy(np.stack(routes, axis=0)),
            "polygons": torch.from_numpy(np.stack(polygons, axis=0)),
            "route_lanes_speed_limit": torch.from_numpy(
                np.stack(route_speeds, axis=0)
            ),
            "route_lanes_has_speed_limit": torch.from_numpy(
                np.stack(route_has_speeds, axis=0)
            ),
        }


def compare_backends(
    trajectories: Sequence[Trajectory],
    scenes: Sequence[T4Scene],
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Score the same windows on both backends and report the disagreement.

    This is the check that keeps the fast path honest.  The GPU scorer exists to
    run inside a training step; the CPU judge is what defines the right answer.
    Running both on real windows and reporting the worst per-component
    deviation is how a regression in the fast path is caught -- silently
    trusting it is how a training run optimises the wrong objective for a week.

    :param trajectories: planned trajectories.
    :param scenes: the windows they are scored against.
    :param tolerance: maximum acceptable absolute deviation per component.
    :return: per-component maxima, the aggregate maximum, and whether the run
        stayed within tolerance.
    """
    gpu_results = T4PDMScorer(backend="gpu").score_batch(trajectories, scenes)
    cpu_results = T4PDMScorer(backend="cpu").score_batch(trajectories, scenes)

    report: Dict[str, Any] = {"num_scenes": len(scenes), "tolerance": tolerance}
    worst = 0.0
    for name in PDM_COMPONENT_ORDER:
        deltas = [
            abs(gpu.components[name] - cpu.components[name])
            for gpu, cpu in zip(gpu_results, cpu_results, strict=True)
        ]
        value = max(deltas) if deltas else 0.0
        report[f"max_abs_delta/{name}"] = value
        worst = max(worst, value)

    score_deltas = [
        abs(gpu.score - cpu.score)
        for gpu, cpu in zip(gpu_results, cpu_results, strict=True)
    ]
    report["max_abs_delta/score"] = max(score_deltas) if score_deltas else 0.0
    worst = max(worst, report["max_abs_delta/score"])
    report["worst"] = worst
    report["within_tolerance"] = worst <= tolerance
    return report
