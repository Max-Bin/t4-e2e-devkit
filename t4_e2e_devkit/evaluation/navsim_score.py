"""T4 adapter for the NavSim v1 PDMS and v2 EPDMS formulas.

The numeric formulas live in :mod:`evaluation.reference.pdms_navsim`.  This
module owns the T4 boundary: trajectory sampling, scene-local map tensors,
recorded future boxes, vehicle dimensions and consecutive-plan state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, isfinite
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from t4_e2e_devkit.common.constants import SCORER_FUTURE_FRAMES, T4_INTERVAL_LENGTH
from t4_e2e_devkit.common.dataclasses import T4Scene, Trajectory
from t4_e2e_devkit.evaluation.reference import pdms_navsim as formulas
from t4_e2e_devkit.planning.simulation.pdm_sim.simulator import simulate_proposals
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

NAVSIM_VERSIONS = ("v1", "v2")
NAVSIM_COMPONENT_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
)
NAVSIM_V1_METRICS = NAVSIM_COMPONENT_METRICS + ("score",)
NAVSIM_V2_METRICS = NAVSIM_COMPONENT_METRICS + ("extended_comfort", "score")
NAVSIM_METRICS = NAVSIM_V2_METRICS

_V1_SCORE_DEPENDENCIES = frozenset(
    {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "history_comfort",
    }
)
_V2_SCORE_DEPENDENCIES = frozenset(NAVSIM_COMPONENT_METRICS) | {"extended_comfort"}


def resolve_navsim_metric_names(
    version: str,
    metric_names: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Validate and order the metrics exposed by one scorer call.

    ``None`` selects every metric available in the requested version.  An
    explicit selection preserves caller order; this order is also used by
    proposal tensors returned from :meth:`T4NavSimScorer.score_proposals`.
    Dependencies needed to calculate ``score`` are added internally and are
    not added to the returned selection.
    """

    normalized_version = str(version).lower().removeprefix("navsim-")
    if normalized_version not in NAVSIM_VERSIONS:
        raise ValueError(
            f"version must be one of {NAVSIM_VERSIONS}, got {version!r}"
        )
    available = (
        NAVSIM_V1_METRICS if normalized_version == "v1" else NAVSIM_V2_METRICS
    )
    if metric_names is None:
        return available
    if isinstance(metric_names, str):
        metric_names = (metric_names,)
    names = tuple(str(name) for name in metric_names)
    if not names:
        raise ValueError("metric_names must contain at least one metric")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    unknown = sorted(set(names) - set(available))
    if duplicates:
        raise ValueError(f"metric_names contains duplicates: {duplicates}")
    if unknown:
        raise ValueError(
            f"unknown metrics for version {normalized_version!r}: {unknown}; "
            f"available metrics are {list(available)}"
        )
    return names


def required_navsim_metric_names(
    version: str,
    metric_names: Sequence[str],
) -> frozenset[str]:
    """Return the metrics that must be computed for an exposed selection."""

    selected = set(resolve_navsim_metric_names(version, metric_names))
    if "score" in selected:
        selected.update(
            _V1_SCORE_DEPENDENCIES
            if str(version).lower().removeprefix("navsim-") == "v1"
            else _V2_SCORE_DEPENDENCIES
        )
    return frozenset(selected)


class NavSimScoringError(ValueError):
    """A NavSim score cannot be computed from the supplied T4 window."""


@dataclass(frozen=True)
class T4NavSimScorerConfig:
    """Configuration shared by the v1 and v2 PDM metric versions."""

    version: str = "v2"
    metric_names: Optional[Sequence[str]] = None
    backend: str = "auto"
    device: Optional[str] = None
    gpu_dtype: str = "float32"
    horizon_s: float = SCORER_FUTURE_FRAMES * T4_INTERVAL_LENGTH
    interval_s: float = T4_INTERVAL_LENGTH
    observation_interval_s: float = 0.5
    human_penalty_filter: Optional[bool] = None
    require_extended_comfort: bool = False
    progress_weight: float = formulas.PROGRESS_WEIGHT
    ttc_weight: float = formulas.TTC_WEIGHT
    lane_keeping_weight: float = formulas.LANE_KEEPING_WEIGHT
    history_comfort_weight: float = formulas.HISTORY_COMFORT_WEIGHT
    extended_comfort_weight: float = formulas.EXTENDED_COMFORT_WEIGHT
    lane_keeping_deviation_m: float = 0.5
    lane_keeping_horizon_s: float = 2.0
    use_simulator: bool = True
    progress_distance_threshold: float = formulas.PROGRESS_DISTANCE_THRESHOLD

    def __post_init__(self) -> None:
        version = str(self.version).lower()
        if version not in NAVSIM_VERSIONS:
            raise ValueError(f"version must be one of {NAVSIM_VERSIONS}, got {self.version!r}")
        object.__setattr__(self, "version", version)
        if self.metric_names is not None:
            object.__setattr__(
                self,
                "metric_names",
                resolve_navsim_metric_names(version, self.metric_names),
            )
        backend = str(self.backend).lower()
        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend must be auto, cpu or gpu")
        object.__setattr__(self, "backend", backend)
        dtype = str(self.gpu_dtype).lower()
        if dtype not in {"float32", "float64"}:
            raise ValueError("gpu_dtype must be float32 or float64")
        object.__setattr__(self, "gpu_dtype", dtype)
        for name in ("horizon_s", "interval_s", "observation_interval_s"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        steps = self.horizon_s / self.interval_s
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1e-8):
            raise ValueError("horizon_s must be an integer number of interval_s samples")
        if self.version == "v1" and self.require_extended_comfort:
            raise ValueError("require_extended_comfort is only valid for version='v2'")
        for name in (
            "progress_weight",
            "ttc_weight",
            "lane_keeping_weight",
            "history_comfort_weight",
            "extended_comfort_weight",
            "lane_keeping_deviation_m",
            "lane_keeping_horizon_s",
            "progress_distance_threshold",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def num_steps(self) -> int:
        return int(round(self.horizon_s / self.interval_s))

    @property
    def use_human_filter(self) -> bool:
        if self.human_penalty_filter is not None:
            return bool(self.human_penalty_filter)
        return self.version == "v2"

    @property
    def selected_metric_names(self) -> tuple[str, ...]:
        """Metrics exposed by default calls using this configuration."""

        return resolve_navsim_metric_names(self.version, self.metric_names)


@dataclass(frozen=True)
class T4NavSimResult:
    """One T4 window's selected NavSim metrics.

    Only requested metrics are stored in :attr:`values`.  An unavailable
    extended-comfort value is omitted because it needs a previous plan; it is
    never encoded as a fake zero. Diagnostic coverage belongs in ``metadata``.
    """

    version: str
    metrics: Mapping[str, float]
    token: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        version = str(self.version).lower().removeprefix("navsim-")
        names = tuple(str(name) for name in self.metrics)
        resolve_navsim_metric_names(version)
        if names:
            resolve_navsim_metric_names(version, names)
        values = {name: float(self.metrics[name]) for name in names}
        if not all(isfinite(value) for value in values.values()):
            raise ValueError("NavSim metric values must be finite")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "metrics", values)

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Names in the exact order selected for this result."""

        return tuple(self.metrics)

    @property
    def values(self) -> dict[str, float]:
        return dict(self.metrics)

    def value(self, name: str) -> float:
        """Return one selected metric, with a clear error for unselected data."""

        try:
            return float(self.metrics[str(name)])
        except KeyError as error:
            raise NavSimScoringError(
                f"metric {name!r} was not requested for this result; "
                f"available metrics are {list(self.metric_names)}"
            ) from error

    @property
    def score(self) -> float:
        """Return the aggregate score when it was explicitly requested."""

        return self.value("score")

    @property
    def extended_comfort(self) -> Optional[float]:
        """Return selected extended comfort, or ``None`` when unavailable."""

        value = self.metrics.get("extended_comfort")
        return None if value is None else float(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "token": self.token,
            "values": self.values,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class T4NavSimProposalResult:
    """Proposal metrics plus the column names for the returned tensor."""

    values: Any
    metric_names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.metric_names)
        if not names:
            raise ValueError("metric_names must contain at least one metric")
        if len(set(names)) != len(names):
            raise ValueError("metric_names must not contain duplicates")
        if getattr(self.values, "ndim", None) != 3:
            raise ValueError("proposal metric values must have shape [batch, proposals, metrics]")
        if self.values.shape[-1] != len(names):
            raise ValueError(
                "proposal metric values width does not match metric_names: "
                f"{self.values.shape[-1]} != {len(names)}"
            )
        object.__setattr__(self, "metric_names", names)

    @property
    def shape(self):
        """Shape of the ``[batch, proposals, metrics]`` tensor."""

        return self.values.shape


@dataclass(frozen=True)
class NavSimFollowup:
    """A scored follow-up scene used by pseudo-closed-loop aggregation."""

    result: T4NavSimResult
    start_xy: tuple[float, float]


class T4NavSimScorer:
    """Score a T4 trajectory with NavSim v1 or v2 semantics."""

    def __init__(self, config: Optional[T4NavSimScorerConfig] = None) -> None:
        self.config = config or T4NavSimScorerConfig()
        if self.config.backend == "auto":
            import torch

            self.config = replace(
                self.config,
                backend="gpu" if torch.cuda.is_available() else "cpu",
            )
        if self.config.backend == "gpu":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("backend='gpu' needs CUDA; pass backend='cpu' for the audit path")
            self.device = torch.device(self.config.device or "cuda")
            if self.device.type != "cuda":
                raise ValueError("backend='gpu' requires a CUDA device")
        else:
            self.device = None

    def _resolve_metric_names(
        self,
        metric_names: Optional[Sequence[str]],
    ) -> tuple[str, ...]:
        return resolve_navsim_metric_names(
            self.config.version,
            self.config.metric_names if metric_names is None else metric_names,
        )

    def score(
        self,
        trajectory: Trajectory,
        scene: T4Scene,
        *,
        metric_names: Optional[Sequence[str]] = None,
        previous_trajectory: Optional[Trajectory] = None,
        previous_scene: Optional[T4Scene] = None,
        human_trajectory: Optional[Trajectory] = None,
        previous_human_trajectory: Optional[Trajectory] = None,
    ) -> T4NavSimResult:
        if self.config.backend == "gpu":
            return self.score_batch(
                [trajectory],
                [scene],
                metric_names=metric_names,
                previous_trajectories=[previous_trajectory],
                previous_scenes=[previous_scene],
                human_trajectories=[human_trajectory],
                previous_human_trajectories=[previous_human_trajectory],
            )[0]
        return self._score_cpu(
            trajectory,
            scene,
            metric_names=self._resolve_metric_names(metric_names),
            previous_trajectory=previous_trajectory,
            previous_scene=previous_scene,
            human_trajectory=human_trajectory,
            previous_human_trajectory=previous_human_trajectory,
        )

    def score_batch(
        self,
        trajectories: Sequence[Trajectory],
        scenes: Sequence[T4Scene],
        *,
        metric_names: Optional[Sequence[str]] = None,
        previous_trajectories: Optional[Sequence[Trajectory | None]] = None,
        previous_scenes: Optional[Sequence[T4Scene | None]] = None,
        human_trajectories: Optional[Sequence[Trajectory | None]] = None,
        previous_human_trajectories: Optional[Sequence[Trajectory | None]] = None,
    ) -> tuple[T4NavSimResult, ...]:
        """Score a batch using one simulator/metric pipeline.

        The CPU backend keeps the scalar reference behavior.  The GPU backend
        batches trajectory simulation and tensor geometry across all windows;
        variable-size map and track tensors are prepared once per window and
        remain on the device during scoring.
        """

        if len(trajectories) != len(scenes) or not scenes:
            raise NavSimScoringError("trajectories and scenes must have the same non-zero length")
        selected_metrics = self._resolve_metric_names(metric_names)
        batch_size = len(scenes)
        previous_trajectories = (
            [None] * batch_size if previous_trajectories is None else list(previous_trajectories)
        )
        previous_scenes = (
            [None] * batch_size if previous_scenes is None else list(previous_scenes)
        )
        human_trajectories = (
            [None] * batch_size if human_trajectories is None else list(human_trajectories)
        )
        previous_human_trajectories = (
            [None] * batch_size
            if previous_human_trajectories is None
            else list(previous_human_trajectories)
        )
        if any(
            len(values) != batch_size
            for values in (
                previous_trajectories,
                previous_scenes,
                human_trajectories,
                previous_human_trajectories,
            )
        ):
            raise NavSimScoringError("all optional trajectory batches must match scenes")
        if self.config.backend == "cpu":
            return tuple(
                self._score_cpu(
                    trajectory,
                    scene,
                    metric_names=selected_metrics,
                    previous_trajectory=previous_trajectory,
                    previous_scene=previous_scene,
                    human_trajectory=human_trajectory,
                    previous_human_trajectory=previous_human_trajectory,
                )
                for trajectory, scene, previous_trajectory, previous_scene,
                human_trajectory, previous_human_trajectory in zip(
                    trajectories,
                    scenes,
                    previous_trajectories,
                    previous_scenes,
                    human_trajectories,
                    previous_human_trajectories,
                    strict=True,
                )
            )

        from t4_e2e_devkit.evaluation.gpu.navsim import score_navsim_batch

        outputs = score_navsim_batch(
            trajectories,
            scenes,
            replace(self.config, metric_names=selected_metrics),
            previous_trajectories=previous_trajectories,
            previous_scenes=previous_scenes,
            human_trajectories=human_trajectories,
        )
        results: list[T4NavSimResult] = []
        for scene, output in zip(scenes, outputs, strict=True):
            current = dict(output["components"])
            human = output.get("human_components")
            if human is not None:
                current = _apply_human_filter(current, human)
            extended = output.get("extended_comfort")
            if extended is not None and not isfinite(float(extended)):
                extended = None
            required_metrics = required_navsim_metric_names(
                self.config.version, selected_metrics
            )
            if (
                "extended_comfort" in required_metrics
                and extended is None
                and self.config.require_extended_comfort
            ):
                raise NavSimScoringError("v2 EPDMS cannot aggregate without extended comfort")
            values = {
                key: float(current[key])
                for key in required_metrics
                if key in NAVSIM_COMPONENT_METRICS and key in current
            }
            score = (
                self._aggregate(values, None if extended is None else float(extended))
                if "score" in selected_metrics
                else None
            )
            metadata = dict(output.get("metadata", {}))
            metadata["human_filter_applied"] = float(human is not None)
            metadata["extended_comfort_available"] = float(extended is not None)
            if score is not None:
                metadata["weighted_metric_weight"] = float(self._available_weight(extended))
            result_values = {
                name: (
                    float(score)
                    if name == "score"
                    else float(extended)
                    if name == "extended_comfort" and extended is not None
                    else float(current[name])
                )
                for name in selected_metrics
                if name == "score"
                or name in current
                or (name == "extended_comfort" and extended is not None)
            }
            results.append(
                T4NavSimResult(
                    version=self.config.version,
                    metrics=result_values,
                    token=scene.scene_metadata.token,
                    metadata=metadata,
                )
            )
        return tuple(results)

    def _score_cpu(
        self,
        trajectory: Trajectory,
        scene: T4Scene,
        *,
        metric_names: Sequence[str],
        previous_trajectory: Optional[Trajectory] = None,
        previous_scene: Optional[T4Scene] = None,
        human_trajectory: Optional[Trajectory] = None,
        previous_human_trajectory: Optional[Trajectory] = None,
    ) -> T4NavSimResult:
        """Score one window.

        For v2, pass the previous plan and scene when available.  The first
        window of a sequence legitimately has no EC; set
        ``require_extended_comfort`` to reject that incomplete score.
        """

        if (previous_trajectory is None) != (previous_scene is None):
            raise NavSimScoringError(
                "previous_trajectory and previous_scene must be provided together"
            )
        if previous_human_trajectory is not None and previous_trajectory is None:
            raise NavSimScoringError(
                "previous_human_trajectory requires a previous prediction pair"
            )
        required_metrics = required_navsim_metric_names(
            self.config.version, metric_names
        )
        current = self._score_components(
            trajectory,
            scene,
            metric_names=required_metrics,
        )
        extended = None
        needs_extended = (
            self.config.version == "v2" and "extended_comfort" in required_metrics
        )
        if needs_extended and previous_trajectory is not None:
            extended = self._extended_comfort(
                trajectory,
                scene,
                previous_trajectory,
                previous_scene,
            )
        elif needs_extended and self.config.require_extended_comfort:
            raise NavSimScoringError(
                "v2 EPDMS requires a previous consecutive plan for extended comfort"
            )

        human = None
        if (
            self.config.version == "v2"
            and self.config.use_human_filter
            and required_metrics.intersection(NAVSIM_COMPONENT_METRICS)
        ):
            reference = human_trajectory or self._human_trajectory(scene)
            human = self._score_components(
                reference,
                scene,
                metric_names=required_metrics,
            )
            if (
                "extended_comfort" in required_metrics
                and previous_human_trajectory is not None
                and previous_scene is not None
            ):
                human["extended_comfort"] = self._extended_comfort(
                    reference,
                    scene,
                    previous_human_trajectory,
                    previous_scene,
                )
            current = _apply_human_filter(current, human)

        values = {
            key: float(current[key])
            for key in required_metrics
            if key in NAVSIM_COMPONENT_METRICS and key in current
        }
        score = (
            self._aggregate(values, extended) if "score" in metric_names else None
        )
        metadata = dict(current["metadata"])
        metadata["human_filter_applied"] = float(human is not None)
        metadata["extended_comfort_available"] = float(extended is not None)
        if score is not None:
            metadata["weighted_metric_weight"] = float(self._available_weight(extended))
        result_values = {
            name: (
                float(score)
                if name == "score"
                else float(extended)
                if name == "extended_comfort" and extended is not None
                else float(current[name])
            )
            for name in metric_names
            if name == "score"
            or name in current
            or (name == "extended_comfort" and extended is not None)
        }
        return T4NavSimResult(
            version=self.config.version,
            metrics=result_values,
            token=scene.scene_metadata.token,
            metadata=metadata,
        )

    def score_sequence(
        self,
        samples: Sequence[tuple[Trajectory, T4Scene]],
        *,
        metric_names: Optional[Sequence[str]] = None,
        require_extended_comfort: bool = False,
    ) -> tuple[T4NavSimResult, ...]:
        """Score consecutive samples, adding EC from the preceding plan.

        Samples must be ordered in time.  A first sample has unavailable EC;
        later samples use the immediately preceding pair.  This method does
        not invent a previous plan for the first sample.
        """

        selected_metrics = self._resolve_metric_names(metric_names)
        if require_extended_comfort and "extended_comfort" not in required_navsim_metric_names(
            self.config.version, selected_metrics
        ):
            raise NavSimScoringError(
                "require_extended_comfort needs extended_comfort or score in metric_names"
            )
        trajectories = [trajectory for trajectory, _ in samples]
        scenes = [scene for _, scene in samples]
        previous_trajectories = [None, *trajectories[:-1]]
        previous_scenes = [None, *scenes[:-1]]
        results = self.score_batch(
            trajectories,
            scenes,
            metric_names=selected_metrics,
            previous_trajectories=previous_trajectories,
            previous_scenes=previous_scenes,
        )
        for index, result in enumerate(results):
            if require_extended_comfort and not result.metadata.get(
                "extended_comfort_available", False
            ):
                raise NavSimScoringError(f"sample {index} has no previous plan for extended comfort")
        return results

    def score_proposals(
        self,
        proposals,
        scenes: Sequence[T4Scene],
        *,
        metric_names: Optional[Sequence[str]] = None,
        trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> T4NavSimProposalResult:
        """Score proposal tensors for detached training-time diagnostics.

        The metric result is not differentiable. Proposal tensors are copied
        out of the training graph, converted to the declared trajectory grid,
        and evaluated through the same batched scorer used by standalone runs.
        The returned object contains a ``[batch, proposals, metrics]`` tensor
        and its exact ``metric_names`` column order.
        """

        import torch

        if not torch.is_tensor(proposals) or proposals.ndim != 4 or proposals.shape[-1] != 3:
            raise NavSimScoringError(
                "proposals must be a floating tensor with shape [B,N,T,3]"
            )
        if not proposals.is_floating_point() or not torch.isfinite(proposals).all():
            raise NavSimScoringError("proposals must contain finite floating-point values")
        if len(scenes) != proposals.shape[0]:
            raise NavSimScoringError("proposal and scene batch sizes must match")
        selected_metrics = self._resolve_metric_names(metric_names)
        sampling = trajectory_sampling or TrajectorySampling(
            num_poses=int(proposals.shape[-2]),
            interval_length=float(self.config.interval_s),
        )
        flat_trajectories = [
            Trajectory(
                poses=proposal.detach().cpu().numpy(),
                trajectory_sampling=sampling,
            )
            for proposal_batch in proposals
            for proposal in proposal_batch
        ]
        flat_scenes = [scene for scene in scenes for _ in range(proposals.shape[1])]
        results = self.score_batch(
            flat_trajectories,
            flat_scenes,
            metric_names=selected_metrics,
        )
        output = torch.full(
            (proposals.shape[0], proposals.shape[1], len(selected_metrics)),
            float("nan"),
            device=proposals.device,
            dtype=proposals.dtype,
        )
        for index, result in enumerate(results):
            batch_index, proposal_index = divmod(index, proposals.shape[1])
            values = result.values
            output[batch_index, proposal_index] = torch.as_tensor(
                [values.get(name, float("nan")) for name in selected_metrics],
                device=proposals.device,
                dtype=proposals.dtype,
            )
        return T4NavSimProposalResult(output, selected_metrics)

    def _score_components(
        self,
        trajectory: Trajectory,
        scene: T4Scene,
        *,
        metric_names: Sequence[str],
    ) -> dict[str, Any]:
        required = set(metric_names)
        poses = _dense_poses(trajectory, self.config)
        frame = scene.current_frame
        map_tensors = frame.map_tensors
        shape = frame.ego_status.ego_shape
        map_metrics = {
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "traffic_light_compliance",
            "time_to_collision_within_bound",
            "lane_keeping",
        }
        needs_map = bool(required.intersection(map_metrics))
        needs_annotations = bool(
            required.intersection(
                {"no_at_fault_collisions", "time_to_collision_within_bound"}
            )
        )
        needs_states = bool(
            required.intersection(
                {
                    "no_at_fault_collisions",
                    "time_to_collision_within_bound",
                    "history_comfort",
                }
            )
        )
        if needs_map and map_tensors is None:
            raise NavSimScoringError("NavSim scoring requires current-frame map tensors")
        if needs_annotations:
            required_annotations = self.config.num_steps + 10
            if (
                scene.future_annotations is None
                or len(scene.future_annotations) < required_annotations
            ):
                raise NavSimScoringError(
                    "NavSim scoring requires current and future tracked-object annotations "
                    f"for {required_annotations} frames"
                )

        states = _simulate(poses, scene, self.config) if needs_states else None
        if needs_map:
            (
                lane_rings,
                route_rings,
                route_centerlines,
                intersection_rings,
                borders,
                red_rings,
                coverage,
            ) = _map_layers(map_tensors)
        else:
            lane_rings, route_rings = [], []
            route_centerlines, intersection_rings = [], []
            borders, red_rings, coverage = [], [], {}

        area_flags = None
        if required.intersection(
            {"no_at_fault_collisions", "time_to_collision_within_bound"}
        ):
            assert states is not None
            area_flags = formulas.ego_area_flags(
                states,
                lane_rings,
                intersection_rings,
                shape.length,
                shape.width,
                center_offset=shape.rear_axle_to_center,
            )

        boxes: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        if needs_annotations:
            assert scene.future_annotations is not None
            boxes = [
                np.asarray(annotation.boxes, dtype=np.float64)
                for annotation in scene.future_annotations[: self.config.num_steps + 1]
            ]
            labels = [
                np.asarray(annotation.labels, dtype=np.int64)
                for annotation in scene.future_annotations[: self.config.num_steps + 1]
            ]

        values: dict[str, float] = {}
        if "no_at_fault_collisions" in required:
            assert states is not None and area_flags is not None
            values["no_at_fault_collisions"] = float(
                formulas.no_at_fault_collision(
                    states,
                    boxes,
                    shape.length,
                    shape.width,
                    agent_labels_per_t=labels,
                    center_offset=shape.rear_axle_to_center,
                    area_flags=area_flags,
                )
            )
        if "drivable_area_compliance" in required:
            values["drivable_area_compliance"] = float(
                _drivable_score(
                    poses,
                    lane_rings,
                    intersection_rings,
                    borders,
                    shape.length,
                    shape.width,
                    shape.rear_axle_to_center,
                )
            )
        if "driving_direction_compliance" in required:
            values["driving_direction_compliance"] = float(
                formulas.ddc_from_route_lanes(poses, route_rings, self.config.interval_s)
            )
        if "time_to_collision_within_bound" in required:
            assert states is not None and area_flags is not None
            values["time_to_collision_within_bound"] = float(
                formulas.time_to_collision(
                    states,
                    boxes,
                    shape.length,
                    shape.width,
                    self.config.interval_s,
                    center_offset=shape.rear_axle_to_center,
                    area_flags=area_flags,
                )
            )
        if "traffic_light_compliance" in required:
            values["traffic_light_compliance"] = float(
                _traffic_light_compliance(
                    poses,
                    red_rings,
                    shape.length,
                    shape.width,
                    shape.rear_axle_to_center,
                )
            )
        if "ego_progress" in required:
            human = _human_poses(scene, self.config)
            ep, ep_gated = formulas.ego_progress_with_gate(poses, human)
            values["ego_progress"] = float(ep)
            coverage["ego_progress_gated"] = float(ep_gated)
        if "lane_keeping" in required:
            values["lane_keeping"] = float(
                _lane_keeping(poses, route_centerlines, intersection_rings, self.config)
            )
        if "history_comfort" in required:
            assert states is not None
            history = _history_comfort(scene, states, self.config)
            if self.config.version == "v1":
                history = float(
                    formulas.ego_is_comfortable(
                        states[None],
                        np.arange(states.shape[0], dtype=np.float64) * self.config.interval_s,
                    ).all(axis=-1)[0]
                )
            values["history_comfort"] = float(history)
        values["metadata"] = coverage
        return values

    def _aggregate(self, values: Mapping[str, float], extended: Optional[float]) -> float:
        if self.config.version == "v1":
            multiplier = values["no_at_fault_collisions"] * values["drivable_area_compliance"]
            weighted = (
                self.config.progress_weight * values["ego_progress"]
                + self.config.ttc_weight * values["time_to_collision_within_bound"]
                + self.config.history_comfort_weight * values["history_comfort"]
            )
            denominator = (
                self.config.progress_weight
                + self.config.ttc_weight
                + self.config.history_comfort_weight
            )
            return float(multiplier * weighted / denominator)
        if extended is None and self.config.require_extended_comfort:
            raise NavSimScoringError("v2 EPDMS cannot aggregate without extended comfort")
        weighted = [
            (self.config.progress_weight, values["ego_progress"]),
            (self.config.ttc_weight, values["time_to_collision_within_bound"]),
            (self.config.lane_keeping_weight, values["lane_keeping"]),
            (self.config.history_comfort_weight, values["history_comfort"]),
        ]
        if extended is not None:
            weighted.append((self.config.extended_comfort_weight, float(extended)))
        denominator = sum(weight for weight, _ in weighted)
        if denominator <= 0.0:
            raise NavSimScoringError("EPDMS weighted metric denominator is zero")
        multiplier = (
            values["no_at_fault_collisions"]
            * values["drivable_area_compliance"]
            * values["driving_direction_compliance"]
            * values["traffic_light_compliance"]
        )
        return float(multiplier * sum(weight * value for weight, value in weighted) / denominator)

    def _available_weight(self, extended: Optional[float]) -> float:
        if self.config.version == "v1":
            return self.config.progress_weight + self.config.ttc_weight + self.config.history_comfort_weight
        return (
            self.config.progress_weight
            + self.config.ttc_weight
            + self.config.lane_keeping_weight
            + self.config.history_comfort_weight
            + (self.config.extended_comfort_weight if extended is not None else 0.0)
        )

    def _human_trajectory(self, scene: T4Scene) -> Trajectory:
        try:
            return scene.get_future_trajectory(
                trajectory_sampling=TrajectorySampling(
                    num_poses=self.config.num_steps,
                    interval_length=self.config.interval_s,
                )
            )
        except (TypeError, ValueError) as error:
            raise NavSimScoringError("T4 scene has no usable human future trajectory") from error

    def _extended_comfort(
        self,
        trajectory: Trajectory,
        scene: T4Scene,
        previous_trajectory: Trajectory,
        previous_scene: T4Scene,
    ) -> float:
        current = _dense_poses(trajectory, self.config)
        previous = _dense_poses(previous_trajectory, self.config)
        current = _local_to_global(current, scene)
        previous = _local_to_global(previous, previous_scene)
        delta_s = (
            float(scene.current_frame.timestamp_us - previous_scene.current_frame.timestamp_us) / 1e6
        )
        if delta_s <= 0.0:
            delta_s = self.config.observation_interval_s
        if delta_s >= self.config.horizon_s:
            raise NavSimScoringError(
                "consecutive plan interval must be shorter than the score horizon"
            )
        return float(
            formulas.extended_comfort_navsim(
                previous,
                current,
                self.config.interval_s,
                observation_interval=delta_s,
                initial_velocity=(
                    previous_scene.current_frame.ego_status.speed,
                    scene.current_frame.ego_status.speed,
                ),
                wheel_base=scene.current_frame.ego_status.ego_shape.wheel_base,
            )
        )


def aggregate_navsim_results(results: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average NavSim fields independently, omitting unavailable EC values."""

    if not results:
        return {"num_scenes": 0.0}
    keys = sorted({str(key) for result in results for key in result})
    output = {"num_scenes": float(len(results))}
    for key in keys:
        values = [float(result[key]) for result in results if key in result and np.isfinite(float(result[key]))]
        if values:
            output[key] = float(np.mean(values))
    return output


def aggregate_pseudo_closed_loop(
    first_stage_score: float,
    first_stage_endpoint_xy: Sequence[float],
    followups: Sequence[NavSimFollowup],
    *,
    sigma_squared: float = 0.1,
) -> float:
    """Apply NavSim v2's Gaussian follow-up weighting to one stage-1 score."""

    if sigma_squared <= 0.0 or not isfinite(float(sigma_squared)):
        raise ValueError("sigma_squared must be finite and positive")
    if not followups:
        raise ValueError("at least one follow-up score is required")
    endpoint = np.asarray(first_stage_endpoint_xy, dtype=np.float64).reshape(-1)
    if endpoint.shape != (2,) or not np.isfinite(endpoint).all():
        raise ValueError("first_stage_endpoint_xy must be two finite values")
    distances = np.asarray(
        [np.sum((endpoint - np.asarray(item.start_xy, dtype=np.float64)) ** 2) for item in followups],
        dtype=np.float64,
    )
    weights = np.exp(-distances / (2.0 * float(sigma_squared)))
    if not np.isfinite(weights).all() or np.isclose(weights.sum(), 0.0):
        weights = np.full(len(followups), 1.0 / len(followups), dtype=np.float64)
    else:
        weights /= weights.sum()
    second_stage = float(
        sum(weight * item.result.score for weight, item in zip(weights, followups, strict=True))
    )
    return float(first_stage_score * second_stage)


def _dense_poses(trajectory: Trajectory, config: T4NavSimScorerConfig) -> np.ndarray:
    try:
        dense = trajectory.resample(
            TrajectorySampling(num_poses=config.num_steps, interval_length=config.interval_s)
        )
    except ValueError as error:
        raise NavSimScoringError(
            f"trajectory must cover {config.horizon_s:g}s for NavSim scoring"
        ) from error
    future = np.asarray(dense.poses, dtype=np.float64)
    poses = np.vstack((np.zeros((1, 3), dtype=np.float64), future))
    heading = np.unwrap(poses[:, 2])
    return np.column_stack((poses[:, 0], poses[:, 1], np.cos(heading), np.sin(heading)))


def _simulate(poses4: np.ndarray, scene: T4Scene, config: T4NavSimScorerConfig) -> np.ndarray:
    xyh = np.column_stack((poses4[:, 0], poses4[:, 1], np.arctan2(poses4[:, 3], poses4[:, 2])))
    shape = scene.current_frame.ego_status.ego_shape
    initial_velocity = np.asarray([scene.current_frame.ego_status.speed], dtype=np.float64)
    if not config.use_simulator:
        return formulas.states_from_poses(poses4[None], config.interval_s)[0]
    return simulate_proposals(
        xyh[None],
        initial_velocity,
        wheel_base=shape.wheel_base,
        dt=config.interval_s,
        width=shape.width,
    )[0]


def _human_poses(scene: T4Scene, config: T4NavSimScorerConfig) -> np.ndarray:
    try:
        human = scene.get_future_trajectory(
            trajectory_sampling=TrajectorySampling(
                num_poses=config.num_steps,
                interval_length=config.interval_s,
            )
        )
    except (TypeError, ValueError) as error:
        raise NavSimScoringError("T4 scene has no usable human future trajectory") from error
    return _dense_poses(human, config)


def _history_comfort(scene: T4Scene, states: np.ndarray, config: T4NavSimScorerConfig) -> float:
    history = np.asarray(scene.get_history_poses(), dtype=np.float64)
    history4 = np.column_stack(
        (history[:, 0], history[:, 1], np.cos(history[:, 2]), np.sin(history[:, 2]))
    )
    history_states = (
        formulas.states_from_poses(history4[:-1][None], config.interval_s)[0]
        if history4.shape[0] > 1
        else np.zeros((0, formulas.STATE_SIZE), dtype=np.float64)
    )
    combined = np.concatenate((history_states, states), axis=0)
    times = np.arange(combined.shape[0], dtype=np.float64) * config.interval_s
    return float(formulas.ego_is_comfortable(combined[None], times).all(axis=-1)[0])


def _map_layers(map_tensors) -> tuple[list, list, list, list, list, list, dict[str, float]]:
    lanes = _valid_rows(map_tensors.lanes, min_columns=8)
    route = _valid_rows(map_tensors.route_lanes, min_columns=8)
    polygons = _valid_rows(map_tensors.polygons, min_columns=2)
    lines = _valid_rows(map_tensors.line_strings, min_columns=2)
    lane_rings = [_lane_ring(row) for row in lanes]
    lane_rings = [ring for ring in lane_rings if ring is not None]
    route_rings = [_lane_ring(row) for row in route]
    route_rings = [ring for ring in route_rings if ring is not None]
    route_centerlines = [row[:, :2] for row in route if row.shape[0] >= 2]
    intersection_rings = [row[:, :2] for row in polygons if row.shape[0] >= 3]
    borders = [row[:, :2] for row in lines if row.shape[1] >= 4 and np.any(row[:, 3] > 0.5)]
    red_rings = [ring for row, ring in zip(route, route_rings, strict=False) if np.any(row[:, 10] > 0.5)]
    route_count = max(len(route), 1)
    line_count = len(lines)
    coverage = {
        "dac_border_frac": float(len(borders) / max(line_count, 1)),
        "ddc_route_frac": float(len(route_rings) / route_count),
        "lane_centerline_frac": float(len(route_centerlines) / route_count),
        "traffic_light_route_frac": float(len(red_rings) / route_count),
    }
    return lane_rings, route_rings, route_centerlines, intersection_rings, borders, red_rings, coverage


def _valid_rows(values: np.ndarray, *, min_columns: int) -> list[np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] < min_columns:
        return []
    rows = []
    for row in array:
        valid = np.linalg.norm(row[:, :2], axis=-1) > 1.0e-6
        if valid.sum() >= 2:
            rows.append(np.ascontiguousarray(row[valid]))
    return rows


def _lane_ring(row: np.ndarray) -> Optional[np.ndarray]:
    if row.shape[0] < 2 or row.shape[1] < 8:
        return None
    center = row[:, :2]
    left = center + row[:, 4:6]
    right = center + row[:, 6:8]
    ring = np.concatenate((left, right[::-1]), axis=0)
    return ring if ring.shape[0] >= 3 else None


def _drivable_score(
    poses: np.ndarray,
    lane_rings: list,
    intersection_rings: list,
    borders: list,
    length: float,
    width: float,
    center_offset: float,
) -> float:
    if borders:
        return float(formulas.dac_from_road_borders(poses, borders, length, width, center_offset))
    return float(
        formulas.dac_semantic(
            poses,
            {"road": lane_rings, "intersection": intersection_rings, "border": []},
            length,
            width,
            center_offset,
        )
    )


def _traffic_light_compliance(
    poses: np.ndarray,
    red_rings: list,
    length: float,
    width: float,
    center_offset: float,
) -> float:
    if not red_rings:
        return 1.0
    from shapely.geometry import Polygon

    red_polygons = []
    for ring in red_rings:
        polygon = Polygon(np.asarray(ring, dtype=np.float64))
        if polygon.is_valid and not polygon.is_empty:
            red_polygons.append(polygon)
    if not red_polygons:
        return 1.0
    for pose in poses:
        heading = float(np.arctan2(pose[3], pose[2]))
        cx = float(pose[0] + center_offset * np.cos(heading))
        cy = float(pose[1] + center_offset * np.sin(heading))
        ego = formulas._polygon(formulas.ego_corners(cx, cy, heading, length, width))
        if any(ego.intersects(red) for red in red_polygons):
            return 0.0
    return 1.0


def _lane_keeping(
    poses: np.ndarray,
    centerlines: list,
    intersection_rings: list,
    config: T4NavSimScorerConfig,
) -> float:
    """NavSim v2 lane keeping with its continuous sample-count threshold."""

    from shapely.geometry import LineString, Point, Polygon

    lines = [LineString(np.asarray(line, dtype=np.float64)) for line in centerlines if len(line) >= 2]
    if not lines:
        return 1.0
    intersections = []
    for ring in intersection_rings:
        polygon = Polygon(np.asarray(ring, dtype=np.float64))
        if polygon.is_valid and not polygon.is_empty:
            intersections.append(polygon)
    required = int(ceil(config.lane_keeping_horizon_s / config.interval_s))
    consecutive = 0
    for pose in poses:
        point = Point(float(pose[0]), float(pose[1]))
        if any(polygon.covers(point) for polygon in intersections):
            consecutive = 0
            continue
        distance = min(point.distance(line) for line in lines)
        consecutive = consecutive + 1 if distance > config.lane_keeping_deviation_m else 0
        if consecutive >= required:
            return 0.0
    return 1.0


def _local_to_global(poses: np.ndarray, scene: T4Scene) -> np.ndarray:
    center = scene.scene_metadata.global_center_pose
    if center is None:
        return np.array(poses, dtype=np.float64, copy=True)
    center = np.asarray(center, dtype=np.float64).reshape(4)
    c, s = center[2], center[3]
    x = center[0] + c * poses[:, 0] - s * poses[:, 1]
    y = center[1] + s * poses[:, 0] + c * poses[:, 1]
    heading = np.unwrap(np.arctan2(poses[:, 3], poses[:, 2]) + np.arctan2(s, c))
    return np.column_stack((x, y, np.cos(heading), np.sin(heading)))


def _apply_human_filter(
    agent: Mapping[str, Any], human: Mapping[str, Any]
) -> dict[str, Any]:
    filtered = dict(agent)
    for key in (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "lane_keeping",
        "history_comfort",
        "extended_comfort",
    ):
        if key in human and human[key] is not None and float(human[key]) == 0.0:
            filtered[key] = 1.0
    return filtered


__all__ = [
    "NAVSIM_COMPONENT_METRICS",
    "NAVSIM_METRICS",
    "NAVSIM_V1_METRICS",
    "NAVSIM_V2_METRICS",
    "NAVSIM_VERSIONS",
    "NavSimFollowup",
    "NavSimScoringError",
    "T4NavSimResult",
    "T4NavSimProposalResult",
    "T4NavSimScorer",
    "T4NavSimScorerConfig",
    "aggregate_navsim_results",
    "aggregate_pseudo_closed_loop",
    "required_navsim_metric_names",
    "resolve_navsim_metric_names",
]
