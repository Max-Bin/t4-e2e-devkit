"""The types the NavSim scorer takes in and hands back.

Split out of :mod:`t4_e2e_devkit.evaluation.navsim_score`, which is a 1252-line
module whose subject is the scoring algorithm.  These five have no algorithm in
them -- a config, two results, a follow-up record and the error -- and they are
what a caller of the scorer actually imports, so they read better away from the
1000 lines they were embedded in.

Every name stays importable from ``evaluation.navsim_score``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from t4_e2e_devkit.common.constants import (
    SCORER_FUTURE_FRAMES,
    T4_INTERVAL_LENGTH,
)
from t4_e2e_devkit.evaluation.reference import pdms_navsim as formulas

#: The metric versions the scorer implements.  The vocabulary below lives here
#: for the same reason the config does: it is what a caller selects from, and the
#: config validates against it.  ``evaluation.navsim_score`` re-exports all of it.
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
        raise ValueError(f"version must be one of {NAVSIM_VERSIONS}, got {version!r}")
    available = NAVSIM_V1_METRICS if normalized_version == "v1" else NAVSIM_V2_METRICS
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
    # The analyzer's queue exemption (epdms/subscores/lane_keeping.hpp
    # defaults): a sample this slow that also covered no more than
    # ``queue_progress_m`` over the trailing ``queue_window_s`` is waiting, not
    # drifting, and neither it nor the following ``queue_release_s`` counts.
    lane_keeping_queue_speed_mps: float = 1.0
    lane_keeping_queue_window_s: float = 1.0
    lane_keeping_queue_progress_m: float = 1.5
    lane_keeping_queue_release_s: float = 1.5
    # The analyzer widens each signalled window by these before exempting it.
    # Only the turn-indicator half is available on T4; see ``lane_change_exempt``.
    lane_keeping_lane_change_pre_s: float = 1.0
    lane_keeping_lane_change_post_s: float = 1.0
    use_simulator: bool = True
    # CUDA Graph capture is opt-in until a deployment validates numerical
    # parity against eager simulation on its target driver and GPU.
    compile_rollout: bool = False
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
            "lane_keeping_queue_speed_mps",
            "lane_keeping_queue_window_s",
            "lane_keeping_queue_progress_m",
            "lane_keeping_queue_release_s",
            "lane_keeping_lane_change_pre_s",
            "lane_keeping_lane_change_post_s",
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
