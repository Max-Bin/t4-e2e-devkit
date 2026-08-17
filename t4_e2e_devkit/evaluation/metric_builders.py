"""Reusable metric builders for the T4 evaluation families.

The builders deliberately accept either a domain object or a plain mapping.
That keeps the metric lifecycle useful for a simulation runner, a cached
artifact reader and a small offline script without introducing a database
model into the public API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional, Sequence

from t4_e2e_devkit.evaluation.closed_loop import (
    ClosedLoopMetricConfig,
    ClosedLoopMetrics,
    compute_closed_loop_metrics,
)
from t4_e2e_devkit.evaluation.metric_api import (
    AbstractMetricBuilder,
    MetricResult,
    MetricStatistic,
)
from t4_e2e_devkit.evaluation.open_loop import (
    OpenLoopMetricConfig,
    compute_open_loop_metrics,
)
from t4_e2e_devkit.planning.simulation.closed_loop import T4ClosedLoopResult


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    values = getattr(value, "values", None)
    if isinstance(values, Mapping):
        return values
    if callable(values):
        result = values()
        if isinstance(result, Mapping):
            return result
    raise TypeError(f"metric input has no scalar values mapping: {type(value).__name__}")


def _statistic_result(
    name: str,
    values: Mapping[str, Any],
    *,
    scenario_token: Optional[str],
    units: Optional[Mapping[str, str]] = None,
) -> MetricResult:
    statistics = tuple(
        MetricStatistic(
            name=str(key),
            value=float(value),
            unit="" if units is None else str(units.get(str(key), "")),
            statistic_type="count" if str(key).startswith("num_") else "scalar",
        )
        for key, value in values.items()
    )
    return MetricResult(name, statistics, scenario_token=scenario_token)


class MappingMetricBuilder(AbstractMetricBuilder):
    """Expose selected scalar fields from a metric-like input."""

    def __init__(
        self,
        name: str,
        fields: Sequence[str],
        *,
        source: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if not str(name):
            raise ValueError("metric builder name must not be empty")
        if not fields:
            raise ValueError("mapping metric builder needs at least one field")
        self._name = str(name)
        self.fields = tuple(str(field) for field in fields)
        self.source = source or (lambda value: value)

    @property
    def name(self) -> str:
        return self._name

    def compute(self, history: Any, *, scenario_token: Optional[str] = None) -> MetricResult:
        values = _as_mapping(self.source(history))
        missing = [field for field in self.fields if field not in values]
        if missing:
            raise ValueError(f"metric input is missing {self.name} fields: {missing}")
        return _statistic_result(
            self.name,
            {field: values[field] for field in self.fields},
            scenario_token=scenario_token,
        )


class OpenLoopMetricBuilder(AbstractMetricBuilder):
    """Compute trajectory fidelity on the declared prediction/GT time grids."""

    def __init__(
        self,
        prediction: Callable[[Any], Any] | str = "prediction",
        ground_truth: Callable[[Any], Any] | str = "ground_truth",
        *,
        config: Optional[OpenLoopMetricConfig] = None,
        name: str = "open_loop",
    ) -> None:
        self._name = str(name)
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.config = config

    @property
    def name(self) -> str:
        return self._name

    def compute(self, history: Any, *, scenario_token: Optional[str] = None) -> MetricResult:
        prediction = _resolve(history, self.prediction)
        ground_truth = _resolve(history, self.ground_truth)
        output = compute_open_loop_metrics(
            prediction,
            ground_truth,
            config=self.config,
            token=scenario_token,
        )
        return _statistic_result(self.name, output.values, scenario_token=scenario_token)


class ClosedLoopMetricBuilder(AbstractMetricBuilder):
    """Compute realized rollout metrics from a closed-loop result or history."""

    def __init__(
        self,
        result: Optional[Callable[[Any], Any] | str] = None,
        *,
        config: Optional[ClosedLoopMetricConfig] = None,
        name: str = "closed_loop",
        **metric_kwargs: Any,
    ) -> None:
        self._name = str(name)
        self.result = result
        self.config = config
        self.metric_kwargs = dict(metric_kwargs)

    @property
    def name(self) -> str:
        return self._name

    def compute(self, history: Any, *, scenario_token: Optional[str] = None) -> MetricResult:
        value = _resolve(history, self.result) if self.result is not None else history
        if isinstance(value, ClosedLoopMetrics):
            output = value
        else:
            if not isinstance(value, T4ClosedLoopResult):
                value = getattr(value, "closed_loop", getattr(value, "result", value))
            if not isinstance(value, T4ClosedLoopResult):
                raise TypeError("closed-loop metric builder needs T4ClosedLoopResult or ClosedLoopMetrics")
            output = compute_closed_loop_metrics(
                value,
                config=self.config,
                token=scenario_token,
                **self.metric_kwargs,
            )
        return _statistic_result(self.name, output.values, scenario_token=scenario_token)


class PDMMetricBuilder(MappingMetricBuilder):
    """Adapter for scalar proposal-scoring outputs supplied by a caller."""

    def __init__(self, fields: Sequence[str] = ("score",), **kwargs: Any) -> None:
        super().__init__("pdm", fields, **kwargs)


class T4SafetyMetricBuilder(MappingMetricBuilder):
    """Adapter for the independent T4 safety metric family."""

    def __init__(self, fields: Sequence[str] = ("safety_score",), **kwargs: Any) -> None:
        super().__init__("t4_safety", fields, **kwargs)


class ComfortMetricBuilder(MappingMetricBuilder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "comfort",
            ("mean_abs_acceleration_mps2", "max_abs_acceleration_mps2", "max_abs_yaw_rate_radps"),
            **kwargs,
        )


class ProgressMetricBuilder(MappingMetricBuilder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "progress",
            ("path_length_m", "final_displacement_m", "goal_reached"),
            **kwargs,
        )


class CollisionMetricBuilder(MappingMetricBuilder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("collision", ("collision", "first_collision_step"), **kwargs)


class DrivableAreaMetricBuilder(MappingMetricBuilder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("drivable_area", ("drivable_violation",), **kwargs)


class TrafficLightMetricBuilder(MappingMetricBuilder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("traffic_light", ("traffic_light_violation",), **kwargs)


class StopLineViolationMetricBuilder(MappingMetricBuilder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("stop_line", ("stop_line_violation",), **kwargs)


class TTCMetricBuilder(MappingMetricBuilder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("ttc", ("min_ttc_s", "ttc_violation"), **kwargs)


def _resolve(value: Any, selector: Callable[[Any], Any] | str) -> Any:
    if callable(selector):
        return selector(value)
    if isinstance(value, Mapping):
        return value[selector]
    return getattr(value, selector)


__all__ = [
    "ClosedLoopMetricBuilder",
    "CollisionMetricBuilder",
    "ComfortMetricBuilder",
    "DrivableAreaMetricBuilder",
    "MappingMetricBuilder",
    "OpenLoopMetricBuilder",
    "PDMMetricBuilder",
    "ProgressMetricBuilder",
    "StopLineViolationMetricBuilder",
    "T4SafetyMetricBuilder",
    "TTCMetricBuilder",
    "TrafficLightMetricBuilder",
]
