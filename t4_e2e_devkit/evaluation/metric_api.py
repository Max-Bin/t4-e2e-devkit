"""NuPlan-style metric builders, time series and aggregation primitives.

The existing :mod:`metric_engine` remains the compact family adapter used by
T4 evaluation.  This module adds the richer builder/statistics boundary used by
simulation callbacks and batch reports without coupling it to a database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from t4_e2e_devkit.evaluation.metrics.abstract_metric import (
    AbstractMetric,
    MetricBase,
    ViolationMetricBase,
    WithinBoundMetricBase,
)
from t4_e2e_devkit.evaluation.metrics.metric_dataframe import MetricStatisticsDataFrame
from t4_e2e_devkit.evaluation.metrics.metric_file import MetricFile, MetricFileKey
from t4_e2e_devkit.evaluation.metrics.metric_result import (
    MetricStatistics as NuPlanMetricStatistics,
)
from t4_e2e_devkit.evaluation.metrics.metric_result import (
    MetricStatisticsType,
    MetricViolation,
    Statistic,
    TimeSeries,
)
from t4_e2e_devkit.evaluation.metrics.weighted_average import WeightedAverageMetricAggregator


@dataclass(frozen=True)
class MetricTimeSeries:
    """A named scalar series sampled at simulation timestamps."""

    timestamps_us: np.ndarray
    values: np.ndarray
    unit: str = ""
    name: Optional[str] = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_us, dtype=np.int64).reshape(-1)
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if len(timestamps) != len(values):
            raise ValueError("metric time-series timestamps and values must align")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            raise ValueError("metric time-series timestamps must be non-decreasing")
        object.__setattr__(self, "timestamps_us", np.ascontiguousarray(timestamps))
        object.__setattr__(self, "values", np.ascontiguousarray(values))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "timestamps_us": self.timestamps_us.tolist(),
            "values": self.values.tolist(),
        }


@dataclass(frozen=True)
class MetricStatistic:
    """One scalar statistic or an optional associated time-series."""

    name: str
    value: float
    unit: str = ""
    statistic_type: str = "scalar"
    time_series: Optional[MetricTimeSeries] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("metric statistic name must not be empty")
        if self.statistic_type not in {"scalar", "time_series", "boolean", "count"}:
            raise ValueError(f"unsupported statistic_type: {self.statistic_type}")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "value", float(self.value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "statistic_type": self.statistic_type,
            "time_series": None if self.time_series is None else self.time_series.as_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MetricResult:
    """Output of one metric builder for one scenario or simulation."""

    metric_name: str
    statistics: tuple[MetricStatistic, ...]
    scenario_token: Optional[str] = None
    failure: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> dict[str, float]:
        """:return: scalar values keyed by statistic name."""

        return {statistic.name: float(statistic.value) for statistic in self.statistics}

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "scenario_token": self.scenario_token,
            "failure": self.failure,
            "statistics": [statistic.as_dict() for statistic in self.statistics],
            "metadata": dict(self.metadata),
        }


# Compatibility name used by metric consumers that call one builder's output
# ``MetricStatistics`` rather than ``MetricResult``.
MetricStatistics = MetricResult


class AbstractMetricBuilder(ABC):
    """Build one metric family from a simulation history-like object."""

    @property
    @abstractmethod
    def name(self) -> str:
        """:return: stable metric family name."""

    def get_metric_name(self) -> str:
        """NuPlan-shaped accessor for callers that avoid properties."""

        return self.name

    @property
    def category(self) -> str:
        """Metric category used by typed report consumers."""

        return "default"

    def get_metric_category(self) -> str:
        return self.category

    def compute_score(self, statistics: Sequence[MetricStatistic]) -> float:
        values = [float(item.value) for item in statistics]
        return 0.0 if not values else float(values[0])

    def get_required_statistics(self) -> tuple[str, ...]:
        """Return required inputs for dependency-aware metric runners."""

        return ()

    def get_aggregator(self) -> "MetricAggregator":
        """Return the default family-preserving aggregator."""

        return MetricAggregator()

    @abstractmethod
    def compute(self, history: Any, *, scenario_token: Optional[str] = None) -> MetricResult:
        """Compute one result from a history or scenario context."""


class CallableMetricBuilder(AbstractMetricBuilder):
    """Adapt a function returning statistics or a :class:`MetricResult`."""

    def __init__(self, name: str, function) -> None:
        if not str(name):
            raise ValueError("metric builder name must not be empty")
        self._name = str(name)
        self.function = function

    @property
    def name(self) -> str:
        return self._name

    def compute(self, history: Any, *, scenario_token: Optional[str] = None) -> MetricResult:
        output = self.function(history)
        if isinstance(output, MetricResult):
            return output
        if isinstance(output, Mapping):
            statistics = tuple(
                MetricStatistic(name=str(name), value=float(value))
                for name, value in output.items()
            )
        else:
            statistics = (MetricStatistic(name=self.name, value=float(output)),)
        return MetricResult(self.name, statistics, scenario_token=scenario_token)


class MetricAggregator:
    """Aggregate scalar statistics without mixing metric families."""

    def aggregate(
        self,
        results: Iterable[MetricResult],
        *,
        weights: Optional[Mapping[str, float]] = None,
        include_failures: bool = False,
    ) -> dict[str, dict[str, float]]:
        grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
        counts: dict[str, int] = {}
        for result in results:
            if result.failure and not include_failures:
                continue
            counts[result.metric_name] = counts.get(result.metric_name, 0) + 1
            weight = 1.0 if weights is None else float(weights.get(result.scenario_token or "", 1.0))
            if weight < 0.0:
                raise ValueError("metric aggregation weights must be non-negative")
            for statistic in result.statistics:
                grouped.setdefault((result.metric_name, statistic.name), []).append(
                    (float(statistic.value), weight)
                )
        output: dict[str, dict[str, float]] = {}
        for metric_name in sorted(counts):
            values: dict[str, float] = {"num_scenarios": float(counts[metric_name])}
            for (family, statistic_name), samples in sorted(grouped.items()):
                if family != metric_name:
                    continue
                denominator = sum(weight for _, weight in samples)
                values[statistic_name] = (
                    float(sum(value * weight for value, weight in samples) / denominator)
                    if denominator > 0.0
                    else float("nan")
                )
            output[metric_name] = values
        return output

    def aggregate_many(
        self,
        results_by_scenario: Iterable[Iterable[MetricResult]],
        *,
        include_failures: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Flatten per-scenario builder output before aggregation."""

        return self.aggregate(
            (result for scenario in results_by_scenario for result in scenario),
            include_failures=include_failures,
        )


class MetricBuilderRegistry:
    """Deterministic registry used by simulation and reporting code."""

    def __init__(self, builders: Optional[Sequence[AbstractMetricBuilder]] = None) -> None:
        self._builders: dict[str, AbstractMetricBuilder] = {}
        for builder in builders or ():
            self.register(builder)

    @property
    def builders(self) -> tuple[AbstractMetricBuilder, ...]:
        return tuple(self._builders.values())

    def register(self, builder: AbstractMetricBuilder) -> None:
        name = str(builder.name)
        if not name:
            raise ValueError("metric builder name must not be empty")
        if name in self._builders:
            raise ValueError(f"metric builder is already registered: {name}")
        self._builders[name] = builder

    def compute(self, history: Any, *, scenario_token: Optional[str] = None) -> tuple[MetricResult, ...]:
        return tuple(
            _compute_builder(builder, history, scenario_token=scenario_token)
            for builder in self._builders.values()
        )


class MetricCallback:
    """Simulation callback that evaluates registered builders at run end."""

    def __init__(
        self,
        builders: Sequence[AbstractMetricBuilder],
        *,
        scenario_token: Optional[str] = None,
    ) -> None:
        self.registry = MetricBuilderRegistry(builders)
        self.scenario_token = scenario_token
        self.results: tuple[MetricResult, ...] = ()

    def on_simulation_end(self, *args: Any) -> None:
        """Compute metrics from either a history or a full runner callback."""

        if len(args) == 1:
            history = args[0]
            token = self.scenario_token
        elif len(args) == 3:
            setup, _planner, history = args
            token = self.scenario_token or _scenario_token(setup)
        else:
            raise TypeError(
                "MetricCallback.on_simulation_end expects history or "
                "(setup, planner, history)"
            )
        self.results = self.registry.compute(history, scenario_token=token)

    def on_simulation_error(self, *args: Any) -> None:
        del args


def _scenario_token(setup: Any) -> Optional[str]:
    scenario = getattr(setup, "scenario", None)
    token = getattr(scenario, "token", getattr(scenario, "scenario_name", None))
    return None if token is None or str(token) == "" else str(token)


def _compute_builder(builder: AbstractMetricBuilder, history: Any, *, scenario_token: Optional[str]) -> MetricResult:
    """Call compact and scenario-shaped builders through one boundary."""

    import inspect

    parameters = inspect.signature(builder.compute).parameters
    if "scenario_token" in parameters:
        return builder.compute(history, scenario_token=scenario_token)
    if "scenario" in parameters:
        return builder.compute(history, scenario=None)  # type: ignore[call-arg]
    return builder.compute(history)  # type: ignore[call-arg]


__all__ = [
    "AbstractMetricBuilder",
    "CallableMetricBuilder",
    "MetricAggregator",
    "MetricBuilderRegistry",
    "MetricCallback",
    "MetricResult",
    "MetricStatistics",
    "MetricStatistic",
    "MetricTimeSeries",
    "AbstractMetric",
    "MetricBase",
    "ViolationMetricBase",
    "WithinBoundMetricBase",
    "MetricStatisticsDataFrame",
    "MetricStatisticsType",
    "MetricViolation",
    "Statistic",
    "TimeSeries",
    "MetricFile",
    "MetricFileKey",
    "NuPlanMetricStatistics",
    "WeightedAverageMetricAggregator",
]
