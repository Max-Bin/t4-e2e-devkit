"""Metric builder and violation base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from .metric_result import (
    MetricStatistics,
    MetricStatisticsType,
    MetricViolation,
    Statistic,
    TimeSeries,
)


class AbstractMetric(ABC):
    """Compute a typed metric from a simulation history."""

    def __init__(self, name: str, category: str, metric_score_unit: Optional[str] = None) -> None:
        if not str(name) or not str(category):
            raise ValueError("metric name and category must not be empty")
        self._name = str(name)
        self._category = str(category)
        self.metric_score_unit = metric_score_unit

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @abstractmethod
    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        """Compute this metric."""

    def compute_score(
        self,
        scenario: Any,
        metric_statistics: list[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        del scenario, time_series
        values = [float(statistic.value) for statistic in metric_statistics if not isinstance(statistic.value, bool)]
        return 0.0 if not values else float(values[0])

    def _construct_metric_results(
        self,
        statistics: list[Statistic],
        *,
        scenario: Any = None,
        time_series: Optional[TimeSeries] = None,
    ) -> list[MetricStatistics]:
        score = self.compute_score(scenario, statistics, time_series)
        return [
            MetricStatistics(
                metric_computator=type(self).__name__,
                name=self.name,
                metric_category=self.category,
                statistics=statistics,
                time_series=time_series,
                metric_score=score,
                metric_score_unit=self.metric_score_unit,
            )
        ]


MetricBase = AbstractMetric


class ViolationMetricBase(AbstractMetric):
    """Aggregate contiguous or instantaneous violations."""

    def __init__(self, name: str, category: str, max_violation_threshold: int = 0, metric_score_unit: Optional[str] = None) -> None:
        super().__init__(name, category, metric_score_unit)
        if max_violation_threshold < 0:
            raise ValueError("max_violation_threshold must be non-negative")
        self.max_violation_threshold = int(max_violation_threshold)
        self.number_of_violations = 0

    def aggregate_metric_violations(
        self,
        metric_violations: Iterable[MetricViolation],
        scenario: Any = None,
        time_series: Optional[TimeSeries] = None,
    ) -> list[MetricStatistics]:
        violations = list(metric_violations)
        if any(item.name != self.name for item in violations):
            raise ValueError("all violations must belong to the same metric")
        self.number_of_violations = len(violations)
        if not violations:
            statistics = [
                Statistic(self.name, MetricStatisticsType.BOOLEAN.unit, MetricStatisticsType.BOOLEAN, True)
            ]
        else:
            durations = [max(1, item.duration) for item in violations]
            total_duration = sum(durations)
            statistics = [
                Statistic(f"number_of_violations_of_{self.name}", MetricStatisticsType.COUNT.unit, MetricStatisticsType.COUNT, len(violations)),
                Statistic(f"max_violation_of_{self.name}", violations[0].unit, MetricStatisticsType.MAX, max(item.extremum for item in violations)),
                Statistic(f"min_violation_of_{self.name}", violations[0].unit, MetricStatisticsType.MIN, min(item.extremum for item in violations)),
                Statistic(f"mean_violation_of_{self.name}", violations[0].unit, MetricStatisticsType.MEAN, sum(item.mean * duration for item, duration in zip(violations, durations, strict=True)) / total_duration),
                Statistic(self.name, MetricStatisticsType.BOOLEAN.unit, MetricStatisticsType.BOOLEAN, False),
            ]
        return self._construct_metric_results(statistics, scenario=scenario, time_series=time_series)

    def compute_score(self, scenario: Any, metric_statistics: list[Statistic], time_series: Optional[TimeSeries] = None) -> float:
        del scenario, metric_statistics, time_series
        return max(0.0, 1.0 - self.number_of_violations / (self.max_violation_threshold + 1))


class WithinBoundMetricBase(ViolationMetricBase):
    """Base for a scalar series that must stay within an inclusive bound."""

    def __init__(self, name: str, category: str, lower_bound: float = float("-inf"), upper_bound: float = float("inf"), **kwargs: Any) -> None:
        super().__init__(name, category, **kwargs)
        if lower_bound > upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    def find_violations(self, time_series: TimeSeries) -> list[MetricViolation]:
        violations: list[MetricViolation] = []
        for index, value in enumerate(time_series.values):
            if self.lower_bound <= value <= self.upper_bound:
                continue
            severity = self.lower_bound - value if value < self.lower_bound else value - self.upper_bound
            timestamp = time_series.time_stamps[index]
            duration = (
                time_series.time_stamps[index + 1] - timestamp
                if index + 1 < len(time_series.time_stamps)
                else 0
            )
            violations.append(MetricViolation(type(self).__name__, self.name, self.category, time_series.unit, timestamp, duration, severity, severity))
        return violations


__all__ = ["AbstractMetric", "MetricBase", "ViolationMetricBase", "WithinBoundMetricBase"]
