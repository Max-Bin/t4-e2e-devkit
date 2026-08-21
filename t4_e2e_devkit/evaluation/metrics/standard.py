"""Small standard metric family for generic simulation histories."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .abstract_metric import AbstractMetric, WithinBoundMetricBase
from .metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries


class _SeriesMetric(AbstractMetric):
    unit = ""

    def _series(self, history: Any) -> TimeSeries:
        samples = list(history)
        timestamps = [int(item.iteration.time_us) for item in samples]
        return TimeSeries(self.unit, timestamps, self._values(samples))

    def _values(self, samples: list[Any]) -> list[float]:
        raise NotImplementedError

    def _statistic(self, series: TimeSeries) -> list[Statistic]:
        values = np.asarray(series.values, dtype=np.float64)
        return [
            Statistic(
                f"max_{self.name}",
                series.unit,
                MetricStatisticsType.MAX,
                float(np.max(np.abs(values))) if values.size else 0.0,
            ),
            Statistic(
                f"mean_{self.name}",
                series.unit,
                MetricStatisticsType.MEAN,
                float(np.mean(np.abs(values))) if values.size else 0.0,
            ),
        ]

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        series = self._series(history)
        return self._construct_metric_results(
            self._statistic(series), scenario=scenario, time_series=series
        )


class AccelerationMetric(_SeriesMetric):
    unit = "m/s^2"

    def __init__(self, name: str = "acceleration", category: str = "comfort") -> None:
        super().__init__(name, category, metric_score_unit="ratio")

    def _values(self, samples: list[Any]) -> list[float]:
        return [_acceleration(item.ego_state) for item in samples]


class JerkMetric(_SeriesMetric):
    unit = "m/s^3"

    def __init__(self, name: str = "jerk", category: str = "comfort") -> None:
        super().__init__(name, category, metric_score_unit="ratio")

    def _values(self, samples: list[Any]) -> list[float]:
        acceleration = np.asarray(
            [_acceleration(item.ego_state) for item in samples], dtype=np.float64
        )
        times = np.asarray([item.iteration.time_s for item in samples], dtype=np.float64)
        if len(acceleration) < 2:
            return [0.0] * len(acceleration)
        return np.gradient(acceleration, times, edge_order=1).tolist()


class YawRateMetric(_SeriesMetric):
    unit = "rad/s"

    def __init__(self, name: str = "yaw_rate", category: str = "comfort") -> None:
        super().__init__(name, category, metric_score_unit="ratio")

    def _values(self, samples: list[Any]) -> list[float]:
        headings = np.unwrap(
            np.asarray([_heading(item.ego_state) for item in samples], dtype=np.float64)
        )
        times = np.asarray([item.iteration.time_s for item in samples], dtype=np.float64)
        if len(headings) < 2:
            return [0.0] * len(headings)
        return np.gradient(headings, times, edge_order=1).tolist()


class ComfortMetric(AbstractMetric):
    def __init__(self, name: str = "comfort", category: str = "comfort") -> None:
        super().__init__(name, category, metric_score_unit="ratio")
        self.acceleration = AccelerationMetric()
        self.jerk = JerkMetric()
        self.yaw_rate = YawRateMetric()

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        results = self.acceleration.compute(history, scenario)
        results += self.jerk.compute(history, scenario)
        results += self.yaw_rate.compute(history, scenario)
        statistics = [statistic for result in results for statistic in result.statistics]
        return self._construct_metric_results(statistics, scenario=scenario)


class ProgressMetric(AbstractMetric):
    def __init__(self, name: str = "progress", category: str = "progress") -> None:
        super().__init__(name, category, metric_score_unit="m")

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        poses = np.asarray([_pose(item.ego_state) for item in history], dtype=np.float64)
        distance = (
            float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())
            if poses.ndim == 2 and len(poses) > 1
            else 0.0
        )
        statistics = [Statistic("path_length", "m", MetricStatisticsType.VALUE, distance)]
        return self._construct_metric_results(statistics, scenario=scenario)


class ExpertL2ErrorMetric(AbstractMetric):
    def __init__(
        self,
        ground_truth: Callable[[Any], Any],
        name: str = "expert_l2_error",
        category: str = "tracking",
    ) -> None:
        super().__init__(name, category, metric_score_unit="m")
        self.ground_truth = ground_truth

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        predicted = np.asarray([_pose(item.ego_state) for item in history], dtype=np.float64)
        truth = np.asarray(self.ground_truth(history), dtype=np.float64)
        count = min(len(predicted), len(truth))
        values = (
            np.linalg.norm(predicted[:count, :2] - truth[:count, :2], axis=1)
            if count
            else np.zeros(0)
        )
        statistics = [
            Statistic(
                "mean_l2",
                "m",
                MetricStatisticsType.MEAN,
                float(values.mean()) if len(values) else 0.0,
            ),
            Statistic(
                "max_l2", "m", MetricStatisticsType.MAX, float(values.max()) if len(values) else 0.0
            ),
        ]
        return self._construct_metric_results(statistics, scenario=scenario)


class ExpertHeadingErrorMetric(AbstractMetric):
    def __init__(
        self,
        ground_truth: Callable[[Any], Any],
        name: str = "expert_heading_error",
        category: str = "tracking",
    ) -> None:
        super().__init__(name, category, metric_score_unit="rad")
        self.ground_truth = ground_truth

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        predicted = np.asarray([_heading(item.ego_state) for item in history], dtype=np.float64)
        truth = np.asarray(self.ground_truth(history), dtype=np.float64).reshape(-1)
        count = min(len(predicted), len(truth))
        values = np.arctan2(
            np.sin(predicted[:count] - truth[:count]), np.cos(predicted[:count] - truth[:count])
        )
        statistics = [
            Statistic(
                "mean_heading_error",
                "rad",
                MetricStatisticsType.MEAN,
                float(np.mean(np.abs(values))) if len(values) else 0.0,
            )
        ]
        return self._construct_metric_results(statistics, scenario=scenario)


class SpeedLimitMetric(WithinBoundMetricBase):
    def __init__(
        self, speed_limit_mps: float, name: str = "speed_limit", category: str = "speed"
    ) -> None:
        super().__init__(
            name,
            category,
            lower_bound=float("-inf"),
            upper_bound=speed_limit_mps,
            max_violation_threshold=0,
            metric_score_unit="ratio",
        )

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        series = TimeSeries(
            "m/s",
            [int(item.iteration.time_us) for item in history],
            [_speed(item.ego_state) for item in history],
        )
        return self.aggregate_metric_violations(self.find_violations(series), scenario, series)


class CollisionMetric(AbstractMetric):
    def __init__(self, name: str = "collision", category: str = "safety") -> None:
        super().__init__(name, category, metric_score_unit="ratio")

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        collisions = sum(1 for item in history if bool(getattr(item, "collision", False)))
        statistics = [
            Statistic("collision_count", "count", MetricStatisticsType.COUNT, collisions),
            Statistic(
                self.name,
                MetricStatisticsType.BOOLEAN.unit,
                MetricStatisticsType.BOOLEAN,
                collisions == 0,
            ),
        ]
        return self._construct_metric_results(statistics, scenario=scenario)


class TTCMetric(WithinBoundMetricBase):
    def __init__(
        self, minimum_ttc_s: float = 1.0, name: str = "ttc", category: str = "safety"
    ) -> None:
        super().__init__(
            name,
            category,
            lower_bound=minimum_ttc_s,
            upper_bound=float("inf"),
            max_violation_threshold=0,
            metric_score_unit="ratio",
        )

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        values = [_ttc(item) for item in history]
        series = TimeSeries("s", [int(item.iteration.time_us) for item in history], values)
        return self.aggregate_metric_violations(self.find_violations(series), scenario, series)


class StopLineViolationMetric(AbstractMetric):
    def __init__(self, name: str = "stop_line", category: str = "safety") -> None:
        super().__init__(name, category, metric_score_unit="ratio")

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        value = sum(1 for item in history if bool(getattr(item, "stop_line_violation", False)))
        statistics = [
            Statistic("violation_count", "count", MetricStatisticsType.COUNT, value),
            Statistic(self.name, "boolean", MetricStatisticsType.BOOLEAN, value == 0),
        ]
        return self._construct_metric_results(statistics, scenario=scenario)


class _EventMetric(AbstractMetric):
    event_name = "event"

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        count = _event_count(history, self.event_name)
        statistics = [
            Statistic(f"{self.event_name}_count", "count", MetricStatisticsType.COUNT, count),
            Statistic(self.name, "boolean", MetricStatisticsType.BOOLEAN, count == 0),
        ]
        return self._construct_metric_results(statistics, scenario=scenario)


class DrivableAreaMetric(_EventMetric):
    event_name = "drivable_area_violation"

    def __init__(self, name: str = "drivable_area", category: str = "safety") -> None:
        super().__init__(name, category, metric_score_unit="ratio")


class LaneDepartureMetric(_EventMetric):
    event_name = "lane_departure"

    def __init__(self, name: str = "lane_departure", category: str = "safety") -> None:
        super().__init__(name, category, metric_score_unit="ratio")


class TrafficLightMetric(_EventMetric):
    event_name = "traffic_light_violation"

    def __init__(self, name: str = "traffic_light", category: str = "safety") -> None:
        super().__init__(name, category, metric_score_unit="ratio")


class GoalReachedMetric(AbstractMetric):
    def __init__(
        self, radius_m: float = 2.0, name: str = "goal_reached", category: str = "progress"
    ) -> None:
        if radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        super().__init__(name, category, metric_score_unit="boolean")
        self.radius_m = float(radius_m)

    def compute_score(
        self,
        scenario: Any,
        metric_statistics: list[Statistic],
        time_series: TimeSeries | None = None,
    ) -> float:
        del scenario, time_series
        return 1.0 if metric_statistics and bool(metric_statistics[0].value) else 0.0

    def compute(self, history: Any, scenario: Any = None) -> list[MetricStatistics]:
        samples = list(history)
        goal = getattr(history, "mission_goal", None)
        if goal is None and scenario is not None:
            goal = getattr(scenario, "get_mission_goal", lambda: None)()
        reached = False
        if samples and goal is not None:
            target = np.asarray(goal, dtype=np.float64).reshape(-1)
            reached = (
                target.size >= 2
                and float(np.linalg.norm(_pose(samples[-1].ego_state)[:2] - target[:2]))
                <= self.radius_m
            )
        statistics = [Statistic(self.name, "boolean", MetricStatisticsType.BOOLEAN, reached)]
        return self._construct_metric_results(statistics, scenario=scenario)


def _pose(state: Any) -> np.ndarray:
    pose = getattr(state, "pose", None)
    if pose is not None:
        value = pose() if callable(pose) else pose
        return np.asarray(value, dtype=np.float64).reshape(-1)[:3]
    return np.asarray(getattr(state, "ego_pose", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)


def _heading(state: Any) -> float:
    return float(_pose(state)[2])


def _speed(state: Any) -> float:
    value = getattr(state, "speed_mps", None)
    if value is not None:
        return float(value)
    value = getattr(state, "speed", None)
    if value is not None:
        return float(value() if callable(value) else value)
    velocity = np.asarray(getattr(state, "ego_velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(velocity[:2]))


def _acceleration(state: Any) -> float:
    value = getattr(state, "acceleration_mps2", None)
    if value is not None:
        return float(value)
    acceleration = np.asarray(
        getattr(state, "ego_acceleration", [0.0, 0.0]), dtype=np.float64
    ).reshape(-1)
    return float(np.linalg.norm(acceleration[:2]))


def _ttc(sample: Any) -> float:
    value = getattr(sample, "ttc_s", getattr(sample, "min_ttc_s", float("inf")))
    return float(value)


def _event_count(history: Any, name: str) -> int:
    return sum(1 for sample in history if bool(getattr(sample, name, False)))


__all__ = [
    "AccelerationMetric",
    "CollisionMetric",
    "ComfortMetric",
    "DrivableAreaMetric",
    "ExpertHeadingErrorMetric",
    "ExpertL2ErrorMetric",
    "GoalReachedMetric",
    "JerkMetric",
    "LaneDepartureMetric",
    "ProgressMetric",
    "SpeedLimitMetric",
    "StopLineViolationMetric",
    "TTCMetric",
    "TrafficLightMetric",
    "YawRateMetric",
]
