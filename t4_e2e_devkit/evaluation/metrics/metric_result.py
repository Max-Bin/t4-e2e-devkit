"""NuPlan-shaped metric result values without parquet/database requirements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union


class MetricStatisticsType(Enum):
    MAX = "MAX"
    MIN = "MIN"
    P90 = "P90"
    MEAN = "MEAN"
    VALUE = "VALUE"
    VELOCITY = "VELOCITY"
    BOOLEAN = "BOOLEAN"
    RATIO = "RATIO"
    COUNT = "COUNT"

    def __str__(self) -> str:
        return self.value

    @property
    def unit(self) -> str:
        if self is MetricStatisticsType.BOOLEAN:
            return "boolean"
        if self is MetricStatisticsType.RATIO:
            return "ratio"
        if self is MetricStatisticsType.COUNT:
            return "count"
        raise ValueError(f"{self.value} has no default unit")

    def serialize(self) -> str:
        return self.value

    @classmethod
    def deserialize(cls, value: str) -> "MetricStatisticsType":
        if isinstance(value, cls):
            return value
        return cls[str(value)]


@dataclass(frozen=True)
class Statistic:
    name: str
    unit: str
    type: MetricStatisticsType
    value: Union[float, bool]

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("statistic name must not be empty")
        if isinstance(self.type, str):
            object.__setattr__(self, "type", MetricStatisticsType.deserialize(self.type))
        if not isinstance(self.value, (bool, int, float)):
            raise TypeError("statistic values must be numeric or boolean")
        if not isinstance(self.value, bool):
            object.__setattr__(self, "value", float(self.value))

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "type": self.type.serialize(),
        }

    @classmethod
    def deserialize(cls, value: dict[str, Any]) -> "Statistic":
        return cls(
            name=str(value["name"]),
            unit=str(value["unit"]),
            value=value["value"],
            type=MetricStatisticsType.deserialize(str(value["type"])),
        )


@dataclass(frozen=True)
class TimeSeries:
    unit: str
    time_stamps: tuple[int, ...]
    values: tuple[float, ...]
    selected_frames: Optional[tuple[int, ...]] = None

    def __init__(
        self,
        unit: str,
        time_stamps: list[int] | tuple[int, ...],
        values: list[float] | tuple[float, ...],
        selected_frames: Optional[list[int] | tuple[int, ...]] = None,
    ) -> None:
        timestamps = tuple(int(value) for value in time_stamps)
        samples = tuple(float(value) for value in values)
        if len(timestamps) != len(samples):
            raise ValueError("time_stamps and values must have equal lengths")
        if any(right < left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("time_stamps must be non-decreasing")
        frames = None if selected_frames is None else tuple(int(value) for value in selected_frames)
        if frames is not None and len(frames) != len(samples):
            raise ValueError("selected_frames must align with values")
        object.__setattr__(self, "unit", str(unit))
        object.__setattr__(self, "time_stamps", timestamps)
        object.__setattr__(self, "values", samples)
        object.__setattr__(self, "selected_frames", frames)

    @property
    def timestamps_us(self) -> tuple[int, ...]:
        return self.time_stamps

    def serialize(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "time_stamps": list(self.time_stamps),
            "values": list(self.values),
            "selected_frames": None if self.selected_frames is None else list(self.selected_frames),
        }

    @classmethod
    def deserialize(cls, value: Optional[dict[str, Any]]) -> Optional["TimeSeries"]:
        if value is None:
            return None
        return cls(
            unit=str(value["unit"]),
            time_stamps=value["time_stamps"],
            values=value["values"],
            selected_frames=value.get("selected_frames"),
        )


@dataclass(frozen=True)
class MetricResult:
    metric_computator: str
    name: str
    metric_category: str

    def serialize(self) -> dict[str, Any]:
        return {
            "metric_computator": self.metric_computator,
            "name": self.name,
            "metric_category": self.metric_category,
        }

    @classmethod
    def deserialize(cls, value: dict[str, Any]) -> "MetricResult":
        """Restore the concrete typed result represented by ``value``."""

        if "statistics" in value:
            return MetricStatistics.deserialize(value)
        if "start_timestamp" in value:
            return MetricViolation.deserialize(value)
        return cls(
            metric_computator=str(value["metric_computator"]),
            name=str(value["name"]),
            metric_category=str(value["metric_category"]),
        )

    def serialize_dataframe(self) -> dict[str, Any]:
        """Return one row in the local metric dataframe format."""

        return self.serialize()


@dataclass(frozen=True)
class MetricStatistics(MetricResult):
    statistics: tuple[Statistic, ...]
    time_series: Optional[TimeSeries] = None
    metric_score: Optional[float] = None
    metric_score_unit: Optional[str] = None

    def __init__(
        self,
        metric_computator: str,
        name: str,
        metric_category: str,
        statistics: list[Statistic] | tuple[Statistic, ...],
        time_series: Optional[TimeSeries] = None,
        metric_score: Optional[float] = None,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "metric_computator", str(metric_computator))
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "metric_category", str(metric_category))
        object.__setattr__(self, "statistics", tuple(statistics))
        object.__setattr__(self, "time_series", time_series)
        object.__setattr__(self, "metric_score", None if metric_score is None else float(metric_score))
        object.__setattr__(self, "metric_score_unit", metric_score_unit)

    def serialize(self) -> dict[str, Any]:
        value = super().serialize()
        value.update(
            {
                "statistics": [statistic.serialize() for statistic in self.statistics],
                "time_series": None if self.time_series is None else self.time_series.serialize(),
                "metric_score": self.metric_score,
                "metric_score_unit": self.metric_score_unit,
            }
        )
        return value

    def serialize_dataframe(self) -> dict[str, Any]:
        columns: dict[str, Any] = {
            "metric_computator": self.metric_computator,
            "metric_statistics_name": self.name,
            "metric_category": self.metric_category,
            "metric_score": self.metric_score,
            "metric_score_unit": self.metric_score_unit,
        }
        for statistic in self.statistics:
            columns.update(
                {
                    f"{statistic.name}_stat_type": statistic.type.serialize(),
                    f"{statistic.name}_stat_unit": statistic.unit,
                    f"{statistic.name}_stat_value": statistic.value,
                }
            )
        series = self.time_series
        columns.update(
            {
                "time_series_unit": None if series is None else series.unit,
                "time_series_timestamps": None if series is None else list(series.time_stamps),
                "time_series_values": None if series is None else list(series.values),
                "time_series_selected_frames": None
                if series is None or series.selected_frames is None
                else list(series.selected_frames),
            }
        )
        return columns

    @classmethod
    def deserialize(cls, value: dict[str, Any]) -> "MetricStatistics":
        return cls(
            metric_computator=str(value["metric_computator"]),
            name=str(value["name"]),
            metric_category=str(value["metric_category"]),
            statistics=[Statistic.deserialize(item) for item in value.get("statistics", [])],
            time_series=TimeSeries.deserialize(value.get("time_series")),
            metric_score=value.get("metric_score"),
            metric_score_unit=value.get("metric_score_unit"),
        )


@dataclass(frozen=True)
class MetricViolation(MetricResult):
    unit: str
    start_timestamp: int
    duration: int
    extremum: float
    mean: float

    def serialize(self) -> dict[str, Any]:
        value = super().serialize()
        value.update(
            {
                "unit": self.unit,
                "start_timestamp": int(self.start_timestamp),
                "duration": int(self.duration),
                "extremum": float(self.extremum),
                "mean": float(self.mean),
            }
        )
        return value

    @classmethod
    def deserialize(cls, value: dict[str, Any]) -> "MetricViolation":
        return cls(
            metric_computator=str(value["metric_computator"]),
            name=str(value["name"]),
            metric_category=str(value["metric_category"]),
            unit=str(value["unit"]),
            start_timestamp=int(value["start_timestamp"]),
            duration=int(value["duration"]),
            extremum=float(value["extremum"]),
            mean=float(value["mean"]),
        )


__all__ = [
    "MetricResult",
    "MetricStatistics",
    "MetricStatisticsType",
    "MetricViolation",
    "Statistic",
    "TimeSeries",
]
