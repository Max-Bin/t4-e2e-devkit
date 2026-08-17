"""Portable metric files and keys."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .metric_result import MetricStatistics


@dataclass(frozen=True)
class MetricFileKey:
    metric_name: str
    log_name: str
    scenario_name: str
    scenario_type: str
    planner_name: str

    def as_dict(self) -> dict[str, str]:
        return {
            "metric_name": self.metric_name,
            "log_name": self.log_name,
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "planner_name": self.planner_name,
        }

    def serialize(self) -> dict[str, str]:
        return self.as_dict()

    @classmethod
    def deserialize(cls, value: dict[str, str]) -> "MetricFileKey":
        return cls(**{key: str(item) for key, item in value.items()})


@dataclass
class MetricFile:
    key: MetricFileKey
    metric_statistics: tuple[MetricStatistics, ...]

    def __init__(self, key: MetricFileKey, metric_statistics: Iterable[MetricStatistics]) -> None:
        self.key = key
        self.metric_statistics = tuple(metric_statistics)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "t4.metric-file.v1",
            "key": self.key.as_dict(),
            "metric_statistics": [item.serialize() for item in self.metric_statistics],
        }

    def serialize(self) -> dict[str, Any]:
        return self.as_dict()

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(output.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.as_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        return output

    @classmethod
    def deserialize(cls, value: dict[str, Any]) -> "MetricFile":
        if value.get("format", "t4.metric-file.v1") != "t4.metric-file.v1":
            raise ValueError("unsupported metric file format")
        return cls(
            MetricFileKey.deserialize(value["key"]),
            [MetricStatistics.deserialize(item) for item in value.get("metric_statistics", ())],
        )

    @classmethod
    def read(cls, path: str | Path) -> "MetricFile":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.deserialize(value)


__all__ = ["MetricFile", "MetricFileKey"]
