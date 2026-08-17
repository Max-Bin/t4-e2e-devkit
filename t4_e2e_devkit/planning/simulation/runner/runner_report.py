"""Portable runner result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RunnerReport:
    succeeded: bool
    error_message: Optional[str]
    start_time: float
    end_time: Optional[float]
    planner_report: Optional[Any]
    scenario_name: str
    planner_name: str
    log_name: str

    @property
    def duration_s(self) -> Optional[float]:
        return None if self.end_time is None else float(self.end_time - self.start_time)

    @property
    def error(self) -> Optional[str]:
        return self.error_message

    def as_dict(self) -> dict[str, Any]:
        return {
            "succeeded": bool(self.succeeded),
            "error_message": self.error_message,
            "start_time": float(self.start_time),
            "end_time": None if self.end_time is None else float(self.end_time),
            "duration_s": self.duration_s,
            "planner_report": (
                None
                if self.planner_report is None
                else getattr(self.planner_report, "as_dict", lambda: self.planner_report)()
            ),
            "scenario_name": self.scenario_name,
            "planner_name": self.planner_name,
            "log_name": self.log_name,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunnerReport":
        return cls(
            succeeded=bool(value.get("succeeded", False)),
            error_message=value.get("error_message", value.get("error")),
            start_time=float(value.get("start_time", 0.0)),
            end_time=(None if value.get("end_time") is None else float(value["end_time"])),
            planner_report=value.get("planner_report"),
            scenario_name=str(value.get("scenario_name", "scenario")),
            planner_name=str(value.get("planner_name", "planner")),
            log_name=str(value.get("log_name", "t4")),
        )


__all__ = ["RunnerReport"]
