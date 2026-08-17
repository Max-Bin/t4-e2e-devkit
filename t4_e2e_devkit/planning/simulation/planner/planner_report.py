"""Planner runtime statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PlannerReport:
    compute_trajectory_runtimes: tuple[float, ...]

    def __init__(self, compute_trajectory_runtimes: Iterable[float] = ()) -> None:
        object.__setattr__(self, "compute_trajectory_runtimes", tuple(float(v) for v in compute_trajectory_runtimes))

    @property
    def mean_runtime_s(self) -> float:
        return float(np.mean(self.compute_trajectory_runtimes)) if self.compute_trajectory_runtimes else 0.0

    @property
    def max_runtime_s(self) -> float:
        return float(np.max(self.compute_trajectory_runtimes)) if self.compute_trajectory_runtimes else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "compute_trajectory_runtimes": list(self.compute_trajectory_runtimes),
            "mean_runtime_s": self.mean_runtime_s,
            "max_runtime_s": self.max_runtime_s,
        }


__all__ = ["PlannerReport"]
