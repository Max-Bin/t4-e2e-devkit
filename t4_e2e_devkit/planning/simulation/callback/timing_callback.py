"""Dependency-free timing callback."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from .abstract_callback import AbstractCallback


class TimingCallback(AbstractCallback):
    """Collect simulation and planner timings in plain Python mappings."""

    def __init__(self) -> None:
        self.simulations: dict[str, dict[str, float]] = {}
        self._simulation_start = 0.0
        self._step_start = 0.0
        self._planner_start = 0.0
        self._step_durations: list[float] = []
        self._planner_durations: list[float] = []
        self._steps_by_token: defaultdict[str, int] = defaultdict(int)

    def on_simulation_start(self, setup: Any) -> None:
        self._simulation_start = time.perf_counter()
        self._step_durations.clear()
        self._planner_durations.clear()
        token = _token(setup)
        self._steps_by_token[token] = 0

    def on_step_start(self, setup: Any, planner: Any) -> None:
        del setup, planner
        self._step_start = time.perf_counter()

    def on_planner_start(self, setup: Any, planner: Any) -> None:
        del setup, planner
        self._planner_start = time.perf_counter()

    def on_planner_end(self, setup: Any, planner: Any, trajectory: Any) -> None:
        del setup, planner, trajectory
        if self._planner_start:
            self._planner_durations.append(time.perf_counter() - self._planner_start)

    def on_step_end(self, setup: Any, planner: Any, sample: Any) -> None:
        del planner, sample
        if self._step_start:
            self._step_durations.append(time.perf_counter() - self._step_start)
        self._steps_by_token[_token(setup)] += 1

    def on_simulation_end(self, setup: Any, planner: Any, history: Any) -> None:
        del planner, history
        elapsed = time.perf_counter() - self._simulation_start
        self.simulations[_token(setup)] = {
            "simulation_elapsed_s": float(elapsed),
            "num_steps": float(self._steps_by_token[_token(setup)]),
            "mean_step_s": _mean(self._step_durations),
            "max_step_s": _max(self._step_durations),
            "mean_planner_s": _mean(self._planner_durations),
            "max_planner_s": _max(self._planner_durations),
        }


def _token(setup: Any) -> str:
    scenario = getattr(setup, "scenario", None)
    return str(getattr(scenario, "token", getattr(scenario, "scene_token", "simulation")))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


__all__ = ["TimingCallback"]
