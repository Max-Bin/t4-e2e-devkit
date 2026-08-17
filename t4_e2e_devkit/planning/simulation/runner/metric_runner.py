"""Run metrics against a saved simulation log."""

from __future__ import annotations

import inspect
import time
from typing import Any

from t4_e2e_devkit.planning.simulation.runner.abstract_runner import AbstractRunner
from t4_e2e_devkit.planning.simulation.runner.runner_report import RunnerReport


class MetricRunner(AbstractRunner):
    def __init__(self, simulation_log: Any, metric_callback: Any) -> None:
        self._simulation_log = simulation_log
        self._metric_callback = metric_callback

    @property
    def scenario(self) -> Any:
        return self._simulation_log.scenario

    @property
    def planner(self) -> Any:
        return self._simulation_log.planner

    def run(self) -> RunnerReport:
        started = time.perf_counter()
        report = RunnerReport(
            succeeded=False,
            error_message=None,
            start_time=started,
            end_time=None,
            planner_report=None,
            scenario_name=str(getattr(self.scenario, "token", "scenario")),
            planner_name=_name(self.planner),
            log_name=str(getattr(self.scenario, "log_name", "t4")),
        )
        try:
            _run_metric_callback(
                self._metric_callback,
                type("Setup", (), {"scenario": self.scenario})(),
                self.planner,
                self._simulation_log.simulation_history,
            )
            report.succeeded = True
        except BaseException as error:
            report.error_message = f"{type(error).__name__}: {error}"
        report.end_time = time.perf_counter()
        return report


def _name(value: Any) -> str:
    name = getattr(value, "name", type(value).__name__)
    return str(name() if callable(name) else name)


def _run_metric_callback(callback: Any, setup: Any, planner: Any, history: Any) -> None:
    method = getattr(callback, "on_simulation_end", None)
    if method is None:
        raise TypeError("metric callback must implement on_simulation_end")
    parameters = tuple(inspect.signature(method).parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        method(setup, planner, history)
    elif len(parameters) >= 3:
        method(setup, planner, history)
    else:
        method(history)


__all__ = ["MetricRunner"]
