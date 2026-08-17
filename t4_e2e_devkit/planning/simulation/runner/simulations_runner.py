"""NuPlan-shaped runner around the T4 stateful simulation."""

from __future__ import annotations

import time
import traceback
from typing import Any, Iterable, Optional

from t4_e2e_devkit.planning.simulation.runner.abstract_runner import AbstractRunner
from t4_e2e_devkit.planning.simulation.runner.runner_report import RunnerReport
from t4_e2e_devkit.planning.simulation.runtime import PlannerReport
from t4_e2e_devkit.planning.simulation.simulation import Simulation


class SimulationRunner(AbstractRunner):
    """Execute one :class:`Simulation` with one planner."""

    def __init__(self, simulation: Simulation, planner: Optional[Any] = None) -> None:
        self._simulation = simulation
        self._planner = simulation.setup.planner if planner is None else planner

    @property
    def simulation(self) -> Simulation:
        return self._simulation

    @property
    def planner(self) -> Any:
        return self._planner

    @property
    def scenario(self) -> Any:
        return self._simulation.scenario

    def run(self) -> RunnerReport:
        started = time.perf_counter()
        setup = self._simulation.setup
        callback = self._simulation.callback
        report = RunnerReport(
            succeeded=False,
            error_message=None,
            start_time=started,
            end_time=None,
            planner_report=None,
            scenario_name=_scenario_name(self.scenario),
            planner_name=_name(self.planner),
            log_name=_log_name(self.scenario),
        )
        try:
            _call(callback, "on_simulation_start", setup)
            _call(callback, "on_initialization_start", setup, self.planner)
            initialization = self._simulation.initialize()
            initialize = getattr(self.planner, "initialize", None)
            if initialize is not None:
                initialize(initialization)
            _call(callback, "on_initialization_end", setup, self.planner)
            while self._simulation.is_simulation_running():
                _call(callback, "on_step_start", setup, self.planner)
                planner_input = self._simulation.get_planner_input()
                _call(callback, "on_planner_start", setup, self.planner)
                trajectory, planner_report = _compute(self.planner, planner_input)
                _call(callback, "on_planner_end", setup, self.planner, trajectory)
                self._simulation.propagate(trajectory, planner_report)
                _call(
                    callback,
                    "on_step_end",
                    setup,
                    self.planner,
                    self._simulation.history.last(),
                )
            _call(callback, "on_simulation_end", setup, self.planner, self._simulation.history)
            report.succeeded = True
            report.planner_report = _planner_report(self.planner)
        except BaseException as error:
            report.error_message = traceback.format_exc()
            try:
                _call(callback, "on_simulation_error", setup, self.planner, error)
            except BaseException:
                pass
        report.end_time = time.perf_counter()
        return report


class SimulationsRunner:
    """Run a sequence of independent runner instances."""

    def __init__(self, runners: Iterable[AbstractRunner]) -> None:
        self.runners = list(runners)

    def run(self) -> list[RunnerReport]:
        return [run_simulation(runner) for runner in self.runners]


BatchSimulationRunner = SimulationsRunner


def run_simulation(runner: AbstractRunner, *, raise_on_error: bool = False) -> RunnerReport:
    report = runner.run()
    if raise_on_error and not report.succeeded:
        raise RuntimeError(report.error_message or "simulation failed")
    return report


def execute_runners(runners: Iterable[AbstractRunner], *, raise_on_error: bool = False) -> list[RunnerReport]:
    """Execute runners serially.

    Use :class:`RunnerExecutor` when a batch needs local workers or rank
    partitioning; keeping this helper serial makes debugging deterministic.
    """

    return [run_simulation(runner, raise_on_error=raise_on_error) for runner in runners]


def _compute(planner: Any, planner_input: Any) -> tuple[Any, PlannerReport]:
    started = time.perf_counter()
    compute = getattr(planner, "compute_trajectory", None)
    if compute is None:
        compute = getattr(planner, "compute_planner_trajectory", None)
    if compute is None:
        raise TypeError("planner must implement compute_trajectory")
    trajectory = compute(planner_input)
    return trajectory, PlannerReport(_name(planner), time.perf_counter() - started)


def _planner_report(planner: Any) -> Any:
    generate = getattr(planner, "generate_planner_report", None)
    return None if generate is None else generate()


def _name(value: Any) -> str:
    name = getattr(value, "name", type(value).__name__)
    return str(name() if callable(name) else name)


def _scenario_name(scenario: Any) -> str:
    return str(getattr(scenario, "scenario_name", getattr(scenario, "token", "scenario")))


def _log_name(scenario: Any) -> str:
    return str(getattr(scenario, "log_name", getattr(scenario, "scene_dir", "t4")))


def _call(callback: Any, name: str, *args: Any) -> None:
    method = getattr(callback, name, None)
    if method is not None:
        method(*args)


__all__ = [
    "BatchSimulationRunner",
    "SimulationRunner",
    "SimulationsRunner",
    "execute_runners",
    "run_simulation",
]
