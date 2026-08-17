"""Generic simulation lifecycle primitives for T4 scenarios.

This module is intentionally database-free.  A scenario, observation source,
planner and ego controller are explicit dependencies, which makes the runtime
usable with T4 replay data and with an outer launcher that supplies its own
scenario index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping, Optional, Protocol, Sequence

import numpy as np

from t4_e2e_devkit.common.actor_state.state_representation import TimePoint
from t4_e2e_devkit.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from t4_e2e_devkit.planning.simulation.simulation_iteration import SimulationIteration


class SimulationTimeController(ABC):
    """State machine controlling simulation timestamps."""

    @abstractmethod
    def reset(self) -> None:
        """Return to the first iteration."""

    @property
    @abstractmethod
    def current_iteration(self) -> SimulationIteration:
        """:return: current simulation iteration."""

    @property
    @abstractmethod
    def reached_end(self) -> bool:
        """:return: whether no further simulation step is available."""

    @abstractmethod
    def advance(self) -> SimulationIteration:
        """Advance and return the new current iteration."""


@dataclass
class StepSimulationTimeController(SimulationTimeController):
    """Fixed-rate simulation clock with an explicit finite horizon."""

    start_time_us: int
    interval_us: int
    num_iterations: int
    _index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.interval_us <= 0:
            raise ValueError("interval_us must be positive")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        self.reset()

    def reset(self) -> None:
        self._index = 0

    @property
    def current_iteration(self) -> SimulationIteration:
        return SimulationIteration(
            time_point=TimePoint(int(self.start_time_us + self._index * self.interval_us)),
            index=self._index,
        )

    @property
    def reached_end(self) -> bool:
        return self._index >= self.num_iterations

    def advance(self) -> SimulationIteration:
        if self._index < self.num_iterations:
            self._index += 1
        return self.current_iteration


@dataclass
class ReplaySimulationTimeController(SimulationTimeController):
    """Simulation clock driven by recorded timestamps."""

    timestamps_us: Sequence[int]
    _index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in self.timestamps_us)
        if not values:
            raise ValueError("timestamps_us must not be empty")
        if any(right < left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("timestamps_us must be non-decreasing")
        self.timestamps_us = values
        self.reset()

    def reset(self) -> None:
        self._index = 0

    @property
    def current_iteration(self) -> SimulationIteration:
        index = min(self._index, len(self.timestamps_us) - 1)
        return SimulationIteration(TimePoint(self.timestamps_us[index]), index)

    @property
    def reached_end(self) -> bool:
        return self._index >= len(self.timestamps_us)

    def advance(self) -> SimulationIteration:
        if self._index < len(self.timestamps_us):
            self._index += 1
        return self.current_iteration


@dataclass(frozen=True)
class SimulationHistorySample:
    """One completed simulation tick."""

    iteration: SimulationIteration
    ego_state: Any
    observation: Any
    trajectory: Any
    planner_report: Optional["PlannerReport"] = None


class SimulationHistory:
    """Ordered, serializable history of simulation samples."""

    def __init__(self, samples: Optional[Iterable[SimulationHistorySample]] = None) -> None:
        self._samples: list[SimulationHistorySample] = list(samples or ())

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, index: int) -> SimulationHistorySample:
        return self._samples[index]

    @property
    def samples(self) -> tuple[SimulationHistorySample, ...]:
        return tuple(self._samples)

    @property
    def last(self) -> Optional[SimulationHistorySample]:
        return self._samples[-1] if self._samples else None

    def append(self, sample: SimulationHistorySample) -> None:
        if self._samples and sample.iteration.index <= self._samples[-1].iteration.index:
            raise ValueError("simulation history iterations must increase strictly")
        self._samples.append(sample)

    def extend(self, samples: Iterable[SimulationHistorySample]) -> None:
        for sample in samples:
            self.append(sample)

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": [
                {
                    "iteration": {
                        "index": sample.iteration.index,
                        "time_us": sample.iteration.time_us,
                    },
                    "ego_state": _portable_value(sample.ego_state),
                    "observation": _portable_value(sample.observation),
                    "trajectory": _portable_value(sample.trajectory),
                    "planner_report": (
                        None
                        if sample.planner_report is None
                        else sample.planner_report.as_dict()
                    ),
                }
                for sample in self._samples
            ]
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationHistory":
        """Restore the portable representation without guessing domain types."""

        samples: list[SimulationHistorySample] = []
        for item in value.get("samples", []):
            iteration = item["iteration"]
            report = item.get("planner_report")
            planner_report = None
            if report is not None:
                planner_report = PlannerReport(
                    planner_name=str(report["planner_name"]),
                    compute_time_s=float(report["compute_time_s"]),
                    metadata=dict(report.get("metadata", {})),
                )
            samples.append(
                SimulationHistorySample(
                    iteration=SimulationIteration(
                        TimePoint(int(iteration["time_us"])), int(iteration["index"])
                    ),
                    ego_state=item.get("ego_state"),
                    observation=item.get("observation"),
                    trajectory=item.get("trajectory"),
                    planner_report=planner_report,
                )
            )
        return cls(samples)

    def to_json(self, path: str | Path) -> None:
        """Write a JSON history using only portable values."""

        import json
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationHistory":
        """Read a history written by :meth:`to_json`."""

        import json
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class SimulationHistoryBuffer:
    """Fixed-size rolling history passed to planners."""

    def __init__(
        self,
        buffer_size: int,
        *,
        initial_state: Any = None,
        initial_observation: Any = None,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self.buffer_size = int(buffer_size)
        self._states: Deque[Any] = deque(maxlen=self.buffer_size)
        self._observations: Deque[Any] = deque(maxlen=self.buffer_size)
        if initial_state is not None or initial_observation is not None:
            self._states.append(initial_state)
            self._observations.append(initial_observation)

    def __len__(self) -> int:
        return len(self._states)

    @property
    def current_state(self) -> tuple[Any, Any]:
        if not self._states:
            raise RuntimeError("simulation history buffer has no current state")
        return self._states[-1], self._observations[-1]

    @property
    def ego_states(self) -> tuple[Any, ...]:
        return tuple(self._states)

    @property
    def observations(self) -> tuple[Any, ...]:
        return tuple(self._observations)

    def append(self, ego_state: Any, observation: Any) -> None:
        self._states.append(ego_state)
        self._observations.append(observation)

    @classmethod
    def initialize_from_history(
        cls, history: SimulationHistory, buffer_size: int
    ) -> "SimulationHistoryBuffer":
        buffer = cls(buffer_size)
        for sample in history.samples[-buffer_size:]:
            buffer.append(sample.ego_state, sample.observation)
        return buffer


@dataclass(frozen=True)
class PlannerReport:
    """Structured planner timing and diagnostic output for one tick."""

    planner_name: str
    compute_time_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "planner_name": self.planner_name,
            "compute_time_s": float(self.compute_time_s),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SimulationRunReport:
    """Execution status kept separate from the portable simulation history."""

    history: SimulationHistory
    succeeded: bool
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "succeeded": bool(self.succeeded),
            "error": self.error,
            "metadata": dict(self.metadata),
            "history": self.history.as_dict(),
        }


class AbstractObservation(Protocol):
    """Observation source consumed by :class:`SimulationRunner`."""

    def reset(self) -> None:
        ...

    def get_observation(
        self, iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> Any:
        ...


class AbstractEgoController(Protocol):
    """Controller that realizes a planner trajectory into the next state."""

    def reset(self) -> None:
        ...

    def update_state(self, trajectory: Any, iteration: SimulationIteration) -> Any:
        ...


class SimulationCallback(Protocol):
    """Optional generic simulation lifecycle hooks."""

    def on_simulation_start(self, setup: "SimulationSetup") -> None:
        ...

    def on_simulation_step(self, sample: SimulationHistorySample) -> None:
        ...

    def on_simulation_end(self, history: SimulationHistory) -> None:
        ...

    def on_simulation_error(self, error: BaseException) -> None:
        ...


@dataclass(frozen=True)
class SimulationSetup:
    """All replaceable pieces required for one simulation run."""

    scenario: Any
    planner: Any
    observation: AbstractObservation
    ego_controller: AbstractEgoController
    time_controller: SimulationTimeController
    history_buffer_size: int = 1
    callbacks: tuple[SimulationCallback, ...] = ()
    planner_initialization: Optional[PlannerInitialization] = None

    def __post_init__(self) -> None:
        if self.history_buffer_size < 1:
            raise ValueError("history_buffer_size must be positive")


class KinematicEgoController:
    """Adapt the T4 kinematic controller protocol to generic simulation."""

    def __init__(self, controller: Optional[Any] = None, initial_state: Any = None) -> None:
        self.controller = controller
        self.state = initial_state

    def reset(self) -> None:
        self.state = None
        if self.controller is not None:
            reset = getattr(self.controller, "reset", None)
            if reset is not None:
                reset()

    def set_state(self, state: Any) -> None:
        self.state = state

    def update_state(self, trajectory: Any, iteration: SimulationIteration) -> Any:
        del iteration
        if self.controller is None:
            if self.state is None:
                raise RuntimeError("KinematicEgoController needs an initial state")
            return self.state
        if self.state is None:
            raise RuntimeError("KinematicEgoController needs an initial state")
        from t4_e2e_devkit.planning.simulation.closed_loop import KinematicState

        reference = getattr(trajectory, "poses", trajectory)
        if not isinstance(self.state, KinematicState):
            raise TypeError("the built-in kinematic controller requires KinematicState")
        self.state = self.controller.step(self.state, np.asarray(reference))
        return self.state


class LogPlaybackController:
    """Return recorded states in sequence, useful for open-loop replay."""

    def __init__(self, states: Sequence[Any]) -> None:
        if not states:
            raise ValueError("LogPlaybackController needs at least one state")
        self.states = tuple(states)
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def update_state(self, trajectory: Any, iteration: SimulationIteration) -> Any:
        del trajectory, iteration
        index = min(self._index + 1, len(self.states) - 1)
        self._index += 1
        return self.states[index]


class SimulationRunner:
    """Execute a generic planner/observation/controller simulation."""

    def run(self, setup: SimulationSetup) -> SimulationHistory:
        callbacks = tuple(setup.callbacks)
        history = SimulationHistory()
        controller = setup.ego_controller
        time_controller = setup.time_controller
        observation_source = setup.observation
        planner = setup.planner
        time_controller.reset()
        reset = getattr(controller, "reset", None)
        if reset is not None:
            reset()
        reset = getattr(observation_source, "reset", None)
        if reset is not None:
            reset()
        try:
            initialization = setup.planner_initialization or _planner_initialization(setup.scenario)
            initialize = getattr(planner, "initialize", None)
            if initialize is not None:
                initialize(initialization)
            for callback in callbacks:
                _call_optional(callback, "on_simulation_start", setup)

            first_observation = observation_source.get_observation(
                time_controller.current_iteration,
                SimulationHistoryBuffer(setup.history_buffer_size),
            )
            history_buffer = SimulationHistoryBuffer(
                setup.history_buffer_size,
                initial_state=_initial_state(setup.scenario),
                initial_observation=first_observation,
            )
            if isinstance(controller, KinematicEgoController) and controller.state is None:
                controller.set_state(_initial_state(setup.scenario))

            while not time_controller.reached_end:
                iteration = time_controller.current_iteration
                observation = (
                    first_observation
                    if iteration.index == 0
                    else observation_source.get_observation(iteration, history_buffer)
                )
                expected_observation = getattr(planner, "observation_type", None)
                if expected_observation is not None:
                    expected_type = (
                        expected_observation
                        if isinstance(expected_observation, type)
                        else expected_observation()
                        if callable(expected_observation)
                        else expected_observation
                    )
                    if expected_type is not None and not isinstance(observation, expected_type):
                        raise TypeError(
                            f"planner expects observation {expected_type}, "
                            f"got {type(observation)}"
                        )
                planner_input = PlannerInput(
                    iteration=iteration,
                    history=history_buffer,
                    traffic_light_data=None,
                )
                trajectory, report = _compute_planner_trajectory(planner, planner_input)
                ego_state = controller.update_state(trajectory, iteration)
                sample = SimulationHistorySample(
                    iteration=iteration,
                    ego_state=ego_state,
                    observation=observation,
                    trajectory=trajectory,
                    planner_report=report,
                )
                history.append(sample)
                history_buffer.append(ego_state, observation)
                for callback in callbacks:
                    _call_optional(callback, "on_simulation_step", sample)
                time_controller.advance()
            for callback in callbacks:
                _call_optional(callback, "on_simulation_end", history)
            return history
        except BaseException as error:
            for callback in callbacks:
                _call_optional(callback, "on_simulation_error", error)
            raise


class SimulationManager:
    """Run multiple independent :class:`SimulationSetup` objects in order."""

    def __init__(self, runner: Optional[SimulationRunner] = None) -> None:
        self.runner = runner or SimulationRunner()

    def run(self, setup: SimulationSetup) -> SimulationHistory:
        return self.runner.run(setup)

    def run_many(self, setups: Iterable[SimulationSetup]) -> list[SimulationHistory]:
        return [self.run(setup) for setup in setups]

    def run_with_report(self, setup: SimulationSetup) -> SimulationRunReport:
        """Run one setup and retain failures as a structured report."""

        try:
            history = self.run(setup)
        except BaseException as error:
            return SimulationRunReport(
                history=SimulationHistory(),
                succeeded=False,
                error=f"{type(error).__name__}: {error}",
            )
        return SimulationRunReport(history=history, succeeded=True)


def _compute_planner_trajectory(planner: Any, planner_input: PlannerInput) -> tuple[Any, PlannerReport]:
    import time

    started = time.perf_counter()
    compute = getattr(planner, "compute_planner_trajectory", None)
    if compute is None:
        compute = getattr(planner, "compute_trajectory", None)
    if compute is None:
        raise TypeError("planner must implement compute_planner_trajectory or compute_trajectory")
    trajectory = compute(planner_input)
    name = getattr(planner, "name", None)
    name = name() if callable(name) else (str(name) if name is not None else type(planner).__name__)
    return trajectory, PlannerReport(name, time.perf_counter() - started)


def _planner_initialization(scenario: Any) -> PlannerInitialization:
    route = getattr(scenario, "get_route_roadblock_ids", lambda: ())()
    goal = getattr(scenario, "get_mission_goal", lambda: None)()
    map_api = getattr(scenario, "get_map_api", lambda: None)()
    return PlannerInitialization(list(route), goal, map_api)


def _initial_state(scenario: Any) -> Any:
    state = getattr(scenario, "initial_ego_state", None)
    if state is not None:
        return state
    state = getattr(scenario, "initial_ego_status", None)
    if state is not None:
        return state
    raise ValueError("scenario must expose initial_ego_state or initial_ego_status")


def _call_optional(callback: Any, name: str, *args: Any) -> None:
    function = getattr(callback, name, None)
    if function is not None:
        function(*args)


def _portable_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _portable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_value(item) for item in value]
    if hasattr(value, "as_dict"):
        return _portable_value(value.as_dict())
    if hasattr(value, "__dict__"):
        return _portable_value(vars(value))
    return str(value)


__all__ = [
    "AbstractEgoController",
    "AbstractObservation",
    "KinematicEgoController",
    "LogPlaybackController",
    "PlannerReport",
    "ReplaySimulationTimeController",
    "SimulationCallback",
    "SimulationHistory",
    "SimulationHistoryBuffer",
    "SimulationHistorySample",
    "SimulationManager",
    "SimulationRunner",
    "SimulationRunReport",
    "SimulationSetup",
    "SimulationTimeController",
    "StepSimulationTimeController",
]
