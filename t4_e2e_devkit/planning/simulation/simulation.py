"""Stateful, step-by-step T4 simulation lifecycle."""

from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np

from t4_e2e_devkit.planning.simulation.callback.multi_callback import MultiCallback
from t4_e2e_devkit.planning.simulation.runtime import (
    SimulationHistory,
    SimulationHistoryBuffer,
    SimulationHistorySample,
    SimulationSetup,
    _initial_state,
    _planner_initialization,
)


class Simulation:
    """Query observations, accept planner trajectories and propagate one tick.

    The class is deliberately independent of a database.  The setup supplies a
    T4 scenario, replay observation source, time controller and ego controller.
    ``SimulationRunner`` is a convenience loop around this state machine.
    """

    def __init__(
        self,
        simulation_setup: SimulationSetup,
        callback: Optional[Any] = None,
        simulation_history_buffer_duration: Optional[float] = None,
    ) -> None:
        self._setup = simulation_setup
        self._time_controller = simulation_setup.time_controller
        self._ego_controller = simulation_setup.ego_controller
        self._observations = simulation_setup.observation
        self._scenario = simulation_setup.scenario
        self._callback = callback or MultiCallback(simulation_setup.callbacks)
        self._history = SimulationHistory(
            map_api=getattr(self._scenario, "get_map_api", lambda: None)(),
            mission_goal=getattr(self._scenario, "get_mission_goal", lambda: None)(),
        )
        self._buffer_duration = simulation_history_buffer_duration
        self._history_buffer: Optional[SimulationHistoryBuffer] = None
        self._is_simulation_running = True
        self._first_observation: Any = None

    def __reduce__(self):
        return self.__class__, (self._setup, self._callback, self._buffer_duration)

    @property
    def setup(self) -> SimulationSetup:
        return self._setup

    @property
    def scenario(self) -> Any:
        return self._scenario

    @property
    def callback(self) -> Any:
        return self._callback

    @property
    def history(self) -> SimulationHistory:
        return self._history

    @property
    def history_buffer(self) -> SimulationHistoryBuffer:
        if self._history_buffer is None:
            raise RuntimeError("simulation must be initialized before reading history_buffer")
        return self._history_buffer

    def is_simulation_running(self) -> bool:
        return self._is_simulation_running and not self._time_controller.reached_end

    def reset(self) -> None:
        self._setup.reset()
        self._history.reset()
        self._history_buffer = None
        self._first_observation = None
        self._is_simulation_running = True

    def initialize(self) -> Any:
        """Reset the setup and seed the rolling history buffer."""

        self.reset()
        initialize = getattr(self._observations, "initialize", None)
        if initialize is not None:
            initialize()
        iteration = self._time_controller.current_iteration
        empty_buffer = SimulationHistoryBuffer(
            self._setup.history_buffer_size,
            sample_interval=_interval_seconds(self._time_controller),
        )
        self._first_observation = self._observations.get_observation(iteration, empty_buffer)
        self._history_buffer = SimulationHistoryBuffer(
            self._history_buffer_size(),
            initial_state=_initial_state(self._scenario),
            initial_observation=self._first_observation,
            sample_interval=_interval_seconds(self._time_controller),
        )
        if hasattr(self._ego_controller, "set_state") and getattr(self._ego_controller, "state", None) is None:
            self._ego_controller.set_state(_initial_state(self._scenario))
        return self._setup.planner_initialization or _planner_initialization(self._scenario)

    def get_planner_input(self):
        if self._history_buffer is None:
            raise RuntimeError("simulation was not initialized")
        if not self.is_simulation_running():
            raise RuntimeError("simulation is not running")
        from t4_e2e_devkit.planning.simulation.planner.abstract_planner import PlannerInput

        iteration = self._time_controller.current_iteration
        traffic_lights = _traffic_lights(self._scenario, iteration.index)
        return PlannerInput(
            iteration=iteration,
            history=self._history_buffer,
            traffic_light_data=list(traffic_lights),
        )

    def propagate(self, trajectory: Any, planner_report: Any = None) -> None:
        """Record the current state and advance to the next iteration."""

        if self._history_buffer is None:
            raise RuntimeError("simulation was not initialized")
        if not self.is_simulation_running():
            raise RuntimeError("simulation is not running")
        iteration = self._time_controller.current_iteration
        ego_state, observation = self._history_buffer.current_state
        traffic_lights = _traffic_lights(self._scenario, iteration.index)
        sample = SimulationHistorySample(
            iteration=iteration,
            ego_state=ego_state,
            observation=observation,
            trajectory=trajectory,
            planner_report=planner_report,
            traffic_light_status=traffic_lights,
        )
        self._history.add_sample(sample)
        next_iteration = self._time_controller.advance()
        if self._time_controller.reached_end:
            self._is_simulation_running = False
            return
        _update_controller(
            self._ego_controller,
            iteration,
            next_iteration,
            ego_state,
            trajectory,
        )
        next_observation = self._observations.get_observation(next_iteration, self._history_buffer)
        self._history_buffer.append(_controller_state(self._ego_controller), next_observation)

    def _history_buffer_size(self) -> int:
        if self._buffer_duration is None:
            return self._setup.history_buffer_size
        interval = _interval_seconds(self._time_controller)
        if interval is None:
            raise ValueError("a buffer duration requires a fixed-rate time controller")
        if self._buffer_duration < interval:
            raise ValueError("simulation history buffer duration must cover one interval")
        return max(self._setup.history_buffer_size, int(self._buffer_duration / interval) + 1)


def _interval_seconds(controller: Any) -> Optional[float]:
    interval_us = getattr(controller, "interval_us", None)
    if interval_us is not None:
        return float(interval_us) * 1.0e-6
    timestamps = getattr(controller, "timestamps_us", None)
    if timestamps is not None and len(timestamps) > 1:
        intervals = np.diff(np.asarray(timestamps, dtype=np.int64))
        if len(intervals) and np.all(intervals == intervals[0]) and intervals[0] > 0:
            return float(intervals[0]) * 1.0e-6
    return None


def _traffic_lights(scenario: Any, index: int) -> tuple[Any, ...]:
    getter = getattr(scenario, "get_traffic_light_status_at_iteration", None)
    if getter is None:
        return ()
    value = getter(index)
    return () if value is None else tuple(value)


def _controller_state(controller: Any) -> Any:
    state = getattr(controller, "state", None)
    if state is not None:
        return state
    get_state = getattr(controller, "get_state", None)
    if get_state is not None:
        return get_state()
    raise RuntimeError("ego controller exposes neither state nor get_state()")


def _update_controller(controller: Any, current: Any, next_iteration: Any, ego_state: Any, trajectory: Any) -> None:
    update = getattr(controller, "update_state", None)
    if update is None:
        raise TypeError("ego controller must implement update_state")
    parameters = list(inspect.signature(update).parameters.values())
    if len(parameters) >= 4:
        update(current, next_iteration, ego_state, trajectory)
    else:
        update(trajectory, next_iteration)


__all__ = ["Simulation"]
