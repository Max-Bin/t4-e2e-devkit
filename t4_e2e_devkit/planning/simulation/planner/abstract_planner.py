"""Planner interface and its input containers.

Native devkit replacement for ``nuplan.planning.simulation.planner.
abstract_planner``.  The names and fields PDM binds to are unchanged; what is
dropped is nuPlan's ``SimulationHistoryBuffer`` and ``PlannerReport``, which
exist to serve nuPlan's simulation runner and its metric engine.

This is the *classical* planner interface -- a rule-based planner stepping
inside a simulation loop, which is what ``PDMClosedPlanner`` is.  It is not the
interface a learned T4 model implements: that one is
:class:`t4_e2e_devkit.agents.AbstractT4Agent`, which consumes a
:class:`~t4_e2e_devkit.common.dataclasses.T4AgentInput` and returns a
:class:`~t4_e2e_devkit.common.dataclasses.Trajectory`.  The two meet in the
evaluation stack: PDM-Closed produces the ego-progress denominator that the
agent's trajectory is scored against.

``SimulationHistoryBuffer`` is kept as a small structural protocol rather than a
concrete class, because PDM only ever reads ``current_state`` off it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

from t4_e2e_devkit.common.actor_state.ego_state import EgoState
from t4_e2e_devkit.common.actor_state.state_representation import StateSE2
from t4_e2e_devkit.common.maps.maps_datatypes import TrafficLightStatusData
from t4_e2e_devkit.planning.simulation.observation.observation_type import Observation
from t4_e2e_devkit.planning.simulation.simulation_iteration import SimulationIteration
from t4_e2e_devkit.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


@runtime_checkable
class SimulationHistoryBuffer(Protocol):
    """The slice of nuPlan's rolling history buffer that PDM actually reads."""

    @property
    def current_state(self) -> Tuple[EgoState, Observation]:
        """:return: the most recent (ego state, observation) pair."""
        ...


@dataclass(frozen=True)
class SimpleHistoryBuffer:
    """A one-frame history buffer.

    T4 evaluation is open-loop per window: the planner is handed the current
    state and the window's observation, not a rolling simulation history.  This
    satisfies :class:`SimulationHistoryBuffer` for that case.
    """

    ego_state: EgoState
    observation: Observation

    @property
    def current_state(self) -> Tuple[EgoState, Observation]:
        """:return: the (ego state, observation) pair this buffer holds."""
        return self.ego_state, self.observation


@dataclass(frozen=True)
class PlannerInitialization:
    """Data required to initialize a planner."""

    route_roadblock_ids: List[str]  # Roadblock ids comprising the goal route
    mission_goal: Optional[StateSE2]  # Mission goal, commonly not reachable in one scenario
    map_api: Any  # Map API; a T4 scene's map tensors, or an AbstractMap


@dataclass(frozen=True)
class PlannerInput:
    """Input to a planner for which a trajectory should be computed."""

    iteration: SimulationIteration  # Iteration and time in a simulation progress
    history: SimulationHistoryBuffer  # Buffer holding past observations and states
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None


class AbstractPlanner(abc.ABC):
    """Interface for a generic ego vehicle planner."""

    # Only oracle planners may set this; it cannot be used for submissions.
    requires_scenario: bool = False

    @abc.abstractmethod
    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Initialize the planner for a scenario.
        :param initialization: planner initialization class.
        """

    @abc.abstractmethod
    def name(self) -> str:
        """:return: string describing the name of the planner."""

    @abc.abstractmethod
    def observation_type(self) -> type:
        """:return: type of observation this planner consumes."""

    @abc.abstractmethod
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Compute the ego vehicle trajectory.
        :param current_input: planner input for this iteration.
        :return: trajectory the ego should follow.
        """

    def compute_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Public entry point; mirrors nuPlan's signature without its reporting.
        :param current_input: planner input for this iteration.
        :return: trajectory the ego should follow.
        """
        return self.compute_planner_trajectory(current_input)
