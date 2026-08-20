"""Adapter from a deployable T4 agent to the generic planner lifecycle."""

from __future__ import annotations

from typing import Optional

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent
from t4_e2e_devkit.planning.simulation.observation.replay import T4ReplayObservation
from t4_e2e_devkit.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)


class T4AgentPlanner(AbstractPlanner):
    """Use one :class:`AbstractT4Agent` inside ``SimulationRunner``.

    The observation source remains responsible for replaying sensors.  This
    adapter only converts the current replay observation back to the agent's
    non-privileged ``T4AgentInput`` and preserves the trajectory sampling the
    agent declared.
    """

    requires_scenario = False

    def __init__(self, agent: AbstractT4Agent) -> None:
        if not isinstance(agent, AbstractT4Agent):
            raise TypeError(f"expected AbstractT4Agent, got {type(agent).__name__}")
        if agent.requires_scene:
            raise ValueError("a deployable simulation planner cannot require privileged scene data")
        self.agent = agent
        self.initialization: Optional[PlannerInitialization] = None

    def initialize(self, initialization: PlannerInitialization) -> None:
        self.initialization = initialization
        self.agent.initialize()

    def name(self) -> str:
        value = self.agent.name
        return str(value() if callable(value) else value)

    def observation_type(self) -> type:
        return T4ReplayObservation

    def compute_planner_trajectory(self, current_input: PlannerInput):
        _, observation = current_input.history.current_state
        if not isinstance(observation, T4ReplayObservation):
            raise TypeError(
                f"T4AgentPlanner expects T4ReplayObservation, got {type(observation).__name__}"
            )
        return self.agent.compute_trajectory(observation.scene.get_agent_input())


__all__ = ["T4AgentPlanner"]
