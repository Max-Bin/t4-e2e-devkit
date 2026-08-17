"""Ground-truth future predictor backed by a T4 scenario."""

from __future__ import annotations

from typing import Any, Type

from t4_e2e_devkit.common.actor_state.agent import Agent
from t4_e2e_devkit.common.actor_state.state_representation import TimePoint
from t4_e2e_devkit.common.actor_state.waypoint import Waypoint
from t4_e2e_devkit.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from t4_e2e_devkit.planning.simulation.predictor.abstract_predictor import (
    AbstractPredictor,
    PredictorInitialization,
    PredictorInput,
)
from t4_e2e_devkit.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class LogFuturePredictor(AbstractPredictor):
    """Attach recorded future tracks to agents in the current observation."""

    requires_scenario = True

    def __init__(self, scenario: Any, future_trajectory_sampling: TrajectorySampling) -> None:
        self.scenario = scenario
        self.future_trajectory_sampling = future_trajectory_sampling

    def initialize(self, initialization: PredictorInitialization) -> None:
        del initialization

    def name(self) -> str:
        return type(self).__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def compute_predicted_trajectories(self, current_input: PredictorInput) -> DetectionsTracks:
        _, observation = current_input.history.current_state
        if not isinstance(observation, DetectionsTracks):
            raise TypeError(f"LogFuturePredictor expects DetectionsTracks, got {type(observation).__name__}")
        current_time_us = int(current_input.iteration.time_us)
        future = self.scenario.get_future_tracked_objects(
            iteration=current_input.iteration.index,
            time_horizon=float(self.future_trajectory_sampling.time_horizon),
            num_samples=int(self.future_trajectory_sampling.num_poses),
        )
        by_token: dict[str, list[Any]] = {}
        for detections in future:
            for agent in detections.tracked_objects.get_agents():
                token = str(agent.track_token or agent.metadata.token)
                if int(agent.metadata.timestamp_us) > current_time_us:
                    by_token.setdefault(token, []).append(agent)
        for agent in observation.tracked_objects.get_agents():
            token = str(agent.track_token or agent.metadata.token)
            agents = by_token.get(token, [])
            waypoints = [
                Waypoint(
                    time_point=TimePoint(int(item.metadata.timestamp_us)),
                    oriented_box=item.box,
                    velocity=item.velocity,
                )
                for item in agents
                if isinstance(item, Agent)
            ]
            agent.predictions = [PredictedTrajectory(1.0, waypoints)] if waypoints else None
        return observation


__all__ = ["LogFuturePredictor"]
