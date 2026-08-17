"""Planner that follows the recorded future ego trajectory."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from t4_e2e_devkit.common.dataclasses import Trajectory
from t4_e2e_devkit.planning.simulation.observation.observation_type import DetectionsTracks
from t4_e2e_devkit.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class LogFuturePlanner(AbstractPlanner):
    """Return recorded future ego poses on a requested time grid."""

    requires_scenario = True

    def __init__(
        self,
        scenario: Any,
        num_poses: int,
        future_time_horizon: float,
    ) -> None:
        self.scenario = scenario
        self.sampling = TrajectorySampling(num_poses=num_poses, time_horizon=future_time_horizon)
        self._last_trajectory: Optional[Trajectory] = None

    def initialize(self, initialization: PlannerInitialization) -> None:
        del initialization

    def name(self) -> str:
        return type(self).__name__

    def observation_type(self) -> type:
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> Trajectory:
        current = self.scenario.get_ego_status_at_iteration(current_input.iteration.index)
        states = list(
            self.scenario.get_ego_future_trajectory(
                current_input.iteration.index,
                float(self.sampling.time_horizon),
                int(self.sampling.num_poses),
            )
        )
        if states and np.allclose(states[0].ego_pose, current.ego_pose):
            states = states[1:]
        if len(states) < int(self.sampling.num_poses):
            if self._last_trajectory is None:
                raise RuntimeError("scenario does not contain the requested future trajectory")
            return self._last_trajectory
        origin = np.asarray(current.ego_pose, dtype=np.float64)
        cos_heading = float(np.cos(origin[2]))
        sin_heading = float(np.sin(origin[2]))
        poses = []
        for state in states[: int(self.sampling.num_poses)]:
            delta = np.asarray(state.ego_pose[:2], dtype=np.float64) - origin[:2]
            local = np.array(
                [cos_heading * delta[0] + sin_heading * delta[1],
                 -sin_heading * delta[0] + cos_heading * delta[1]],
            )
            poses.append([local[0], local[1], float(state.ego_pose[2] - origin[2])])
        self._last_trajectory = Trajectory(
            np.asarray(poses, dtype=np.float32),
            trajectory_sampling=self.sampling,
        )
        return self._last_trajectory


__all__ = ["LogFuturePlanner"]
