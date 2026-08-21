"""Deterministic constant-acceleration baseline planner."""

from __future__ import annotations

from typing import Any

import numpy as np

from t4_e2e_devkit.common.dataclasses import Trajectory
from t4_e2e_devkit.planning.simulation.observation.observation_type import DetectionsTracks
from t4_e2e_devkit.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class SimplePlanner(AbstractPlanner):
    """Plan a straight trajectory with bounded longitudinal acceleration."""

    def __init__(
        self,
        horizon_seconds: float,
        sampling_time: float,
        acceleration: float = 0.0,
        max_velocity: float = 5.0,
    ) -> None:
        if horizon_seconds <= 0.0 or sampling_time <= 0.0:
            raise ValueError("horizon_seconds and sampling_time must be positive")
        if max_velocity <= 0.0:
            raise ValueError("max_velocity must be positive")
        self.sampling = TrajectorySampling(
            num_poses=int(round(horizon_seconds / sampling_time)),
            interval_length=sampling_time,
        )
        self.acceleration = float(acceleration)
        self.max_velocity = float(max_velocity)

    def initialize(self, initialization: PlannerInitialization) -> None:
        del initialization

    def name(self) -> str:
        return type(self).__name__

    def observation_type(self) -> type:
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> Trajectory:
        ego_state, _ = current_input.history.current_state
        speed = _speed(ego_state)
        times = np.arange(1, int(self.sampling.num_poses) + 1, dtype=np.float64) * float(
            self.sampling.interval_length
        )
        displacement = speed * times + 0.5 * self.acceleration * times * times
        displacement = np.maximum.accumulate(np.maximum(0.0, displacement))
        return Trajectory(
            np.column_stack(
                (displacement, np.zeros_like(displacement), np.zeros_like(displacement))
            ).astype(np.float32),
            trajectory_sampling=self.sampling,
        )


ConstantVelocityPlanner = SimplePlanner


def _speed(state: Any) -> float:
    value = getattr(state, "speed_mps", None)
    if value is not None:
        return float(value)
    speed = getattr(state, "speed", None)
    if speed is not None:
        return float(speed() if callable(speed) else speed)
    velocity = np.asarray(getattr(state, "ego_velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(velocity[:2]))


__all__ = ["ConstantVelocityPlanner", "SimplePlanner"]
