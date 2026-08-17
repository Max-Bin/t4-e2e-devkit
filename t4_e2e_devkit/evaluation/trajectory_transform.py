"""Convert a sampled trajectory into simulator states.

The input may use any sampling carried by :class:`Trajectory`. The simulator
queries the resulting states on its own grid. The vehicle footprint comes from
the scene's ``ego_shape``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.actor_state.ego_state import EgoState
from t4_e2e_devkit.common.actor_state.state_representation import StateSE2, TimePoint
from t4_e2e_devkit.common.dataclasses import Trajectory
from t4_e2e_devkit.common.geometry.convert import relative_to_absolute_poses
from t4_e2e_devkit.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from t4_e2e_devkit.planning.simulation.planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)
from t4_e2e_devkit.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def transform_trajectory(
    pred_trajectory: Trajectory,
    initial_ego_state: EgoState,
) -> InterpolatedTrajectory:
    """
    Transform a trajectory from the ego frame into global coordinates.
    :param pred_trajectory: trajectory dataclass in the ego frame.
    :param initial_ego_state: ego state the trajectory starts from.
    :return: interpolated trajectory in global coordinates.
    """
    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(
        initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length
    )

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # Velocity and acceleration are left at zero: the LQR tracker and the
    # bicycle model re-derive them while following the path, so seeding them
    # here would be overwritten rather than respected.
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps, strict=True)
    ]

    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Resample an interpolated trajectory onto the scorer's fixed time grid.
    :param trajectory: trajectory in global coordinates.
    :param future_sampling: sampling parameters for the interpolation.
    :param start_time: time point the grid starts at.
    :return: array of interpolated trajectory states.
    """
    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)
