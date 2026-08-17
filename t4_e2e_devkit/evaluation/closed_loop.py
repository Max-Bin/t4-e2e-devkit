"""Metrics for realized sensor-replay closed-loop rollouts.

The runner advances only the ego state. This module evaluates the resulting
rollout separately from proposal scoring and open-loop imitation. Event data
recorded by the replay runner is consumed automatically; explicit event
arguments remain available for custom rollout harnesses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from t4_e2e_devkit.planning.simulation.closed_loop import T4ClosedLoopResult


@dataclass(frozen=True)
class ClosedLoopMetricConfig:
    """Thresholds for rollout-level progress and stuck detection."""

    goal_radius_m: float = 2.0
    stuck_speed_mps: float = 0.5
    stuck_window_s: float = 2.0
    stuck_distance_m: float = 1.0

    def __post_init__(self) -> None:
        if self.goal_radius_m <= 0.0:
            raise ValueError("goal_radius_m must be positive")
        if self.stuck_speed_mps < 0.0:
            raise ValueError("stuck_speed_mps must be non-negative")
        if self.stuck_window_s <= 0.0:
            raise ValueError("stuck_window_s must be positive")
        if self.stuck_distance_m < 0.0:
            raise ValueError("stuck_distance_m must be non-negative")


@dataclass(frozen=True)
class ClosedLoopMetrics:
    """Per-rollout metrics over the realized ego states."""

    duration_s: float
    path_length_m: float
    final_displacement_m: float
    final_speed_mps: float
    max_speed_mps: float
    mean_abs_acceleration_mps2: float
    max_abs_acceleration_mps2: float
    max_abs_yaw_rate_radps: float
    stuck: float
    goal_reached: Optional[float] = None
    collision: Optional[float] = None
    first_collision_step: Optional[float] = None
    timeout: Optional[float] = None
    termination_reason: Optional[str] = None
    token: Optional[str] = None

    @property
    def values(self) -> Mapping[str, float]:
        """Return the available scalar metrics."""

        values = {
            "duration_s": self.duration_s,
            "path_length_m": self.path_length_m,
            "final_displacement_m": self.final_displacement_m,
            "final_speed_mps": self.final_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "mean_abs_acceleration_mps2": self.mean_abs_acceleration_mps2,
            "max_abs_acceleration_mps2": self.max_abs_acceleration_mps2,
            "max_abs_yaw_rate_radps": self.max_abs_yaw_rate_radps,
            "stuck": self.stuck,
        }
        optional = {
            "goal_reached": self.goal_reached,
            "collision": self.collision,
            "first_collision_step": self.first_collision_step,
            "timeout": self.timeout,
        }
        values.update({name: value for name, value in optional.items() if value is not None})
        return values


def compute_closed_loop_metrics(
    result: T4ClosedLoopResult,
    *,
    config: Optional[ClosedLoopMetricConfig] = None,
    goal_pose_world: Optional[np.ndarray] = None,
    collision_steps: Optional[Sequence[int]] = None,
    timeout: Optional[bool] = None,
    termination_reason: Optional[str] = None,
    token: Optional[str] = None,
) -> ClosedLoopMetrics:
    """Evaluate one realized rollout.

    Explicit event arguments override the events attached to ``result``. If no
    event source is available, the corresponding metric is omitted instead of
    reporting a false zero. A goal is interpreted by position; a 3-vector is
    ``(x, y, heading)`` and a 4-vector is ``(x, y, cos, sin)``.
    """

    config = config or ClosedLoopMetricConfig()
    if goal_pose_world is None:
        goal_pose_world = result.goal_pose_world
    if collision_steps is None:
        collision_steps = result.collision_steps
    if timeout is None:
        timeout = result.timeout
    if termination_reason is None:
        termination_reason = result.termination_reason
    states = result.states
    if len(states) < 2:
        raise ValueError("a closed-loop result needs an initial state and one realized step")

    poses = result.realized_poses_world.astype(np.float64)
    displacement = np.diff(poses[:, :2], axis=0)
    step_distances = np.linalg.norm(displacement, axis=-1)
    speeds = np.asarray([state.speed_mps for state in states], dtype=np.float64)
    accelerations = np.asarray(
        [state.acceleration_mps2 for state in states[1:]], dtype=np.float64
    )
    yaw_rates = np.asarray([state.yaw_rate_radps for state in states[1:]], dtype=np.float64)

    window_steps = max(1, int(math.ceil(config.stuck_window_s / result.dt_s)))
    recent_positions = poses[max(0, len(poses) - window_steps - 1) :, :2]
    recent_distance = float(np.linalg.norm(np.diff(recent_positions, axis=0), axis=-1).sum())
    stuck = float(
        recent_distance <= config.stuck_distance_m
        and float(np.max(speeds[-len(recent_positions) :])) <= config.stuck_speed_mps
    )

    goal_reached = None
    if goal_pose_world is not None:
        goal_xy = _goal_xy(goal_pose_world)
        goal_reached = float(np.linalg.norm(poses[-1, :2] - goal_xy) <= config.goal_radius_m)

    collision = None
    first_collision_step = None
    if collision_steps is not None:
        collision_indices = sorted(int(step) for step in collision_steps)
        collision = float(bool(collision_indices))
        first_collision_step = float(collision_indices[0]) if collision_indices else -1.0

    return ClosedLoopMetrics(
        duration_s=float((len(states) - 1) * result.dt_s),
        path_length_m=float(step_distances.sum()),
        final_displacement_m=float(np.linalg.norm(poses[-1, :2] - poses[0, :2])),
        final_speed_mps=float(speeds[-1]),
        max_speed_mps=float(np.max(speeds)),
        mean_abs_acceleration_mps2=float(np.mean(np.abs(accelerations))),
        max_abs_acceleration_mps2=float(np.max(np.abs(accelerations))),
        max_abs_yaw_rate_radps=float(np.max(np.abs(yaw_rates))),
        stuck=stuck,
        goal_reached=goal_reached,
        collision=collision,
        first_collision_step=first_collision_step,
        timeout=None if timeout is None else float(timeout),
        termination_reason=termination_reason,
        token=token,
    )


def aggregate_closed_loop_metrics(results: Sequence[ClosedLoopMetrics]) -> dict[str, float]:
    """Average rollout metrics while omitting unavailable event fields."""

    if not results:
        return {"num_rollouts": 0.0}
    names = sorted({name for result in results for name in result.values})
    report: dict[str, float] = {"num_rollouts": float(len(results))}
    for name in names:
        values = [result.values[name] for result in results if name in result.values]
        report[name] = float(np.mean(values))
    reasons = [result.termination_reason for result in results if result.termination_reason]
    for reason in sorted(set(reasons)):
        report[f"termination/{reason}"] = float(reasons.count(reason) / len(results))
    return report


def _goal_xy(goal_pose_world: np.ndarray) -> np.ndarray:
    goal = np.asarray(goal_pose_world, dtype=np.float64).reshape(-1)
    if goal.shape[0] not in (3, 4):
        raise ValueError("goal_pose_world must contain 3 or 4 values")
    return goal[:2]


__all__ = [
    "ClosedLoopMetricConfig",
    "ClosedLoopMetrics",
    "aggregate_closed_loop_metrics",
    "compute_closed_loop_metrics",
]
