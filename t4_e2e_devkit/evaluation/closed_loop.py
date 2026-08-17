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
class ClosedLoopTrace:
    """Per-step values from one realized closed-loop rollout.

    A trace row describes the realized state after the action taken for the
    corresponding ``source_frame``.  It is intentionally independent from
    the metric aggregate so a report can be recomputed or inspected without
    rerunning the agent.
    """

    step: np.ndarray
    source_frames: np.ndarray
    time_s: np.ndarray
    poses_world: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    yaw_rate_radps: np.ndarray
    steering_rad: np.ndarray
    step_distance_m: np.ndarray
    path_length_m: np.ndarray
    goal_distance_m: Optional[np.ndarray] = None
    collision: Optional[np.ndarray] = None
    plan_available: Optional[np.ndarray] = None
    plan_num_poses: Optional[np.ndarray] = None
    plan_interval_s: Optional[np.ndarray] = None
    agent_count: Optional[np.ndarray] = None
    min_agent_clearance_m: Optional[np.ndarray] = None
    ttc_s: Optional[np.ndarray] = None
    ttc_violation: Optional[np.ndarray] = None
    drivable_violation: Optional[np.ndarray] = None
    road_border_violation: Optional[np.ndarray] = None
    road_border_distance_m: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Normalize arrays and reject misaligned traces early."""

        length = int(np.asarray(self.step).reshape(-1).shape[0])
        required = (
            "step",
            "source_frames",
            "time_s",
            "speed_mps",
            "acceleration_mps2",
            "yaw_rate_radps",
            "steering_rad",
            "step_distance_m",
            "path_length_m",
        )
        for name in required:
            values = np.asarray(getattr(self, name)).reshape(-1)
            if len(values) != length:
                raise ValueError(
                    f"closed-loop trace field {name!r} has {len(values)} rows; "
                    f"expected {length}"
                )
            object.__setattr__(self, name, np.ascontiguousarray(values))

        poses = np.asarray(self.poses_world, dtype=np.float64)
        if poses.shape != (length, 3):
            raise ValueError(
                "closed-loop trace poses_world must have shape "
                f"({length}, 3), got {poses.shape}"
            )
        object.__setattr__(self, "poses_world", np.ascontiguousarray(poses))
        object.__setattr__(self, "step", np.asarray(self.step, dtype=np.int64).reshape(-1))
        object.__setattr__(
            self,
            "source_frames",
            np.asarray(self.source_frames, dtype=np.int64).reshape(-1),
        )
        for name in ("time_s", "speed_mps", "acceleration_mps2", "yaw_rate_radps",
                     "steering_rad", "step_distance_m", "path_length_m"):
            object.__setattr__(
                self,
                name,
                np.asarray(getattr(self, name), dtype=np.float64).reshape(-1),
            )

        for name, dtype in (
            ("goal_distance_m", np.float64),
            ("collision", bool),
            ("plan_available", bool),
            ("plan_num_poses", np.int64),
            ("plan_interval_s", np.float64),
            ("agent_count", np.float64),
            ("min_agent_clearance_m", np.float64),
            ("ttc_s", np.float64),
            ("ttc_violation", np.float64),
            ("drivable_violation", np.float64),
            ("road_border_violation", np.float64),
            ("road_border_distance_m", np.float64),
        ):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.asarray(values, dtype=dtype).reshape(-1)
            if len(values) != length:
                raise ValueError(
                    f"closed-loop trace field {name!r} has {len(values)} rows; "
                    f"expected {length}"
                )
            object.__setattr__(self, name, np.ascontiguousarray(values))

    def rows(self, token: Optional[str] = None) -> list[dict[str, object]]:
        """Return JSON/CSV-friendly rows for this trace."""

        rows: list[dict[str, object]] = []
        for index in range(len(self.step)):
            row: dict[str, object] = {
                "token": token or "",
                "step": int(self.step[index]),
                "source_frame": int(self.source_frames[index]),
                "time_s": float(self.time_s[index]),
                "x": float(self.poses_world[index, 0]),
                "y": float(self.poses_world[index, 1]),
                "heading": float(self.poses_world[index, 2]),
                "speed_mps": float(self.speed_mps[index]),
                "acceleration_mps2": float(self.acceleration_mps2[index]),
                "yaw_rate_radps": float(self.yaw_rate_radps[index]),
                "steering_rad": float(self.steering_rad[index]),
                "step_distance_m": float(self.step_distance_m[index]),
                "path_length_m": float(self.path_length_m[index]),
            }
            optional = {
                "goal_distance_m": self.goal_distance_m,
                "collision": self.collision,
                "plan_available": self.plan_available,
                "plan_num_poses": self.plan_num_poses,
                "plan_interval_s": self.plan_interval_s,
                "agent_count": self.agent_count,
                "min_agent_clearance_m": self.min_agent_clearance_m,
                "ttc_s": self.ttc_s,
                "ttc_violation": self.ttc_violation,
                "drivable_violation": self.drivable_violation,
                "road_border_violation": self.road_border_violation,
                "road_border_distance_m": self.road_border_distance_m,
            }
            for name, values in optional.items():
                if values is not None:
                    value = values[index]
                    if values.dtype == bool:
                        row[name] = bool(value)
                    elif np.issubdtype(values.dtype, np.number) and not np.isfinite(value):
                        row[name] = None
                    else:
                        row[name] = value.item()
            rows.append(row)
        return rows


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
    min_agent_clearance_m: Optional[float] = None
    min_ttc_s: Optional[float] = None
    ttc_violation: Optional[float] = None
    drivable_violation: Optional[float] = None
    road_border_violation: Optional[float] = None
    min_road_border_distance_m: Optional[float] = None
    termination_reason: Optional[str] = None
    token: Optional[str] = None
    trace: Optional[ClosedLoopTrace] = None

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
            "min_agent_clearance_m": self.min_agent_clearance_m,
            "min_ttc_s": self.min_ttc_s,
            "ttc_violation": self.ttc_violation,
            "drivable_violation": self.drivable_violation,
            "road_border_violation": self.road_border_violation,
            "min_road_border_distance_m": self.min_road_border_distance_m,
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

    geometry_values = _geometry_metric_values(result)

    trace = _build_closed_loop_trace(
        result,
        poses=poses,
        speeds=speeds,
        accelerations=accelerations,
        yaw_rates=yaw_rates,
        goal_pose_world=goal_pose_world,
        collision_steps=collision_steps,
    )

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
        min_agent_clearance_m=geometry_values["min_agent_clearance_m"],
        min_ttc_s=geometry_values["min_ttc_s"],
        ttc_violation=geometry_values["ttc_violation"],
        drivable_violation=geometry_values["drivable_violation"],
        road_border_violation=geometry_values["road_border_violation"],
        min_road_border_distance_m=geometry_values["min_road_border_distance_m"],
        termination_reason=termination_reason,
        token=token,
        trace=trace,
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


def _build_closed_loop_trace(
    result: T4ClosedLoopResult,
    *,
    poses: np.ndarray,
    speeds: np.ndarray,
    accelerations: np.ndarray,
    yaw_rates: np.ndarray,
    goal_pose_world: Optional[np.ndarray],
    collision_steps: Optional[Sequence[int]],
) -> ClosedLoopTrace:
    """Build one row per simulated action from the runner result."""

    source_frames = np.asarray(result.source_frames, dtype=np.int64)
    num_steps = len(source_frames)
    realized = np.asarray(poses[1:], dtype=np.float64)
    displacement = np.diff(np.asarray(poses, dtype=np.float64)[:, :2], axis=0)
    step_distance = np.linalg.norm(displacement, axis=-1)

    collision = None
    if collision_steps is not None:
        collision = np.zeros(num_steps, dtype=bool)
        for step in collision_steps:
            if 0 <= int(step) < num_steps:
                collision[int(step)] = True

    goal_distance = None
    if goal_pose_world is not None:
        goal_distance = np.linalg.norm(
            realized[:, :2] - _goal_xy(goal_pose_world)[None, :], axis=-1
        )

    plan_available = np.asarray([plan is not None for plan in result.plans], dtype=bool)
    plan_num_poses = np.asarray(
        [len(plan) if plan is not None else 0 for plan in result.plans], dtype=np.int64
    )
    plan_interval_s = np.asarray(
        [
            plan.trajectory_sampling.interval_length if plan is not None else np.nan
            for plan in result.plans
        ],
        dtype=np.float64,
    )

    geometry = result.geometry

    def geometry_column(name: str) -> Optional[np.ndarray]:
        if geometry is None:
            return None
        values = [
            np.nan if event is None or getattr(event, name) is None else getattr(event, name)
            for event in geometry
        ]
        if all(not np.isfinite(value) for value in values):
            return None
        return np.asarray(values, dtype=np.float64)

    return ClosedLoopTrace(
        step=np.arange(num_steps, dtype=np.int64),
        source_frames=source_frames,
        time_s=np.arange(1, num_steps + 1, dtype=np.float64) * result.dt_s,
        poses_world=realized,
        speed_mps=speeds[1:] if len(speeds) == num_steps + 1 else speeds,
        acceleration_mps2=accelerations,
        yaw_rate_radps=yaw_rates,
        steering_rad=np.asarray(
            [state.steering_rad for state in result.states[1:]], dtype=np.float64
        ),
        step_distance_m=step_distance,
        path_length_m=np.cumsum(step_distance),
        goal_distance_m=goal_distance,
        collision=collision,
        plan_available=plan_available,
        plan_num_poses=plan_num_poses,
        plan_interval_s=plan_interval_s,
        agent_count=geometry_column("agent_count"),
        min_agent_clearance_m=geometry_column("min_agent_clearance_m"),
        ttc_s=geometry_column("ttc_s"),
        ttc_violation=geometry_column("ttc_violation"),
        drivable_violation=geometry_column("drivable_violation"),
        road_border_violation=geometry_column("road_border_violation"),
        road_border_distance_m=geometry_column("road_border_distance_m"),
    )


def _geometry_metric_values(result: T4ClosedLoopResult) -> dict[str, Optional[float]]:
    """Reduce optional per-tick geometry events without inventing zeros."""

    values: dict[str, Optional[float]] = {
        "min_agent_clearance_m": None,
        "min_ttc_s": None,
        "ttc_violation": None,
        "drivable_violation": None,
        "road_border_violation": None,
        "min_road_border_distance_m": None,
    }
    if result.geometry is None:
        return values

    def finite(name: str) -> list[float]:
        return [
            float(value)
            for event in result.geometry
            if event is not None
            for value in [getattr(event, name)]
            if value is not None and np.isfinite(value)
        ]

    for name in ("min_agent_clearance_m", "ttc_s", "road_border_distance_m"):
        available = finite(name)
        if available:
            values[
                {
                    "min_agent_clearance_m": "min_agent_clearance_m",
                    "ttc_s": "min_ttc_s",
                    "road_border_distance_m": "min_road_border_distance_m",
                }[name]
            ] = float(min(available))
    for name in ("ttc_violation", "drivable_violation", "road_border_violation"):
        available = [
            bool(getattr(event, name))
            for event in result.geometry
            if event is not None and getattr(event, name) is not None
        ]
        if available:
            values[name] = float(any(available))
    return values


__all__ = [
    "ClosedLoopMetricConfig",
    "ClosedLoopMetrics",
    "ClosedLoopTrace",
    "aggregate_closed_loop_metrics",
    "compute_closed_loop_metrics",
]
