"""Portable artifacts for T4 sensor-replay closed-loop rollouts.

Artifacts contain the realized ego states, the plans emitted at each replan,
event flags and the per-tick metric trace.  They deliberately do not contain
raw sensor payloads: those remain in the T4 scene and can be addressed by the
recorded source frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from t4_e2e_devkit.common.dataclasses import Trajectory
from t4_e2e_devkit.evaluation.closed_loop import (
    ClosedLoopMetrics,
    ClosedLoopTrace,
)
from t4_e2e_devkit.planning.simulation.closed_loop import (
    KinematicState,
    T4ClosedLoopResult,
)
from t4_e2e_devkit.planning.simulation.closed_loop_geometry import ReplayGeometry
from t4_e2e_devkit.planning.simulation.interfaces import TrafficAgentState

CLOSED_LOOP_ARTIFACT_FORMAT = "t4.closed_loop.rollout"
CLOSED_LOOP_ARTIFACT_VERSION = 1


def write_rollout_artifact(
    path: str | Path,
    *,
    token: str,
    result: T4ClosedLoopResult,
    metrics: ClosedLoopMetrics,
    config: Mapping[str, Any],
    config_fingerprint: str,
    attempts: int = 1,
) -> Path:
    """Write one successful rollout artifact atomically."""

    payload = {
        "format": CLOSED_LOOP_ARTIFACT_FORMAT,
        "version": CLOSED_LOOP_ARTIFACT_VERSION,
        "status": "ok",
        "token": token,
        "config": dict(config),
        "config_fingerprint": config_fingerprint,
        "attempts": int(attempts),
        "result": _result_to_payload(result),
        "metrics": _metrics_to_payload(metrics, token),
    }
    return _atomic_write_json(path, payload)


def write_failed_artifact(
    path: str | Path,
    *,
    token: str,
    config: Mapping[str, Any],
    config_fingerprint: str,
    error: str,
    attempts: int,
) -> Path:
    """Write a failed row artifact so the failure is inspectable and retryable."""

    payload = {
        "format": CLOSED_LOOP_ARTIFACT_FORMAT,
        "version": CLOSED_LOOP_ARTIFACT_VERSION,
        "status": "failed",
        "token": token,
        "config": dict(config),
        "config_fingerprint": config_fingerprint,
        "attempts": int(attempts),
        "error": error,
    }
    return _atomic_write_json(path, payload)


def load_rollout_metrics(
    path: str | Path,
    *,
    token: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
) -> Optional[ClosedLoopMetrics]:
    """Load a completed artifact when it matches the requested run.

    ``None`` means that the artifact is absent, failed, malformed, or belongs
    to another configuration.  The caller can then rerun that row.
    """

    try:
        payload = load_rollout_artifact(path)
        if (
            payload.get("status") != "ok"
            or (token is not None and payload.get("token") != token)
            or (
                config_fingerprint is not None
                and payload.get("config_fingerprint") != config_fingerprint
            )
        ):
            return None
        return _metrics_from_payload(payload["metrics"], str(payload["token"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_rollout_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate one rollout artifact payload.

    The returned dictionary is intentionally plain JSON data so merge/report
    tools can inspect it without constructing a dataset reader or an agent.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"rollout artifact {path} must contain a JSON object")
    if payload.get("format") != CLOSED_LOOP_ARTIFACT_FORMAT:
        raise ValueError(f"unsupported rollout artifact format in {path}")
    if payload.get("version") != CLOSED_LOOP_ARTIFACT_VERSION:
        raise ValueError(f"unsupported rollout artifact version in {path}")
    if payload.get("status") not in {"ok", "failed"}:
        raise ValueError(f"invalid rollout artifact status in {path}")
    if not isinstance(payload.get("token"), str) or not payload["token"]:
        raise ValueError(f"rollout artifact {path} has no token")
    return payload


def rollout_artifact_path(directory: str | Path, row_index: int, token: str) -> Path:
    """Return the deterministic filename used for one rollout token."""

    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    return Path(directory) / f"{int(row_index):08d}-{digest}.json"


def write_rollout_payload(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write an already validated artifact payload atomically."""

    return _atomic_write_json(path, payload)


def _result_to_payload(result: T4ClosedLoopResult) -> dict[str, Any]:
    return {
        "source_frames": result.source_frames.astype(np.int64).tolist(),
        "states": [_state_to_list(state) for state in result.states],
        "plans": [_plan_to_payload(plan) for plan in result.plans],
        "dt_s": float(result.dt_s),
        "goal_pose_world": _array_or_none(result.goal_pose_world),
        "collision_steps": (
            None
            if result.collision_steps is None
            else [int(step) for step in result.collision_steps]
        ),
        "timeout": result.timeout,
        "termination_reason": result.termination_reason,
        "geometry": (
            None
            if result.geometry is None
            else [
                None if event is None else _geometry_to_payload(event) for event in result.geometry
            ]
        ),
        "traffic_states": (
            None
            if result.traffic_states is None
            else [
                [_traffic_state_to_payload(state) for state in states]
                for states in result.traffic_states
            ]
        ),
    }


def _metrics_to_payload(metrics: ClosedLoopMetrics, token: str) -> dict[str, Any]:
    return {
        "token": token,
        "values": {name: float(value) for name, value in metrics.values.items()},
        "termination_reason": metrics.termination_reason,
        "trace": None if metrics.trace is None else metrics.trace.rows(token),
    }


def _metrics_from_payload(payload: Mapping[str, Any], token: str) -> ClosedLoopMetrics:
    values = payload["values"]
    required = (
        "duration_s",
        "path_length_m",
        "final_displacement_m",
        "final_speed_mps",
        "max_speed_mps",
        "mean_abs_acceleration_mps2",
        "max_abs_acceleration_mps2",
        "max_abs_yaw_rate_radps",
        "stuck",
    )
    kwargs: dict[str, Any] = {name: float(values[name]) for name in required}
    for name in (
        "goal_reached",
        "collision",
        "first_collision_step",
        "timeout",
        "min_agent_clearance_m",
        "min_ttc_s",
        "ttc_violation",
        "drivable_violation",
        "road_border_violation",
        "min_road_border_distance_m",
    ):
        if name in values:
            kwargs[name] = float(values[name])
    trace_rows = payload.get("trace")
    kwargs["trace"] = _trace_from_rows(trace_rows) if trace_rows else None
    kwargs["termination_reason"] = payload.get("termination_reason")
    kwargs["token"] = token
    return ClosedLoopMetrics(**kwargs)


def _trace_from_rows(rows: list[Mapping[str, Any]]) -> ClosedLoopTrace:
    def column(name: str, dtype: Any) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=dtype)

    def optional_column(name: str, dtype: Any) -> Optional[np.ndarray]:
        if not any(name in row and row[name] is not None for row in rows):
            return None
        return np.asarray(
            [row.get(name, False if dtype is bool else np.nan) for row in rows],
            dtype=dtype,
        )

    return ClosedLoopTrace(
        step=column("step", np.int64),
        source_frames=column("source_frame", np.int64),
        time_s=column("time_s", np.float64),
        poses_world=np.asarray(
            [[row["x"], row["y"], row["heading"]] for row in rows],
            dtype=np.float64,
        ),
        speed_mps=column("speed_mps", np.float64),
        acceleration_mps2=column("acceleration_mps2", np.float64),
        yaw_rate_radps=column("yaw_rate_radps", np.float64),
        steering_rad=column("steering_rad", np.float64),
        step_distance_m=column("step_distance_m", np.float64),
        path_length_m=column("path_length_m", np.float64),
        goal_distance_m=optional_column("goal_distance_m", np.float64),
        collision=optional_column("collision", bool),
        plan_available=optional_column("plan_available", bool),
        plan_num_poses=optional_column("plan_num_poses", np.int64),
        plan_interval_s=optional_column("plan_interval_s", np.float64),
        agent_count=optional_column("agent_count", np.float64),
        min_agent_clearance_m=optional_column("min_agent_clearance_m", np.float64),
        ttc_s=optional_column("ttc_s", np.float64),
        ttc_violation=optional_column("ttc_violation", np.float64),
        drivable_violation=optional_column("drivable_violation", np.float64),
        road_border_violation=optional_column("road_border_violation", np.float64),
        road_border_distance_m=optional_column("road_border_distance_m", np.float64),
    )


def _state_to_list(state: KinematicState) -> list[float]:
    return [
        float(state.x),
        float(state.y),
        float(state.heading),
        float(state.speed_mps),
        float(state.acceleration_mps2),
        float(state.yaw_rate_radps),
        float(state.steering_rad),
    ]


def _geometry_to_payload(event: ReplayGeometry) -> dict[str, Any]:
    return {
        "agent_count": event.agent_count,
        "min_agent_clearance_m": event.min_agent_clearance_m,
        "ttc_s": event.ttc_s,
        "ttc_violation": event.ttc_violation,
        "drivable_violation": event.drivable_violation,
        "road_border_violation": event.road_border_violation,
        "road_border_distance_m": event.road_border_distance_m,
    }


def _traffic_state_to_payload(state: TrafficAgentState) -> dict[str, Any]:
    return {
        "track_token": state.track_token,
        "label": int(state.label),
        "box": np.asarray(state.box, dtype=np.float32).tolist(),
        "velocity": np.asarray(state.velocity, dtype=np.float32).tolist(),
    }


def _plan_to_payload(plan: Optional[Trajectory]) -> Optional[dict[str, Any]]:
    if plan is None:
        return None
    return {
        "poses": np.asarray(plan.poses, dtype=np.float32).tolist(),
        "num_poses": int(plan.trajectory_sampling.num_poses),
        "interval_length": float(plan.trajectory_sampling.interval_length),
    }


def _array_or_none(value: Optional[np.ndarray]) -> Optional[list[float]]:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "CLOSED_LOOP_ARTIFACT_FORMAT",
    "CLOSED_LOOP_ARTIFACT_VERSION",
    "load_rollout_artifact",
    "load_rollout_metrics",
    "rollout_artifact_path",
    "write_failed_artifact",
    "write_rollout_payload",
    "write_rollout_artifact",
]
