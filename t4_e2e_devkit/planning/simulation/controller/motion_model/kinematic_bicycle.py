"""A small rear-axle kinematic bicycle model."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from t4_e2e_devkit.planning.simulation.closed_loop import KinematicState

from .abstract_motion_model import AbstractMotionModel


class KinematicBicycleMotionModel(AbstractMotionModel):
    """Integrate acceleration and steering for :class:`KinematicState`.

    Commands may be a mapping with ``acceleration_mps2`` and ``steering_rad``,
    a two-value array, or a state/reference already produced by a tracker. A
    tracked state is returned unchanged, which makes the model composable with
    perfect or LQR-style trackers.
    """

    def __init__(
        self,
        wheel_base_m: float = 2.7,
        *,
        max_speed_mps: float | None = None,
        max_steering_rad: float | None = None,
    ) -> None:
        if wheel_base_m <= 0.0:
            raise ValueError("wheel_base_m must be positive")
        if max_speed_mps is not None and max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive when provided")
        if max_steering_rad is not None and max_steering_rad <= 0.0:
            raise ValueError("max_steering_rad must be positive when provided")
        self.wheel_base_m = float(wheel_base_m)
        self.max_speed_mps = None if max_speed_mps is None else float(max_speed_mps)
        self.max_steering_rad = None if max_steering_rad is None else float(max_steering_rad)

    def propagate_state(self, ego_state: Any, command: Any, sampling_time: Any) -> KinematicState:
        if not isinstance(ego_state, KinematicState):
            raise TypeError("KinematicBicycleMotionModel requires KinematicState")
        if isinstance(command, KinematicState):
            return command
        dt_s = _seconds(sampling_time)
        if dt_s <= 0.0:
            raise ValueError("sampling_time must be positive")
        acceleration, steering = _command(command)
        if self.max_steering_rad is not None:
            steering = float(np.clip(steering, -self.max_steering_rad, self.max_steering_rad))
        speed = max(0.0, ego_state.speed_mps + acceleration * dt_s)
        if self.max_speed_mps is not None:
            speed = min(speed, self.max_speed_mps)
        yaw_rate = speed * math.tan(steering) / self.wheel_base_m
        heading = ego_state.heading + yaw_rate * dt_s
        return KinematicState(
            x=ego_state.x + speed * math.cos(heading) * dt_s,
            y=ego_state.y + speed * math.sin(heading) * dt_s,
            heading=heading,
            speed_mps=speed,
            acceleration_mps2=acceleration,
            yaw_rate_radps=yaw_rate,
            steering_rad=steering,
        )


def _seconds(value: Any) -> float:
    if hasattr(value, "time_s"):
        return float(value.time_s)
    return float(value)


def _command(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        acceleration = value.get("acceleration_mps2", value.get("acceleration", 0.0))
        steering = value.get("steering_rad", value.get("steering", 0.0))
        return float(acceleration), float(steering)
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("bicycle command needs acceleration and steering")
    return float(values[0]), float(values[1])


__all__ = ["KinematicBicycleMotionModel"]
