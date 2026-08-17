"""Perfect trajectory tracking for T4 kinematic states."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class PerfectTrackingController:
    """Set the state to the next planned pose when a state supports it."""

    def __init__(self, scenario: Optional[Any] = None, initial_state: Optional[Any] = None) -> None:
        self.scenario = scenario
        self.state = initial_state
        self._index = 0

    def reset(self) -> None:
        self.state = None
        self._index = 0

    def get_state(self) -> Any:
        if self.state is not None:
            return self.state
        if self.scenario is None:
            raise RuntimeError("PerfectTrackingController has no initial state")
        self.state = self.scenario.initial_ego_state
        return self.state

    def update_state(self, *args: Any) -> None:
        trajectory = args[-1] if len(args) >= 4 else args[0]
        next_iteration = args[1] if len(args) >= 4 else None
        if hasattr(trajectory, "get_state_at_time") and next_iteration is not None:
            self.state = trajectory.get_state_at_time(next_iteration.time_point)
            return
        poses = np.asarray(getattr(trajectory, "poses", trajectory))
        if poses.ndim != 2 or len(poses) == 0:
            raise ValueError("perfect tracking requires a non-empty [N, 3] trajectory")
        if self.state is None:
            self.get_state()
        try:
            from t4_e2e_devkit.planning.simulation.closed_loop import KinematicState

            if isinstance(self.state, KinematicState):
                heading = float(self.state.heading + poses[0, 2])
                cosine, sine = np.cos(self.state.heading), np.sin(self.state.heading)
                x = self.state.x + cosine * float(poses[0, 0]) - sine * float(poses[0, 1])
                y = self.state.y + sine * float(poses[0, 0]) + cosine * float(poses[0, 1])
                speed = float(np.hypot(poses[0, 0], poses[0, 1]))
                self.state = KinematicState(
                    x,
                    y,
                    heading,
                    speed,
                )
        except ImportError:
            pass
        self._index += 1


__all__ = ["PerfectTrackingController"]
