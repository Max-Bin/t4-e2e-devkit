"""Lightweight T4 kinematic controller adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class KinematicBicycleController:
    """Use a tracker to realize a local pose trajectory."""

    def __init__(
        self,
        tracker: Optional[Any] = None,
        initial_state: Any = None,
        *,
        trajectory_in_local_frame: bool = True,
    ) -> None:
        if tracker is None:
            from t4_e2e_devkit.planning.simulation.closed_loop import PerfectTracker

            tracker = PerfectTracker()
        self.tracker = tracker
        self.state = initial_state
        self.trajectory_in_local_frame = bool(trajectory_in_local_frame)

    def reset(self) -> None:
        self.state = None
        reset = getattr(self.tracker, "reset", None)
        if reset is not None:
            reset()

    def set_state(self, state: Any) -> None:
        self.state = state

    def get_state(self) -> Any:
        if self.state is None:
            raise RuntimeError("KinematicBicycleController has no current state")
        return self.state

    def step(self, reference: Any) -> Any:
        if self.state is None:
            raise RuntimeError("KinematicBicycleController has no current state")
        values = getattr(reference, "poses", reference)
        values = np.asarray(values, dtype=np.float64)
        if self.trajectory_in_local_frame and isinstance(self.state, _kinematic_state_type()):
            values = _local_to_world(values, self.state)
        if hasattr(self.tracker, "track_state"):
            self.state = self.tracker.track_state(self.state, values)
        else:
            self.state = self.tracker.step(self.state, values)
        return self.state

    def update_state(self, *args: Any) -> None:
        trajectory = args[-1] if len(args) >= 4 else args[0]
        self.step(trajectory)


__all__ = ["KinematicBicycleController"]


def _kinematic_state_type():
    from t4_e2e_devkit.planning.simulation.closed_loop import KinematicState

    return KinematicState


def _local_to_world(values: np.ndarray, state: Any) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("trajectory must have shape [N, 3]")
    cosine, sine = np.cos(state.heading), np.sin(state.heading)
    result = values.copy()
    x_local, y_local = values[:, 0], values[:, 1]
    result[:, 0] = state.x + cosine * x_local - sine * y_local
    result[:, 1] = state.y + sine * x_local + cosine * y_local
    result[:, 2] = state.heading + values[:, 2]
    return result
