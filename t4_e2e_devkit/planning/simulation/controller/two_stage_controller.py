"""Tracker plus motion-model controller composition."""

from __future__ import annotations

from typing import Any, Optional


class TwoStageController:
    """Apply a tracker and then a motion model, if supplied."""

    def __init__(
        self,
        scenario: Optional[Any] = None,
        tracker: Optional[Any] = None,
        motion_model: Optional[Any] = None,
    ) -> None:
        self.scenario = scenario
        self.tracker = tracker
        self.motion_model = motion_model
        self.state = None

    def reset(self) -> None:
        self.state = None
        for component in (self.tracker, self.motion_model):
            reset = getattr(component, "reset", None)
            if reset is not None:
                reset()

    def get_state(self) -> Any:
        if self.state is None:
            if self.scenario is None:
                raise RuntimeError("TwoStageController has no initial state")
            self.state = self.scenario.initial_ego_state
        return self.state

    def update_state(self, *args: Any) -> None:
        if len(args) >= 4:
            current_iteration, next_iteration, ego_state, trajectory = args
        else:
            trajectory, next_iteration = args
            current_iteration, ego_state = None, self.get_state()
        dynamic = trajectory
        if self.tracker is not None:
            if hasattr(self.tracker, "track_trajectory"):
                dynamic = self.tracker.track_trajectory(
                    current_iteration, next_iteration, ego_state, trajectory
                )
            elif hasattr(self.tracker, "track_state"):
                values = getattr(trajectory, "poses", trajectory)
                dynamic = self.tracker.track_state(ego_state, values)
            elif hasattr(self.tracker, "track"):
                values = getattr(trajectory, "poses", trajectory)
                dynamic = self.tracker.track(ego_state, values)
        if self.motion_model is not None and hasattr(self.motion_model, "propagate_state"):
            sampling_time = (
                next_iteration.time_point - current_iteration.time_point
                if current_iteration is not None
                else next_iteration
            )
            self.state = self.motion_model.propagate_state(ego_state, dynamic, sampling_time)
        else:
            self.state = dynamic


__all__ = ["TwoStageController"]
