"""Recorded ego-state playback controller."""

from __future__ import annotations

from typing import Any, Iterable, Optional


class LogPlaybackController:
    """Advance through states supplied by a scenario or an in-memory sequence."""

    def __init__(
        self, scenario: Optional[Any] = None, states: Optional[Iterable[Any]] = None
    ) -> None:
        if (
            states is None
            and scenario is not None
            and not hasattr(scenario, "get_ego_state_at_iteration")
        ):
            states = scenario
            scenario = None
        self.scenario = scenario
        self.states = None if states is None else tuple(states)
        if self.states is None and self.scenario is None:
            raise ValueError("LogPlaybackController needs a scenario or states")
        if self.states is not None and not self.states:
            raise ValueError("LogPlaybackController needs at least one state")
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def get_state(self) -> Any:
        if self.states is not None:
            return self.states[min(self._index, len(self.states) - 1)]
        return self.scenario.get_ego_state_at_iteration(self._index)

    def update_state(self, *args: Any) -> None:
        next_iteration = args[1] if len(args) >= 4 else args[-1]
        self._index = int(getattr(next_iteration, "index", next_iteration))


__all__ = ["LogPlaybackController"]
