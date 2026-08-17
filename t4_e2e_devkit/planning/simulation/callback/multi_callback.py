"""Fan-out callback composition."""

from __future__ import annotations

from typing import Any, Iterable

from .abstract_callback import AbstractCallback


class MultiCallback(AbstractCallback):
    """Call a sequence of callbacks in registration order."""

    def __init__(self, callbacks: Iterable[Any] = ()) -> None:
        self._callbacks = list(callbacks)

    @property
    def callbacks(self) -> list[Any]:
        return self._callbacks

    def __len__(self) -> int:
        return len(self._callbacks)

    def _dispatch(self, name: str, *args: Any) -> None:
        for callback in self._callbacks:
            method = getattr(callback, name, None)
            if method is not None:
                method(*args)

    def on_initialization_start(self, setup: Any, planner: Any) -> None:
        self._dispatch("on_initialization_start", setup, planner)

    def on_initialization_end(self, setup: Any, planner: Any) -> None:
        self._dispatch("on_initialization_end", setup, planner)

    def on_step_start(self, setup: Any, planner: Any) -> None:
        self._dispatch("on_step_start", setup, planner)

    def on_step_end(self, setup: Any, planner: Any, sample: Any) -> None:
        self._dispatch("on_step_end", setup, planner, sample)

    def on_planner_start(self, setup: Any, planner: Any) -> None:
        self._dispatch("on_planner_start", setup, planner)

    def on_planner_end(self, setup: Any, planner: Any, trajectory: Any) -> None:
        self._dispatch("on_planner_end", setup, planner, trajectory)

    def on_simulation_start(self, setup: Any) -> None:
        self._dispatch("on_simulation_start", setup)

    def on_simulation_end(self, setup: Any, planner: Any, history: Any) -> None:
        self._dispatch("on_simulation_end", setup, planner, history)

    def on_simulation_error(self, setup: Any, planner: Any, error: BaseException) -> None:
        self._dispatch("on_simulation_error", setup, planner, error)


__all__ = ["MultiCallback"]
