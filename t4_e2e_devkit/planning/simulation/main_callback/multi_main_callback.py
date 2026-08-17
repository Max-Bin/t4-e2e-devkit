"""Fan-out for outer-run callbacks."""

from __future__ import annotations

from typing import Any, Iterable

from .abstract_main_callback import AbstractMainCallback


class MultiMainCallback(AbstractMainCallback):
    """Dispatch each lifecycle hook in registration order."""

    def __init__(self, callbacks: Iterable[Any] = ()) -> None:
        self.callbacks = list(callbacks)

    def _dispatch(self, name: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            method = getattr(callback, name, None)
            if method is not None:
                method(*args, **kwargs)

    def on_run_simulation_start(self, *args: Any, **kwargs: Any) -> None:
        self._dispatch("on_run_simulation_start", *args, **kwargs)

    def on_run_simulation_end(self, *args: Any, **kwargs: Any) -> None:
        self._dispatch("on_run_simulation_end", *args, **kwargs)

    def on_run_simulation_error(self, error: BaseException) -> None:
        self._dispatch("on_run_simulation_error", error)


__all__ = ["MultiMainCallback"]
