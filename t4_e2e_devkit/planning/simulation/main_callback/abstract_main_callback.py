"""Outer-run callback boundary."""

from __future__ import annotations

from typing import Any


class AbstractMainCallback:
    """Optional hooks around a batch execution."""

    def on_run_simulation_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_run_simulation_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_run_simulation_error(self, error: BaseException) -> None:
        del error


__all__ = ["AbstractMainCallback"]
