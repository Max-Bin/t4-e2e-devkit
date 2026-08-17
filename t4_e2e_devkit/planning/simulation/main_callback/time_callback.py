"""Wall-clock accounting for an outer batch run."""

from __future__ import annotations

import time
from typing import Any

from .abstract_main_callback import AbstractMainCallback


class TimeCallback(AbstractMainCallback):
    """Record elapsed wall time without requiring a logger."""

    def __init__(self) -> None:
        self.start_time = 0.0
        self.elapsed_s = 0.0

    def on_run_simulation_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.start_time = time.perf_counter()
        self.elapsed_s = 0.0

    def on_run_simulation_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.elapsed_s = time.perf_counter() - self.start_time


__all__ = ["TimeCallback"]
