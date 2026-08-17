"""Lifecycle callback base class.

Every hook is optional in practice.  The no-op defaults make small callbacks
pleasant to write while keeping the complete lifecycle explicit.
"""

from __future__ import annotations

from typing import Any


class AbstractCallback:
    """Base class for simulation callbacks."""

    def on_initialization_start(self, setup: Any, planner: Any) -> None:
        del setup, planner

    def on_initialization_end(self, setup: Any, planner: Any) -> None:
        del setup, planner

    def on_step_start(self, setup: Any, planner: Any) -> None:
        del setup, planner

    def on_step_end(self, setup: Any, planner: Any, sample: Any) -> None:
        del setup, planner, sample

    def on_planner_start(self, setup: Any, planner: Any) -> None:
        del setup, planner

    def on_planner_end(self, setup: Any, planner: Any, trajectory: Any) -> None:
        del setup, planner, trajectory

    def on_simulation_start(self, setup: Any) -> None:
        del setup

    def on_simulation_end(self, setup: Any, planner: Any, history: Any) -> None:
        del setup, planner, history

    def on_simulation_error(self, setup: Any, planner: Any, error: BaseException) -> None:
        del setup, planner, error


__all__ = ["AbstractCallback"]
