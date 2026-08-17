"""Runner protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractRunner(ABC):
    @abstractmethod
    def run(self) -> Any:
        """Run and return a :class:`RunnerReport`."""

    @property
    @abstractmethod
    def scenario(self) -> Any:
        """Scenario handled by this runner."""

    @property
    @abstractmethod
    def planner(self) -> Any:
        """Planner handled by this runner."""


__all__ = ["AbstractRunner"]
