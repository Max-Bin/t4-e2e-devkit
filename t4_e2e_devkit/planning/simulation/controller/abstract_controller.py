"""Controller protocol used by :class:`Simulation`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractEgoController(ABC):
    @abstractmethod
    def get_state(self) -> Any:
        """Return the current ego state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""

    @abstractmethod
    def update_state(self, current_iteration: Any, next_iteration: Any, ego_state: Any, trajectory: Any) -> None:
        """Propagate the current state using a planned trajectory."""


AbstractController = AbstractEgoController

__all__ = ["AbstractController", "AbstractEgoController"]
