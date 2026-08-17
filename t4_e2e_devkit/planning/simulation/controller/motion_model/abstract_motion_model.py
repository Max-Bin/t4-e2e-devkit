"""Motion-model boundary used by two-stage ego controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractMotionModel(ABC):
    """Propagate an ego state for one simulation interval."""

    @abstractmethod
    def propagate_state(self, ego_state: Any, command: Any, sampling_time: Any) -> Any:
        """Return the state after applying ``command`` for ``sampling_time``."""


__all__ = ["AbstractMotionModel"]
