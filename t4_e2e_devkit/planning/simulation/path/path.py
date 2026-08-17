# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/planning/simulation/path/path.py
# Commit : e924167
# Tool   : tools/vendor.py
#
# Re-run ``python tools/vendor.py sync`` to update this file, and
# ``python tools/vendor.py check`` to detect drift against its source.
#
# Only ``import`` statements were rewritten; every numeric expression is
# byte-identical to the source. Edits belong upstream, or in a devkit module
# that wraps this one.
# =============================================================================

from abc import ABCMeta, abstractmethod
from typing import List

from t4_e2e_devkit.common.actor_state.state_representation import ProgressStateSE2


class AbstractPath(metaclass=ABCMeta):
    """
    Generic agent or ego path interface
    """

    @abstractmethod
    def get_start_progress(self) -> float:
        """
        Get the path start progress.
        :return: Progress at start.
        """
        pass

    @abstractmethod
    def get_end_progress(self) -> float:
        """
        Get the path end progress
        :return: Progress at end.
        """
        pass

    @abstractmethod
    def get_state_at_progress(self, progress: float) -> ProgressStateSE2:
        """
        Get the state of the actor at the specified progress.
        :param progress: Progress for which to query a state.
        :return: State at the specified progress.

        :raises Exception: Throws an exception in case a progress is beyond range of a path.
        """
        pass

    @abstractmethod
    def get_state_at_progresses(self, progresses: List[float]) -> List[ProgressStateSE2]:
        """
        Get the state of the actor at the specified progresses.
        :param progresses: Progresses for which to query states.
        :return: States at the specified progresses.

        :raises Exception: Throws an exception in case a progress is beyond range of a path.
        """
        pass

    @abstractmethod
    def get_sampled_path(self) -> List[ProgressStateSE2]:
        """
        Get the sampled states along the trajectory.
        :return: Discrete path consisting of states.
        """
        pass
