# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/planning/simulation/trajectory/predicted_trajectory.py
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

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Union

from t4_e2e_devkit.common.actor_state.ego_state import EgoState
from t4_e2e_devkit.common.actor_state.waypoint import Waypoint
from t4_e2e_devkit.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from t4_e2e_devkit.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

WaypointTypes = Union[Waypoint, EgoState]


@dataclass
class PredictedTrajectory:
    """Stores a predicted trajectory, along with its probability."""

    # Probability assigned to this trajectory prediction
    probability: float

    # List of predicted waypoints, if None, we appended the predictions to have desired length
    waypoints: List[Optional[WaypointTypes]]

    @property
    def valid_waypoints(self) -> List[WaypointTypes]:
        """
        Interface to get only valid waypoints
        :return: waypoints which are not None
        """
        return [w for w in self.waypoints if w]

    @cached_property
    def trajectory(self) -> AbstractTrajectory:
        """
        Interface to compute trajectory from waypoints
        :return: trajectory from waypoints
        """
        return InterpolatedTrajectory(self.valid_waypoints)

    def __len__(self) -> int:
        """
        :return: number of waypoints in trajectory
        """
        return len(self.waypoints)
