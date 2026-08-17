# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/common/actor_state/ego_temporal_state.py
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

from __future__ import annotations

from functools import cached_property
from typing import List, Optional

from t4_e2e_devkit.common.actor_state.agent import Agent
from t4_e2e_devkit.common.actor_state.agent_temporal_state import AgentTemporalState
from t4_e2e_devkit.common.actor_state.ego_state import EgoState
from t4_e2e_devkit.common.actor_state.tracked_objects_types import TrackedObjectType
from t4_e2e_devkit.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class EgoTemporalState(AgentTemporalState):
    """
    Temporal ego state, with future and past trajectory
    """

    def __init__(
        self,
        current_state: EgoState,
        past_trajectory: Optional[PredictedTrajectory] = None,
        predictions: Optional[List[PredictedTrajectory]] = None,
    ):
        """
        Initialize temporal state
        :param current_state: current state of ego
        :param past_trajectory: past trajectory, where last waypoint represents the same position as current state
        :param predictions: multimodal predictions, or future trajectory
        """
        super().__init__(
            initial_time_stamp=current_state.time_point, predictions=predictions, past_trajectory=past_trajectory
        )
        self._ego_current_state = current_state

    @property
    def ego_current_state(self) -> EgoState:
        """
        :return: the current ego state
        """
        return self._ego_current_state

    @property
    def ego_previous_state(self) -> Optional[EgoState]:
        """
        :return: the previous ego state if exists. This is just a proxy to make sure the return type is correct.
        """
        return self.previous_state

    @cached_property
    def agent(self) -> Agent:
        """
        Casts the EgoTemporalState to an Agent object.
        :return: An Agent object with the parameters of EgoState
        """
        return Agent(
            metadata=self.ego_current_state.scene_object_metadata,
            tracked_object_type=TrackedObjectType.EGO,
            oriented_box=self.ego_current_state.car_footprint.oriented_box,
            velocity=self.ego_current_state.dynamic_car_state.center_velocity_2d,
            past_trajectory=self.past_trajectory,
            predictions=self.predictions,
        )
