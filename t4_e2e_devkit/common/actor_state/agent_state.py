# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/common/actor_state/agent_state.py
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

import copy
from typing import Optional

from t4_e2e_devkit.common.actor_state.oriented_box import OrientedBox
from t4_e2e_devkit.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from t4_e2e_devkit.common.actor_state.state_representation import StateSE2, StateVector2D
from t4_e2e_devkit.common.actor_state.tracked_objects_types import TrackedObjectType


class AgentState(SceneObject):
    """
    Class describing Agent State (including dynamics) in the scene, representing Vehicles, Bicycles and Pedestrians.
    """

    def __init__(
        self,
        tracked_object_type: TrackedObjectType,
        oriented_box: OrientedBox,
        velocity: StateVector2D,
        metadata: SceneObjectMetadata,
        angular_velocity: Optional[float] = None,
    ):
        """
        Representation of an Agent in the scene (Vehicles, Pedestrians, Bicyclists and GenericObjects).
        :param tracked_object_type: Type of the current agent.
        :param oriented_box: Geometrical representation of the Agent.
        :param velocity: Velocity (vectorial) of Agent.
        :param metadata: Agent's metadata.
        :param angular_velocity: The scalar angular velocity of the agent, if available.
        """
        super().__init__(tracked_object_type=tracked_object_type, oriented_box=oriented_box, metadata=metadata)
        self._velocity = velocity
        self._angular_velocity = angular_velocity

    @property
    def velocity(self) -> StateVector2D:
        """
        Getter for velocity.
        :return: The agent vectorial velocity.
        """
        return self._velocity

    @property
    def angular_velocity(self) -> Optional[float]:
        """
        Getter for angular.
        :return: The agent angular velocity.
        """
        return self._angular_velocity

    @classmethod
    def from_new_pose(cls, agent: AgentState, pose: StateSE2) -> AgentState:
        """
        Initializer that create the same agent in a different pose.
        :param agent: A sample agent.
        :param pose: The new pose.
        :return: A new agent.
        """
        return AgentState(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=OrientedBox.from_new_pose(agent.box, pose),
            velocity=agent.velocity,
            angular_velocity=agent.angular_velocity,
            metadata=copy.deepcopy(agent.metadata),
        )
