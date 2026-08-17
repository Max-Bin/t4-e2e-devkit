# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/common/actor_state/static_object.py
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

from t4_e2e_devkit.common.actor_state.oriented_box import OrientedBox
from t4_e2e_devkit.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from t4_e2e_devkit.common.actor_state.state_representation import StateVector2D
from t4_e2e_devkit.common.actor_state.tracked_objects_types import TrackedObjectType


class StaticObject(SceneObject):
    """Represents static objects in the scene."""

    def __init__(
        self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, metadata: SceneObjectMetadata
    ):
        """
        :param tracked_object_type: Classification type of the object.
        :param oriented_box: OrientedBox representing the StaticObject geometrically.
        :param metadata: Metadata of a static object.
        """
        super().__init__(tracked_object_type, oriented_box, metadata)

        # TODO: these fields can be removed once we check how they are accessed
        self.predictions = None
        self.past_trajectory = None
        self.velocity = StateVector2D(0.0, 0.0)
