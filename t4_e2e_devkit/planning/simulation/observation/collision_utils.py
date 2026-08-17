"""Collision taxonomy shared by the PDM scorer.

Native devkit replacement for ``nuplan.planning.metrics.utils.collision_utils``.
Only the enum and the two track-category lists are reachable from the devkit;
nuPlan's module additionally builds ``Statistic`` objects for its metric engine,
which is what forced a dependency on the whole nuPlan metrics stack.

``CollisionType`` values are the nuPlan integers unchanged -- ``PDMScorer``
compares against them by identity, and the at-fault rule ("front, lateral and
stopped-track collisions are the ego's fault; rear and stopped-ego collisions
are not") is expressed in those terms.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List

from t4_e2e_devkit.common.actor_state.tracked_objects_types import TrackedObjectType

VRU_types: List[TrackedObjectType] = [
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]

object_types: List[TrackedObjectType] = [
    TrackedObjectType.TRAFFIC_CONE,
    TrackedObjectType.BARRIER,
    TrackedObjectType.CZONE_SIGN,
    TrackedObjectType.GENERIC_OBJECT,
]


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4
