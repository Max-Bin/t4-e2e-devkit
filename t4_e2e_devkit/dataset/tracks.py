"""Converting T4 annotations into the scorer's track representation.

The PDM scorer works with nuPlan-shaped ``TrackedObject`` instances -- it needs
each object's polygon, its track token for the already-collided dedup, and its
type for the at-fault classification.  T4 stores the same information as two
arrays.  This module is the seam.

The label mapping is defined here at the T4 annotation boundary::

    0 car, 1 truck, 2 bus  ->  VEHICLE
    3 bicycle              ->  BICYCLE
    4 pedestrian           ->  PEDESTRIAN

That mapping drives PDM's collision semantics -- types in ``AGENT_TYPES`` score
0.0 on an at-fault collision while static objects score 0.5, and the
ahead/behind fault test only runs for agents -- so a second copy that drifted by
one entry would change every reported score while raising nothing.

Box columns come from :class:`~t4_e2e_devkit.common.enums.T4BoxIndex`:
``[x, y, z, width, length, height, yaw, vx, vy]``, width before length.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.actor_state.agent import Agent
from t4_e2e_devkit.common.actor_state.oriented_box import OrientedBox
from t4_e2e_devkit.common.actor_state.scene_object import SceneObjectMetadata
from t4_e2e_devkit.common.actor_state.state_representation import StateSE2, StateVector2D
from t4_e2e_devkit.common.actor_state.static_object import StaticObject
from t4_e2e_devkit.common.actor_state.tracked_objects import TrackedObjects
from t4_e2e_devkit.common.actor_state.tracked_objects_types import (
    AGENT_TYPES,
    TrackedObjectType,
)
from t4_e2e_devkit.common.enums import T4BoxIndex
from t4_e2e_devkit.planning.simulation.observation.observation_type import DetectionsTracks

#: T4 class id -> tracked object type.
T4_LABEL_TO_TRACKED_OBJECT_TYPE: Dict[int, TrackedObjectType] = {
    0: TrackedObjectType.VEHICLE,
    1: TrackedObjectType.VEHICLE,
    2: TrackedObjectType.VEHICLE,
    3: TrackedObjectType.BICYCLE,
    4: TrackedObjectType.PEDESTRIAN,
}

#: Types that get an ``Agent`` (with velocity) rather than a ``StaticObject``.
AGENT_TRACK_TYPES = frozenset(AGENT_TYPES)


def tracked_object_type_for_label(label: int) -> TrackedObjectType:
    """
    Map one T4 class id to a tracked object type.
    :param label: T4 ``gt_labels`` value.
    :return: the corresponding tracked object type.
    :raises ValueError: for a label the mapping does not cover.  The judge
        defaults an unknown label to ``GENERIC_OBJECT``; the devkit raises
        instead, because at the dataset boundary a class the mapping has never
        seen means the converter's vocabulary changed, and silently scoring it
        as a static obstacle would hide that.
    """
    try:
        return T4_LABEL_TO_TRACKED_OBJECT_TYPE[int(label)]
    except KeyError:
        raise ValueError(
            f"unmapped T4 track label {label}; known labels are "
            f"{sorted(T4_LABEL_TO_TRACKED_OBJECT_TYPE)}. If the converter's class "
            "vocabulary changed, update T4_LABEL_TO_TRACKED_OBJECT_TYPE with "
            "a deliberate agent/static choice."
        ) from None


def annotations_to_tracked_objects(
    annotations,
    track_tokens: Optional[Sequence[str]] = None,
    velocities: Optional[npt.NDArray[np.floating]] = None,
    timestamp_us: int = 0,
) -> TrackedObjects:
    """
    Build a :class:`TrackedObjects` collection from one frame's annotations.
    :param annotations: :class:`~t4_e2e_devkit.common.dataclasses.Annotations`
        whose ``boxes`` follow :class:`T4BoxIndex`.
    :param track_tokens: stable per-object identifiers; synthesised by index
        when absent.  PDM's already-collided dedup keys on these, so an
        identifier that changes between frames makes one collision count twice.
    :param velocities: ``[M, 2]`` planar velocities; taken from box columns 7-8
        when absent.
    :param timestamp_us: timestamp recorded on each object's metadata.
    :return: tracked objects for this frame.
    """
    boxes = np.asarray(annotations.boxes, dtype=np.float64)
    labels = np.asarray(annotations.labels, dtype=np.int64)
    has_velocity_columns = boxes.ndim == 2 and boxes.shape[1] >= T4BoxIndex.size()

    tokens = list(track_tokens) if track_tokens is not None else None
    if tokens is None:
        tokens = getattr(annotations, "track_tokens", None)
    if tokens is None:
        tokens = [f"t4_track_{index}" for index in range(len(boxes))]
    if velocities is None:
        velocities = getattr(annotations, "velocities", None)

    objects: List[object] = []
    for index, (box, label) in enumerate(zip(boxes, labels, strict=True)):
        object_type = tracked_object_type_for_label(int(label))
        pose = StateSE2(
            float(box[T4BoxIndex.X]),
            float(box[T4BoxIndex.Y]),
            float(box[T4BoxIndex.HEADING]),
        )
        # Floored exactly as the judge's _make_single_object does: a degenerate
        # box would otherwise produce an empty polygon and vanish from the
        # occupancy map instead of being scored.
        oriented_box = OrientedBox(
            pose,
            length=max(float(box[T4BoxIndex.LENGTH]), 1.0e-3),
            width=max(float(box[T4BoxIndex.WIDTH]), 1.0e-3),
            height=max(float(box[T4BoxIndex.HEIGHT]), 1.0e-3),
        )
        metadata = SceneObjectMetadata(
            timestamp_us=int(timestamp_us),
            token=str(tokens[index]),
            track_id=None,
            track_token=str(tokens[index]),
        )
        if object_type in AGENT_TRACK_TYPES:
            if velocities is not None and len(velocities) > index:
                velocity = StateVector2D(float(velocities[index][0]), float(velocities[index][1]))
            elif has_velocity_columns:
                velocity = StateVector2D(
                    float(box[T4BoxIndex.VELOCITY_X]), float(box[T4BoxIndex.VELOCITY_Y])
                )
            else:
                velocity = StateVector2D(0.0, 0.0)
            objects.append(Agent(object_type, oriented_box, velocity, metadata=metadata))
        else:
            objects.append(StaticObject(object_type, oriented_box, metadata=metadata))

    return TrackedObjects(objects)


def annotations_to_detections_tracks(
    annotations,
    track_tokens: Optional[Sequence[str]] = None,
    velocities: Optional[npt.NDArray[np.floating]] = None,
    timestamp_us: int = 0,
) -> DetectionsTracks:
    """
    Wrap one frame's annotations as a perception observation.
    :param annotations: annotations whose boxes follow :class:`T4BoxIndex`.
    :param track_tokens: stable per-object identifiers.
    :param velocities: ``[M, 2]`` planar velocities.
    :param timestamp_us: timestamp recorded on each object's metadata.
    :return: detections tracks the PDM observation buffer accepts.
    """
    return DetectionsTracks(
        annotations_to_tracked_objects(annotations, track_tokens, velocities, timestamp_us)
    )
