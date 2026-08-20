"""Observation containers.

Native devkit replacement for ``nuplan.planning.simulation.observation.
observation_type``.  Class and field names are kept identical so vendored PDM
code binds to it unchanged; what differs is the sensor vocabulary.  nuPlan's
version types its ``Sensors`` payload with ``nuplan.database`` image and point
cloud classes, which would pull the whole nuPlan database layer in for a
container the devkit never fills that way -- T4 sensor data arrives as plain
arrays from :mod:`t4_e2e_devkit.dataset`.

The channel enums are T4's, not nuPlan's ``CAM_F0``/``CAM_B0`` rig.  The
authoritative ordered camera register still lives in
:mod:`t4_e2e_devkit.common.constants`; these names exist so an observation can
say which channel it carries.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type, Union

from t4_e2e_devkit.common.actor_state.tracked_objects import TrackedObjects


class CameraChannel(Enum):
    """Camera channels of the T4 rig, named as they appear on disk."""

    CAM_FRONT = "CAM_FRONT"
    CAM_FRONT_WIDE = "CAM_FRONT_WIDE"
    CAM_FRONT_LEFT = "CAM_FRONT_LEFT"
    CAM_FRONT_LEFT_WIDE = "CAM_FRONT_LEFT_WIDE"
    CAM_FRONT_RIGHT = "CAM_FRONT_RIGHT"
    CAM_FRONT_RIGHT_WIDE = "CAM_FRONT_RIGHT_WIDE"
    # x2_dev carries a centred rear camera; the prd_jt rigs do not.
    CAM_BACK = "CAM_BACK"
    CAM_BACK_LEFT = "CAM_BACK_LEFT"
    CAM_BACK_LEFT_WIDE = "CAM_BACK_LEFT_WIDE"
    CAM_BACK_RIGHT = "CAM_BACK_RIGHT"
    CAM_BACK_RIGHT_WIDE = "CAM_BACK_RIGHT_WIDE"
    CAM_BACK_WIDE = "CAM_BACK_WIDE"
    CAM_TRAFFIC_LIGHT_FAR = "CAM_TRAFFIC_LIGHT_FAR"


class LidarChannel(Enum):
    """LiDAR channels of the T4 rig.

    T4 scenes ship a single concatenated cloud (``data/LIDAR_CONCAT.pack``),
    already merged across the physical sensors.
    """

    LIDAR_CONCAT = "LIDAR_CONCAT"


SensorChannel = Union[CameraChannel, LidarChannel]


@dataclass
class Observation(ABC):  # noqa: B024 - marker base, matching nuPlan's shape
    """Abstract observation container.

    Intentionally has no abstract members: it exists so vendored PDM code can
    type-annotate "some observation" and dispatch on ``detection_type()``.
    """

    @classmethod
    def detection_type(cls) -> str:
        """:return: detection type of the observation."""
        return cls.__name__

    @classmethod
    def observation_type(cls) -> Type["Observation"]:
        """:return: concrete observation class represented by this payload."""
        return cls


@dataclass
class AbstractObservation(Observation):
    """Named base for observation providers and simulator integrations."""


@dataclass
class Sensors(AbstractObservation):
    """Raw sensor output: point clouds and images, as arrays.

    ``pointcloud`` values are ``[N, 5]`` float arrays of
    ``(x, y, z, intensity, ring_or_time)``; ``images`` values are ``[H, W, 3]``
    uint8 arrays.  Typed as ``Any`` so this module stays import-light -- the
    shape contract is enforced at the dataset boundary, not here.
    """

    pointcloud: Optional[Dict[LidarChannel, Any]]
    images: Optional[Dict[CameraChannel, Any]]


@dataclass
class DetectionsTracks(AbstractObservation):
    """Output of the perception system, i.e. tracks."""

    tracked_objects: TrackedObjects


__all__ = [
    "AbstractObservation",
    "CameraChannel",
    "DetectionsTracks",
    "LidarChannel",
    "Observation",
    "SensorChannel",
    "Sensors",
]
