"""Array-layout indices and categorical enums for the T4 contract.

These name the columns of the raw arrays that move between the reader, the
agents and the scorer.  They exist so a slice reads ``boxes[..., BoundingBoxIndex.HEADING]``
rather than ``boxes[..., 6]``.

The indices are plain class attributes, so ``StateSE2Index.X`` is an ``int``
rather than an enum member. This keeps array access stable across supported
Python versions.
"""

from __future__ import annotations

from enum import IntEnum


class SceneFrameType(IntEnum):
    """Whether a frame was recorded or generated."""

    ORIGINAL = 0
    SYNTHETIC = 1


class StateSE2Index:
    """Columns of an SE(2) state array ``[x, y, heading]``."""

    X = 0
    Y = 1
    HEADING = 2

    POINT = slice(0, 2)
    STATE_SE2 = slice(0, 3)

    @classmethod
    def size(cls) -> int:
        """:return: number of columns in one SE(2) row."""
        return 3


class T4BoxIndex:
    """Columns of a T4 ``gt_boxes`` row.

    ``[x, y, z, width, length, height, yaw, vx, vy]`` -- nine columns, and note
    that **width precedes length**.  This is the layout the T4 converter writes
    into ``derived/frames.pack`` and the one every scoring path consumes: the
    CPU judge's ``_t4_box_corners`` reads half-length from column 4 and
    half-width from column 3, and the GPU oracle reads the velocity pair from
    columns 7 and 8.

    It is deliberately not :class:`BoundingBoxIndex`. The alternate seven-
    column layout puts length before width and carries no velocity, so reading a
    T4 row through it silently swaps every object's footprint axes.
    """

    X = 0
    Y = 1
    Z = 2
    WIDTH = 3
    LENGTH = 4
    HEIGHT = 5
    HEADING = 6
    VELOCITY_X = 7
    VELOCITY_Y = 8

    POINT2D = slice(0, 2)
    POSITION = slice(0, 3)
    DIMENSION = slice(3, 6)  # (width, length, height)
    VELOCITY_2D = slice(7, 9)

    @classmethod
    def size(cls) -> int:
        """:return: number of columns in one T4 box row."""
        return 9


class BoundingBoxIndex:
    """Columns of the alternate seven-column 3D box layout.

    ``[x, y, z, length, width, height, heading]``.  Kept for code ported from
    a source using this layout; T4 data on disk uses :class:`T4BoxIndex`.
    """

    X = 0
    Y = 1
    Z = 2
    LENGTH = 3
    WIDTH = 4
    HEIGHT = 5
    HEADING = 6

    POINT2D = slice(0, 2)
    POSITION = slice(0, 3)
    DIMENSION = slice(3, 6)

    @classmethod
    def size(cls) -> int:
        """:return: number of columns in one box row."""
        return 7


class LidarIndex:
    """Columns of a T4 LiDAR point array.

    T4 points are ``[x, y, z, intensity, ring_or_time]``. The concatenated cloud
    has no sixth object-id column, so reading column 5 is an error.
    """

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4

    POINT2D = slice(0, 2)
    POSITION = slice(0, 3)

    @classmethod
    def size(cls) -> int:
        """:return: number of columns in one point row."""
        return 5


class EgoStatusIndex:
    """Columns of the ego status row handed to an agent.

    ``[x, y, heading, vx, vy, ax, ay]`` in the centre frame.  All seven columns
    are differenced from the pose history by a single function, so training and
    rollout share one definition.

    The ``velocity``/``acceleration`` arrays in ``derived/scalars.npz`` are
    deliberately NOT this: they come from ``/localization/kinematic_state`` and
    ``/localization/acceleration``, where the EKF's stop filter rewrites any
    speed below 0.1 m/s to exactly zero (a third of ``prd_jt`` frames), the
    lateral components are identically zero dataset-wide, and the CAN/IMU twist
    trails the NDT pose by roughly 130 ms.  They are also body-frame per frame,
    so pairing them with centre-frame pose columns disagrees about axes in a
    turn.  Deployment still reports the EKF estimate -- the on-vehicle
    controller needs the vehicle's own numbers -- which is why that lives on
    ``control_state`` and not here.
    """

    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6

    POSE = slice(0, 3)
    VELOCITY_2D = slice(3, 5)
    ACCELERATION_2D = slice(5, 7)

    @classmethod
    def size(cls) -> int:
        """:return: number of columns in one ego status row."""
        return 7


class T4TrackLabel(IntEnum):
    """Classes in the T4 bundle's ``gt_labels`` field.

    Five classes, three of which collapse to ``VEHICLE`` for scoring.  The
    label-to-scoring-type mapping is owned by the reference judge
    (``evaluation.reference.pdm_closed._LABEL_TO_TYPE``) and re-exported by
    :mod:`t4_e2e_devkit.dataset.tracks`; it lives there rather than here so the
    judge and the devkit cannot disagree about which class is an agent.
    """

    CAR = 0
    TRUCK = 1
    BUS = 2
    BICYCLE = 3
    PEDESTRIAN = 4


class TurnIndicator(IntEnum):
    """T4's five-class historical turn-indicator field.

    Read from the scene but not consumed by the reference agents: route intent
    is expressed as attention over the route-lane tokens, not as a handcrafted
    left/straight/right command.
    """

    NONE = 0
    DISABLE = 1
    ENABLE_LEFT = 2
    ENABLE_RIGHT = 3
    KEEP = 4


class SensorModality(IntEnum):
    """Which sensor streams an agent asks the reader to decode."""

    CAMERA = 0
    LIDAR = 1
