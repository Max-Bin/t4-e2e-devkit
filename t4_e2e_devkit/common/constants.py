"""Shared constants for the T4 scene, training and evaluation contracts.

Data shapes and time ranges live here once. Architecture-specific values do
not belong in this module.

Temporal roles:

===========================  =========  =============================================
quantity                     value      role
===========================  =========  =============================================
reader context               30 + 1     past frames plus the current frame, at 10 Hz
raw future GT                80         8 s at 10 Hz
scorer horizon               40         4 s at 10 Hz; the PDM scoring horizon
default trajectory           8          poses at 0.5 s
===========================  =========  =============================================

The default trajectory is only a default. A :class:`Trajectory` carries its
own sampling and may use another valid grid.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Temporal window
# --------------------------------------------------------------------------- #

T4_FRAME_RATE_HZ = 10.0
T4_INTERVAL_LENGTH = 1.0 / T4_FRAME_RATE_HZ  # 0.1 s between source frames

INPUT_T = 30  # history frames, excluding the current one
OUTPUT_T = 80  # raw future GT frames (8 s)
PAST_FRAMES = INPUT_T + 1  # history window including the current frame
FUTURE_FRAMES = OUTPUT_T
MIN_T4_FRAMES = PAST_FRAMES + FUTURE_FRAMES  # shortest usable scene

# Scoring and default trajectory horizons.
SCORER_FUTURE_FRAMES = 40  # 4 s at 10 Hz -- the PDM scoring horizon
PDM_OBSERVATION_FRAMES = 50  # 5 s at 10 Hz -- PDM's TTC observation context
TRAJECTORY_POSES = 8  # default trajectory poses
TRAJECTORY_INTERVAL = 0.5  # seconds between default trajectory poses
TRAJECTORY_TIME_HORIZON = TRAJECTORY_POSES * TRAJECTORY_INTERVAL  # 4.0 s
FUTURE_STRIDE = int(round(TRAJECTORY_INTERVAL * T4_FRAME_RATE_HZ))  # 5 frames

# Spacing between consecutive training windows, in source frames.
DEFAULT_CENTER_STRIDE = 5

POSE_DIM = 4  # x, y, cos(yaw), sin(yaw)
MAX_NUM_NEIGHBORS = 32
MAX_NUM_AGENTS = MAX_NUM_NEIGHBORS + 1  # including ego

# --------------------------------------------------------------------------- #
# Vector map tensor shapes
# --------------------------------------------------------------------------- #

NUM_SEGMENTS_IN_LANE = 140
NUM_SEGMENTS_IN_ROUTE = 25
NUM_POLYGONS = 10
NUM_LINE_STRINGS = 60
POINTS_PER_LANELET = 20
POINTS_PER_POLYGON = 40
POINTS_PER_LINE_STRING = 20
POLYGON_TYPE_NUM = 1
LINE_STRING_TYPE_NUM = 2

# --------------------------------------------------------------------------- #
# Segment-matrix column layout
#
# One lane/route segment point is:
#   [X, Y, dX, dY, LeftBoundX, LeftBoundY, RightBoundX, RightBoundY,
#    traffic-light one-hot (5), left line-type one-hot (10),
#    right line-type one-hot (10)]
# --------------------------------------------------------------------------- #

X = 0
Y = 1
dX = 2
dY = 3
LB_X = 4
LB_Y = 5
RB_X = 6
RB_Y = 7

TRAFFIC_LIGHT = 8
TRAFFIC_LIGHT_GREEN = 8
TRAFFIC_LIGHT_YELLOW = 9
TRAFFIC_LIGHT_RED = 10
TRAFFIC_LIGHT_WHITE = 11
TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT = 12
TRAFFIC_LIGHT_ONE_HOT_DIM = 5

LINE_TYPE_NUM = 10
LINE_TYPE_LEFT_START = 13
LINE_TYPE_RIGHT_START = LINE_TYPE_LEFT_START + LINE_TYPE_NUM
SEGMENT_POINT_DIM = LINE_TYPE_RIGHT_START + LINE_TYPE_NUM  # 33

# Ego past trajectory columns.
EGO_AGENT_PAST_IDX_X = 0
EGO_AGENT_PAST_IDX_Y = 1
EGO_AGENT_PAST_IDX_COS = 2
EGO_AGENT_PAST_IDX_SIN = 3

# --------------------------------------------------------------------------- #
# Ego shape
#
# T4 carries the ego footprint per scene in ``derived/scalars.npz`` as
# ``shape = [wheel_base, length, width]``.  It is read, never assumed -- the
# fleet mixes vehicle types (jpntaxi, j6), and nuPlan's pacifica parameters are
# the wrong vehicle for all of them.
# --------------------------------------------------------------------------- #

EGO_SHAPE_DIM = 3
EGO_SHAPE_IDX_WHEEL_BASE = 0
EGO_SHAPE_IDX_LENGTH = 1
EGO_SHAPE_IDX_WIDTH = 2

# --------------------------------------------------------------------------- #
# Camera register
#
# The ORDER is part of the learned camera-register contract: training, cache
# staging, scorer audits and deployment must all resolve the same list, and a
# scene's own ``derived/cam_names.json`` fixes the on-disk order.
# --------------------------------------------------------------------------- #

# There is no single camera register across the T4 fleet, and assuming one is a
# real failure rather than a theoretical one.  Measured over sampled scenes:
#
#   prd_jt / prd_jt_val, 96%   11 cams, five wide views as JPEG, narrow as HEVC
#   prd_jt / prd_jt_val,  4%    8 cams, no CAM_FRONT_WIDE, no rear wide views
#   x2_dev,             100%   11 cams, ONE wide view, a real CAM_BACK, and nine
#                              JPEG directories rather than five
#
# So a fixed five-wide training profile resolves on 96% of prd_jt and on no
# x2_dev scene at all; x2_dev has its own six-camera surround profile below.
# Roof and traffic-light channels are kept in ``T4_ALL_CAMERA_NAMES`` for schema
# decoding but are never model input.

#: Five wide views: the reference camera profile for the main prd_jt rig.  The
#: reference papers use front, front-left, front-right and a rear camera; this
#: rig has no centred CAM_BACK, so the two wide rear views bracket that
#: direction symmetrically.
T4_WIDE5_CAMERA_NAMES: tuple[str, ...] = (
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT_WIDE",
)

#: The x2_dev surround register.  x2_dev exports a single wide view, so ``wide5``
#: cannot resolve there, but it exports six road-facing narrow views as JPEG
#: directories -- 2880x1860, one file per frame, zero distortion coefficients --
#: which is the same storage the wide views use on prd_jt.  The order mirrors
#: ``wide5``'s centre-first convention, front row then rear row, and is part of
#: the learned camera contract: changing it invalidates every checkpoint trained
#: through this profile.
T4_X2_SURROUND6_CAMERA_NAMES: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# Channels the reader will decode.  The gate is storage and direction, not focal
# length: a channel is supported when its rig exports it as one JPEG per frame
# and it faces the road.  The same narrow names are HEVC on prd_jt and JPEG on
# x2_dev, so support is decided per scene by what is actually on disk -- see
# ``dataset.rigs.readable_camera_names``.  ``CAM_BACK_WIDE`` occurs in a small
# subset only; the roof-centre views are excluded by direction, not storage.
T4_SUPPORTED_CAMERA_NAMES: tuple[str, ...] = (
    *T4_WIDE5_CAMERA_NAMES,
    "CAM_BACK_WIDE",
    *T4_X2_SURROUND6_CAMERA_NAMES,
)

#: Named profiles, resolvable by name from a config.  One profile per rig: a
#: profile that silently spanned two registers would train one model on two
#: input layouts.
T4_CAMERA_PROFILES: dict[str, tuple[str, ...]] = {
    "wide5": T4_WIDE5_CAMERA_NAMES,
    "x2_surround6": T4_X2_SURROUND6_CAMERA_NAMES,
}

#: Preference order used by ``resolve_camera_names`` when a run says ``"auto"``.
#: The two profiles are disjoint, so no rig satisfies both; the order only fixes
#: which one an inspection reports first.  There is no video fallback.
T4_CAMERA_PROFILE_PREFERENCE: tuple[str, ...] = ("wide5", "x2_surround6")

#: Every camera channel observed across prd_jt, prd_jt_val and x2_dev.  Not a
#: profile -- no single scene carries all of these.
T4_ALL_CAMERA_NAMES: tuple[str, ...] = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT",
    "CAM_BACK_RIGHT_WIDE",
    "CAM_BACK_WIDE",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_FRONT_WIDE",
    "CAM_TOP_LEFT_CENTER",
    "CAM_TOP_RIGHT_CENTER",
    "CAM_TRAFFIC_LIGHT_FAR",
    "CAM_TRAFFIC_LIGHT_NEAR",
)

#: Channels that are not part of any surround profile: they point at signal heads
#: rather than at the road, so they do not compose with a surround register.
T4_NON_SURROUND_CAMERA_NAMES: tuple[str, ...] = (
    "CAM_TRAFFIC_LIGHT_FAR",
    "CAM_TRAFFIC_LIGHT_NEAR",
    "CAM_TOP_LEFT_CENTER",
    "CAM_TOP_RIGHT_CENTER",
)

# Reader resolution as [H, W].
T4_DEFAULT_IMAGE_SIZE_HW: tuple[int, int] = (672, 1148)

# --------------------------------------------------------------------------- #
# LiDAR
# --------------------------------------------------------------------------- #

# T4 points are [x, y, z, intensity, ring_or_time].
T4_LIDAR_POINT_DIM = 5
T4_LIDAR_PACK_NAME = "LIDAR_CONCAT"

# --------------------------------------------------------------------------- #
# Scene directory layout
#
# A T4 scene directory is the canonical E2E input.  These names are the on-disk
# contract owned by the upstream T4 converter; the devkit reads them and never
# invents an alternative archive format.
# --------------------------------------------------------------------------- #

SCENE_META = "derived/meta.json"
SCENE_SCALARS = "derived/scalars.npz"
SCENE_FRAMES = "derived/frames.pack"
SCENE_CAM_NAMES = "derived/cam_names.json"
SCENE_LIDAR = "data/LIDAR_CONCAT.pack"
SCENE_CAMERA_DIR = "data"

# E2E rows are restricted to the subtrees whose scenes are converted for
# planning.  Standalone perception training has its own annotated-data boundary,
# and this guard is what protects it: ``annotated_data`` must never enter an E2E
# data list.
#
#   prd_jt      training clips, annotation-free (online tracker labels)
#   prd_jt_val  held-out validation; same converter output, different vehicles
#   x2_dev      the X2 development fleet
#
# x2_dev additionally keeps its raw ``annotation/`` tables inside each scene
# directory, so its camera calibration -- including the distortion coefficients
# that ``derived/scalars.npz`` does not carry -- can be read back per scene.
# prd_jt cannot: its ``source_scene_dir`` no longer exists.
T4_E2E_SUBTREES: tuple[str, ...] = ("prd_jt", "prd_jt_val", "x2_dev")

# Per-scene raw annotation tables, present in x2_dev but not in prd_jt.  This is
# the only route back to distortion coefficients and per-sensor timestamps.
SCENE_RAW_ANNOTATION_DIR = "annotation"
SCENE_RAW_CALIBRATED_SENSOR = "annotation/calibrated_sensor.json"
SCENE_RAW_SENSOR = "annotation/sensor.json"

# --------------------------------------------------------------------------- #
# Data list
# --------------------------------------------------------------------------- #

DATA_LIST_FORMAT = "t4-e2e.datalist"
DATA_LIST_VERSION = 1
