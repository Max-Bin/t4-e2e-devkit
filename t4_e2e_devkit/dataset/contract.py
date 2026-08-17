"""The on-disk field names and the collated batch contract.

Two vocabularies meet here and they are not the same, which is the reason this
module exists rather than the names being inlined at each call site:

* **Bundle fields** are what ``derived/frames.pack`` stores -- ``route``,
  ``lines``, ``lanes_speed``.  Those names are owned by the upstream T4
  converter and the devkit does not get to rename them on disk.
* **Contract fields** are what a model receives -- ``route_lanes``,
  ``line_strings``, ``lanes_speed_limit``.  These are the names every agent,
  feature builder and config in this repository uses.

:data:`BUNDLE_TO_CONTRACT` is the one place the two are related.  A rename in
the converter is a one-line change here instead of a search across the models.

The typed dicts below are documentation that type checkers can enforce; they
are intentionally ``total=False`` because which optional streams are present
depends on the agent's :class:`~t4_e2e_devkit.common.dataclasses.SensorConfig`.
The runtime check that matters is :func:`assert_batch_contract`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, TypedDict

if TYPE_CHECKING:
    import numpy as np
    import torch

# --------------------------------------------------------------------------- #
# On-disk names
# --------------------------------------------------------------------------- #

#: Per-frame map fields in ``derived/frames.pack``.
BUNDLE_MAP_FIELDS: tuple[str, ...] = (
    "lanes",
    "lanes_speed",
    "lanes_has_speed",
    "route",
    "route_speed",
    "route_has_speed",
    "polygons",
    "lines",
)

#: Per-frame agent ground truth in the same bundle.  Variable-length: a frame's
#: object count is its own, and a scene missing these fields is an error rather
#: than a scene with no traffic.
BUNDLE_GT_FIELDS: tuple[str, ...] = ("gt_boxes", "gt_labels")

#: Whole-scene arrays in ``derived/scalars.npz``.
SCALAR_FIELDS: tuple[str, ...] = ("trajectory", "velocity", "turn", "goal", "shape")

#: Bundle field name -> contract field name.
BUNDLE_TO_CONTRACT: Dict[str, str] = {
    "lanes": "lanes",
    "lanes_speed": "lanes_speed_limit",
    "lanes_has_speed": "lanes_has_speed_limit",
    "route": "route_lanes",
    "route_speed": "route_lanes_speed_limit",
    "route_has_speed": "route_lanes_has_speed_limit",
    "polygons": "polygons",
    "lines": "line_strings",
}

#: The inverse, for code that has to address the bundle directly.
CONTRACT_TO_BUNDLE: Dict[str, str] = {
    contract: bundle for bundle, contract in BUNDLE_TO_CONTRACT.items()
}

#: The map fields a model receives, in contract naming.
CONTRACT_MAP_FIELDS: tuple[str, ...] = tuple(BUNDLE_TO_CONTRACT[f] for f in BUNDLE_MAP_FIELDS)


# --------------------------------------------------------------------------- #
# Sample and batch
# --------------------------------------------------------------------------- #


class T4Sample(TypedDict, total=False):
    """One window before collation: numpy, no batch dimension."""

    # -- ego ---------------------------------------------------------------- #
    ego_agent_past: "np.ndarray"  # [PAST_FRAMES, 4] x, y, cos, sin in the centre frame
    ego_status: "np.ndarray"  # [7] x, y, heading, vx, vy, ax, ay
    ego_agent_future: "np.ndarray"  # [FUTURE_FRAMES, 3] dx, dy, dheading
    ego_shape: "np.ndarray"  # [3] wheel_base, length, width
    turn_indicators: "np.ndarray"  # [PAST_FRAMES]

    # -- map ---------------------------------------------------------------- #
    lanes: "np.ndarray"  # [140, 20, 33]
    lanes_speed_limit: "np.ndarray"  # [140, 1]
    lanes_has_speed_limit: "np.ndarray"  # [140, 1]
    route_lanes: "np.ndarray"  # [25, 20, 33]
    route_lanes_speed_limit: "np.ndarray"  # [25, 1]
    route_lanes_has_speed_limit: "np.ndarray"  # [25, 1]
    polygons: "np.ndarray"  # [10, 40, 3]
    line_strings: "np.ndarray"  # [60, 20, 4]
    goal_pose: "np.ndarray"  # [4] gx, gy, cos, sin in the centre frame

    # -- sensors (present only when the agent's SensorConfig asks) ---------- #
    camera_images: "np.ndarray"  # [N_cameras, 3, H, W], register order
    camera_names: List[str]
    camera_intrinsics: "np.ndarray"  # [N_cameras, 3, 3]
    camera_extrinsics: "np.ndarray"  # [N_cameras, 4, 4]
    points: "np.ndarray"  # [N, 5], ragged across samples

    # -- targets ------------------------------------------------------------ #
    gt_bboxes_3d: "np.ndarray"  # [M, 9]; empty only when the scene truly has none
    gt_labels_3d: "np.ndarray"  # [M]
    future_agent_boxes: List["np.ndarray"]  # centre-frame boxes, one array per future frame
    future_agent_labels: List["np.ndarray"]
    agent_gt_available: bool
    pdm_progress: float  # PDM-Closed reference progress; the EP denominator

    # -- identity ----------------------------------------------------------- #
    global_center_pose: "np.ndarray"  # [4] global x, y, cos, sin
    scene_dir: str
    scene_id: str
    center_idx: "np.ndarray"  # scalar int64
    token: str


class T4Batch(TypedDict, total=False):
    """A collated batch.

    Ragged streams -- point clouds, per-frame object lists -- stay Python lists
    rather than being padded.  Padding them would mean either a per-batch
    maximum that changes the tensor shape run to run, or a fixed cap that
    silently drops objects; both have produced real bugs.
    """

    ego_agent_past: "torch.Tensor"
    ego_status: "torch.Tensor"
    ego_agent_future: "torch.Tensor"
    ego_shape: "torch.Tensor"
    turn_indicators: "torch.Tensor"

    lanes: "torch.Tensor"
    lanes_speed_limit: "torch.Tensor"
    lanes_has_speed_limit: "torch.Tensor"
    route_lanes: "torch.Tensor"
    route_lanes_speed_limit: "torch.Tensor"
    route_lanes_has_speed_limit: "torch.Tensor"
    polygons: "torch.Tensor"
    line_strings: "torch.Tensor"
    goal_pose: "torch.Tensor"

    camera_images: "torch.Tensor"
    camera_names: List[List[str]]
    camera_intrinsics: "torch.Tensor"
    camera_extrinsics: "torch.Tensor"
    points: List["torch.Tensor"]

    gt_bboxes_3d: List["torch.Tensor"]
    gt_labels_3d: List["torch.Tensor"]
    future_agent_boxes: List[List["np.ndarray"]]
    future_agent_labels: List[List["np.ndarray"]]
    agent_gt_available: List[bool]
    pdm_progress: "torch.Tensor"

    global_center_pose: "torch.Tensor"
    scene_dir: List[str]
    scene_id: List[str]
    center_idx: "torch.Tensor"
    token: List[str]


#: Keys every batch carries regardless of sensor configuration.  Sensor streams
#: are deliberately absent: a LiDAR-only agent must not be required to produce
#: camera tensors, which is the whole reason both models can share one loader.
REQUIRED_BATCH_KEYS: frozenset[str] = frozenset(
    {
        "ego_agent_past",
        "ego_status",
        "ego_shape",
        "goal_pose",
        *CONTRACT_MAP_FIELDS,
        "global_center_pose",
        "scene_dir",
        "scene_id",
        "center_idx",
        "token",
    }
)

#: Additional keys a training batch must carry.
REQUIRED_TRAINING_KEYS: frozenset[str] = frozenset({"ego_agent_future"})


def assert_batch_contract(
    batch: dict,
    *,
    training: bool = False,
    require_cameras: bool = False,
    require_lidar: bool = False,
) -> None:
    """Fail at the loader boundary rather than inside a model's forward.

    :param batch: the collated batch to check.
    :param training: also require the supervision targets.
    :param require_cameras: also require the camera stream.
    :param require_lidar: also require the point cloud stream.
    :raises KeyError: listing every missing key, not just the first.
    """
    required = set(REQUIRED_BATCH_KEYS)
    if training:
        required |= REQUIRED_TRAINING_KEYS
    if require_cameras:
        required.add("camera_images")
    if require_lidar:
        required.add("points")

    missing = required - batch.keys()
    if missing:
        raise KeyError(
            f"batch is missing required keys {sorted(missing)}; "
            f"present keys: {sorted(batch.keys())}"
        )
