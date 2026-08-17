"""Typed boundaries shared by readers, agents, visualizers and evaluators.

Arrays are numpy at this boundary. Feature builders convert them to torch after
batching and device placement are known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.constants import (
    DEFAULT_PDM_WEIGHTS,
    EGO_SHAPE_IDX_LENGTH,
    EGO_SHAPE_IDX_WHEEL_BASE,
    EGO_SHAPE_IDX_WIDTH,
    FUTURE_STRIDE,
    PDM_COMPONENT_ORDER,
    T4_DEFAULT_CAMERA_NAMES,
    T4_INTERVAL_LENGTH,
    T4_LIDAR_POINT_DIM,
    T4_SUPPORTED_CAMERA_NAMES,
    TRAJECTORY_INTERVAL,
    TRAJECTORY_POSES,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

if TYPE_CHECKING:
    from t4_e2e_devkit.dataset.route import T4RouteMetadata
    from t4_e2e_devkit.dataset.scene_tags import T4SceneTag

# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #


@dataclass
class Camera:
    """One camera view at one frame.

    The extrinsic is camera-to-ego (base_link), matching the stored calibration.
    Camera coordinates are ``(x right, y down, z forward)``; ego coordinates are
    ``(x forward, y left, z up)``.

    To project an ego-frame point into the image::

        p_cam = (p_ego - camera2ego_translation) @ camera2ego_rotation
        uv    = (p_cam @ intrinsics.T)[:2] / p_cam[2]        # only if p_cam[2] > 0

    ``intrinsics`` is already rescaled to the resolution ``image`` is stored at,
    so no further scaling is needed.
    """

    name: str
    image: Optional[npt.NDArray[np.uint8]] = None  # [H, W, 3], RGB
    camera2ego_rotation: Optional[npt.NDArray[np.float64]] = None  # [3, 3]
    camera2ego_translation: Optional[npt.NDArray[np.float64]] = None  # [3]
    intrinsics: Optional[npt.NDArray[np.float64]] = None  # [3, 3], at image resolution
    distortion: Optional[npt.NDArray[np.float64]] = None
    timestamp_us: Optional[int] = None
    # Boxes moved to THIS camera's capture time and ego frame. Per-camera because
    # the correction is: the channels of one frame trail the LiDAR sweep by 50 to
    # 116 ms, differing by 66 ms between them, which is 0.3-0.8 m of displacement
    # at urban speed. ``None`` when the scene carries no per-sensor timestamps.
    annotations: Optional["Annotations"] = None

    @property
    def is_present(self) -> bool:
        """Whether this slot actually carries an image.

        A missing slot is not an error: the reader's default policy fills it
        with the ImageNet mean and the window stays trainable.  A model that
        cannot tolerate a hole says so through its data-list filter, not here.
        """
        return self.image is not None

    @property
    def is_calibrated(self) -> bool:
        """:return: whether this view can project between ego and image."""
        return (
            self.intrinsics is not None
            and self.camera2ego_rotation is not None
            and self.camera2ego_translation is not None
        )

    def ego_to_camera(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """
        Transform ego-frame points into this camera's frame.
        :param points: ``[N, 3]`` in ego (base_link) coordinates.
        :return: ``[N, 3]`` in camera coordinates; ``z > 0`` is in front.
        :raises ValueError: when the view carries no calibration.
        """
        if not self.is_calibrated:
            raise ValueError(f"camera {self.name!r} has no calibration to project with")
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rotation = np.asarray(self.camera2ego_rotation, dtype=np.float64)
        translation = np.asarray(self.camera2ego_translation, dtype=np.float64).reshape(3)
        # camera2ego is camera -> ego, so the inverse is a transpose and a shift.
        return (values - translation) @ rotation

    def project_to_image(
        self,
        points: npt.NDArray[np.floating],
        min_depth: float = 0.5,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """
        Project ego-frame points onto this camera's image plane.
        :param points: ``[N, 3]`` in ego (base_link) coordinates.
        :param min_depth: points closer than this are dropped, since a point at
            or behind the pinhole projects to a meaningless pixel rather than to
            no pixel.
        :return: ``([N, 2]`` pixel coordinates, ``[N]`` validity mask``)``.  The
            pixel array is full length with invalid rows left unspecified, so a
            caller can index it with the mask without re-aligning.
        """
        camera_points = self.ego_to_camera(points)
        depth = camera_points[:, 2]
        valid = depth > float(min_depth)

        pixels = np.zeros((camera_points.shape[0], 2), dtype=np.float64)
        if valid.any():
            homogeneous = camera_points[valid] @ np.asarray(self.intrinsics, dtype=np.float64).T
            pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
        return pixels, valid

    def image_bounds_mask(
        self,
        pixels: npt.NDArray[np.floating],
        valid: Optional[npt.NDArray[np.bool_]] = None,
    ) -> npt.NDArray[np.bool_]:
        """
        Which projected pixels fall inside this view's image.
        :param pixels: ``[N, 2]`` pixel coordinates.
        :param valid: optional depth-validity mask to combine with.
        :return: ``[N]`` boolean mask.
        """
        if self.image is None:
            raise ValueError(f"camera {self.name!r} carries no image to bound against")
        height, width = self.image.shape[:2]
        pixels = np.asarray(pixels, dtype=np.float64)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        return inside if valid is None else (inside & valid)


@dataclass
class Cameras:
    """The ordered camera register for one frame.

    Order is part of the learned camera-register contract, so this preserves
    insertion order and exposes it as :attr:`names`.  Access by name
    (``cameras["CAM_FRONT_WIDE"]``) rather than by position wherever the code
    means a specific view.
    """

    cameras: Dict[str, Camera] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Camera:
        return self.cameras[name]

    def __contains__(self, name: str) -> bool:
        return name in self.cameras

    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self):
        return iter(self.cameras.values())

    @property
    def names(self) -> List[str]:
        """:return: camera names in register order."""
        return list(self.cameras.keys())

    def stacked_images(self) -> npt.NDArray[np.uint8]:
        """:return: ``[N_cameras, H, W, 3]`` in register order."""
        images = [camera.image for camera in self.cameras.values()]
        if any(image is None for image in images):
            missing = [name for name, cam in self.cameras.items() if cam.image is None]
            raise ValueError(f"cannot stack images, missing views: {missing}")
        return np.stack(images, axis=0)

    @classmethod
    def empty(cls, names: Sequence[str] = T4_DEFAULT_CAMERA_NAMES) -> Cameras:
        """:return: a register with every named slot present but unfilled."""
        return cls({name: Camera(name=name) for name in names})


@dataclass
class Lidar:
    """A T4 LiDAR sweep: the scene's concatenated cloud at one frame."""

    lidar_pc: Optional[npt.NDArray[np.float32]] = None  # [N, 5]

    def __post_init__(self) -> None:
        if self.lidar_pc is None:
            return
        values = np.asarray(self.lidar_pc, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != T4_LIDAR_POINT_DIM:
            raise ValueError(
                f"T4 LiDAR points are [x, y, z, intensity, ring_or_time]; "
                f"got shape {values.shape}"
            )
        self.lidar_pc = np.ascontiguousarray(values)

    @property
    def is_present(self) -> bool:
        """:return: whether a cloud was decoded for this frame."""
        return self.lidar_pc is not None


# --------------------------------------------------------------------------- #
# Ego
# --------------------------------------------------------------------------- #


@dataclass
class EgoShape:
    """The ego footprint, read from the scene.

    T4 stores ``[wheel_base, length, width]`` per scene in
    ``derived/scalars.npz``.  It is never inferred: nuPlan's pacifica parameters
    are the wrong vehicle for every vehicle in this fleet, and the collision and
    comfort terms are both footprint-sensitive.
    """

    wheel_base: float
    length: float
    width: float

    @classmethod
    def from_array(cls, array: npt.NDArray[np.floating]) -> EgoShape:
        """:param array: ``[3]`` of ``(wheel_base, length, width)``."""
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        if values.shape[0] != 3:
            raise ValueError(f"ego_shape must be [wheel_base, length, width]; got shape {values.shape}")
        return cls(
            wheel_base=float(values[EGO_SHAPE_IDX_WHEEL_BASE]),
            length=float(values[EGO_SHAPE_IDX_LENGTH]),
            width=float(values[EGO_SHAPE_IDX_WIDTH]),
        )

    def as_array(self) -> npt.NDArray[np.float64]:
        """:return: ``[3]`` of ``(wheel_base, length, width)``."""
        return np.array([self.wheel_base, self.length, self.width], dtype=np.float64)

    @property
    def half_length(self) -> float:
        """:return: half the vehicle length."""
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        """:return: half the vehicle width."""
        return self.width / 2.0

    @property
    def rear_axle_to_center(self) -> float:
        """Longitudinal offset from the rear axle to the footprint centre.

        The scorer works in centre coordinates while poses are rear-axle, so
        this shift appears in every corner computation.
        """
        return self.length / 2.0 - (self.length - self.wheel_base) / 2.0


@dataclass
class EgoStatus:
    """Ego kinematics at one frame, in the centre frame of the window.

    ``ego_pose``/``ego_velocity``/``ego_acceleration`` are differenced from the
    pose history by one shared function so training and rollout agree; see
    :class:`~t4_e2e_devkit.common.enums.EgoStatusIndex` for why the recorded EKF
    arrays are deliberately not used here.

    ``control_state`` carries the recorded EKF estimate for deployment, where
    the on-vehicle controller needs the vehicle's own numbers.
    """

    ego_pose: npt.NDArray[np.float32]  # [3] x, y, heading
    ego_velocity: npt.NDArray[np.float32]  # [2] vx, vy
    ego_acceleration: npt.NDArray[np.float32]  # [2] ax, ay
    ego_shape: EgoShape
    driving_command: Optional[npt.NDArray[np.int64]] = None
    turn_indicator: Optional[int] = None
    control_state: Optional[Mapping[str, Any]] = None
    in_global_frame: bool = False

    def as_array(self) -> npt.NDArray[np.float32]:
        """:return: ``[7]`` of ``(x, y, heading, vx, vy, ax, ay)``."""
        return np.concatenate(
            [
                np.asarray(self.ego_pose, dtype=np.float32).reshape(3),
                np.asarray(self.ego_velocity, dtype=np.float32).reshape(2),
                np.asarray(self.ego_acceleration, dtype=np.float32).reshape(2),
            ]
        )

    @property
    def speed(self) -> float:
        """:return: planar speed in m/s."""
        return float(np.hypot(self.ego_velocity[0], self.ego_velocity[1]))


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #


def _portable_source_label(value: Optional[str]) -> Optional[str]:
    """Keep local filesystem prefixes out of serialized map metadata."""
    if value is None:
        return None
    return Path(value).name


@dataclass(frozen=True)
class MapObjectMatch:
    """One scene-local map row and its source-object matching evidence."""

    layer: str
    row_index: int
    source_object_id: Optional[str]
    source_path: Optional[str]
    frame_index: Optional[int]
    match_distance_m: Optional[float]
    candidate_ids: Tuple[str, ...] = ()
    reason: str = "matched"

    @property
    def matched(self) -> bool:
        """:return: whether the row was assigned a real source object ID."""
        return self.source_object_id is not None

    def as_dict(self) -> Dict[str, Any]:
        """:return: a JSON-compatible audit record."""
        return {
            "layer": self.layer,
            "row_index": self.row_index,
            "source_object_id": self.source_object_id,
            "source_path": _portable_source_label(self.source_path),
            "frame_index": self.frame_index,
            "match_distance_m": self.match_distance_m,
            "candidate_ids": list(self.candidate_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MapObjectIds:
    """Optional IDs parallel to the scene-local vector-map rows.

    IDs are deliberately side metadata rather than extra tensor channels. A
    model still receives the unchanged numeric map contract, while planners,
    visualizers and audits can recover stable Lanelet2/route identities when
    the source map is available. ``None`` marks padding or an object that could
    not be matched without inventing an ID.
    """

    lane_ids: Tuple[Optional[str], ...] = ()
    route_lane_ids: Tuple[Optional[str], ...] = ()
    polygon_ids: Tuple[Optional[str], ...] = ()
    line_string_ids: Tuple[Optional[str], ...] = ()
    source_path: Optional[str] = None
    frame_index: Optional[int] = None
    matches: Tuple[MapObjectMatch, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        """:return: IDs and matching evidence in a JSON-compatible mapping."""
        return {
            "lane_ids": list(self.lane_ids),
            "route_lane_ids": list(self.route_lane_ids),
            "polygon_ids": list(self.polygon_ids),
            "line_string_ids": list(self.line_string_ids),
            "source_path": _portable_source_label(self.source_path),
            "frame_index": self.frame_index,
            "matches": [match.as_dict() for match in self.matches],
        }

    def write_json(self, path: str | Path) -> str:
        """Write an explicit audit sidecar and return its path.

        The caller chooses the output location. Runtime artifacts should live
        under an ignored results/cache directory rather than beside source
        data or in the repository.
        """
        import json

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return str(output)


@dataclass
class MapTensors:
    """The vector map for one frame, in the ego frame of that frame.

    A T4 scene ships its own map slice per frame, so there is nothing to query --
    the geometry that matters
    is already cropped to the window and expressed in the right frame.  Missing
    map fields are a hard error at the reader, never silently zero-filled: a
    camera-only run that quietly dropped the route would still train, and
    produce a model that cannot follow one.

    Column layout of a lane/route point is documented in
    :mod:`t4_e2e_devkit.common.constants`.
    """

    lanes: npt.NDArray[np.float32]  # [140, 20, 33]
    lanes_speed_limit: npt.NDArray[np.float32]  # [140, 1]
    lanes_has_speed_limit: npt.NDArray[np.bool_]  # [140, 1]
    route_lanes: npt.NDArray[np.float32]  # [25, 20, 33]
    route_lanes_speed_limit: npt.NDArray[np.float32]  # [25, 1]
    route_lanes_has_speed_limit: npt.NDArray[np.bool_]  # [25, 1]
    polygons: npt.NDArray[np.float32]  # [10, 40, 3]
    line_strings: npt.NDArray[np.float32]  # [60, 20, 4]
    object_ids: Optional[MapObjectIds] = None

    def as_dict(self) -> Dict[str, npt.NDArray]:
        """:return: the map fields keyed by their contract names."""
        return {
            "lanes": self.lanes,
            "lanes_speed_limit": self.lanes_speed_limit,
            "lanes_has_speed_limit": self.lanes_has_speed_limit,
            "route_lanes": self.route_lanes,
            "route_lanes_speed_limit": self.route_lanes_speed_limit,
            "route_lanes_has_speed_limit": self.route_lanes_has_speed_limit,
            "polygons": self.polygons,
            "line_strings": self.line_strings,
        }

    @property
    def has_route(self) -> bool:
        """:return: whether any route lane carries geometry."""
        return bool(np.any(np.abs(self.route_lanes) > 0))


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


@dataclass
class Annotations:
    """Tracked objects at one frame, in that frame's ego coordinates.

    ``boxes`` follows :class:`~t4_e2e_devkit.common.enums.T4BoxIndex`:
    ``[x, y, z, width, length, height, yaw, vx, vy]``.
    """

    boxes: npt.NDArray[np.float32]  # [M, 9] see T4BoxIndex
    labels: npt.NDArray[np.int64]  # [M]
    track_tokens: Optional[List[str]] = None
    velocities: Optional[npt.NDArray[np.float32]] = None  # [M, 2]; else columns 7-8

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if boxes.ndim != 2 or boxes.shape[1] < 7:
            raise ValueError(
                "annotation boxes must be [M, >=7] following T4BoxIndex "
                f"(x, y, z, width, length, height, yaw[, vx, vy]); got {boxes.shape}"
            )
        if len(labels) != len(boxes):
            raise ValueError(
                f"annotation labels ({len(labels)}) and boxes ({len(boxes)}) disagree"
            )
        if self.track_tokens is not None and len(self.track_tokens) != len(boxes):
            raise ValueError(
                f"annotation track_tokens ({len(self.track_tokens)}) and boxes "
                f"({len(boxes)}) disagree"
            )
        if self.velocities is not None:
            velocities = np.asarray(self.velocities, dtype=np.float32)
            if velocities.shape != (len(boxes), 2):
                raise ValueError(
                    f"annotation velocities must be [M, 2] for M={len(boxes)}; "
                    f"got {velocities.shape}"
                )
            self.velocities = np.ascontiguousarray(velocities)
        self.boxes = np.ascontiguousarray(boxes)
        self.labels = np.ascontiguousarray(labels)
        if self.track_tokens is not None:
            self.track_tokens = list(self.track_tokens)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @classmethod
    def empty(cls) -> Annotations:
        """:return: an annotation set with no objects.

        Only for scenes that genuinely contain none.  A scene whose GT fields
        are *absent* must raise at the reader instead -- treating missing GT as
        an empty traffic scene silently teaches the model that the road is
        clear.
        """
        return cls(
            boxes=np.zeros((0, 9), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #

DEFAULT_TRAJECTORY_SAMPLING = TrajectorySampling(
    num_poses=TRAJECTORY_POSES,
    interval_length=TRAJECTORY_INTERVAL,
)


@dataclass
class Trajectory:
    """A planned or recorded ego trajectory in local coordinates.

    Pose ``i`` is at ``(i + 1) * interval_length`` seconds.  The current ego
    pose at ``t=0`` is implicit and is not stored in :attr:`poses`.
    """

    poses: npt.NDArray[np.float32]  # [num_poses, 3] x, y, heading
    trajectory_sampling: TrajectorySampling = DEFAULT_TRAJECTORY_SAMPLING

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses, dtype=np.float32)
        if poses.ndim != 2:
            raise ValueError(f"Trajectory poses need two dimensions (samples, pose); got {poses.shape}")
        if poses.shape[0] != self.trajectory_sampling.num_poses:
            raise ValueError(
                f"Trajectory has {poses.shape[0]} poses but its sampling declares "
                f"{self.trajectory_sampling.num_poses}"
            )
        if poses.shape[1] != 3:
            raise ValueError(f"Trajectory poses need (x, y, heading) at the last dim; got {poses.shape}")
        # The dataclass is the numpy boundary.  Normalising here keeps callers
        # that pass lists or float64 arrays from getting a delayed AttributeError
        # in ``__len__`` or silently changing dtype in the scorer.
        self.poses = np.ascontiguousarray(poses)

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    @property
    def timestamps(self) -> npt.NDArray[np.float32]:
        """Time of every pose, in seconds from the current ego state."""

        return np.arange(1, len(self) + 1, dtype=np.float32) * float(
            self.trajectory_sampling.interval_length
        )

    @property
    def duration(self) -> float:
        """:return: declared future horizon in seconds."""

        return float(self.trajectory_sampling.time_horizon)

    def resample(
        self,
        target_sampling: TrajectorySampling,
        *,
        allow_extrapolation: bool = False,
    ) -> "Trajectory":
        """Resample this trajectory on another uniform time grid.

        Positions use linear interpolation.  Headings are unwrapped before
        interpolation, so a path crossing ``-pi/pi`` takes the short turn.
        The current pose is included as a known knot at ``t=0``.  Extrapolation
        is opt-in because extending a plan beyond its declared horizon changes
        its meaning.
        """

        source_interval = float(self.trajectory_sampling.interval_length)
        target_interval = float(target_sampling.interval_length)
        source_times = np.arange(len(self) + 1, dtype=np.float64) * source_interval
        target_times = np.arange(1, target_sampling.num_poses + 1, dtype=np.float64) * target_interval
        source_poses = np.vstack((np.zeros((1, 3), dtype=np.float64), self.poses.astype(np.float64)))

        if target_times.size:
            source_horizon = float(source_times[-1])
            target_horizon = float(target_times[-1])
            if target_horizon > source_horizon + 1e-9 and not allow_extrapolation:
                raise ValueError(
                    "cannot resample a trajectory beyond its horizon: "
                    f"source={source_horizon:g}s, target={target_horizon:g}s"
                )

        def interpolate(values: np.ndarray) -> np.ndarray:
            if not allow_extrapolation:
                return np.interp(target_times, source_times, values)
            result = np.empty(target_times.shape, dtype=np.float64)
            for index, time in enumerate(target_times):
                if time <= source_times[-1]:
                    result[index] = np.interp(time, source_times, values)
                    continue
                slope = (values[-1] - values[-2]) / source_interval
                result[index] = values[-1] + slope * (time - source_times[-1])
            return result

        xy = np.stack(
            [interpolate(source_poses[:, column]) for column in (0, 1)], axis=-1
        )
        unwrapped_heading = np.unwrap(source_poses[:, 2])
        heading = interpolate(unwrapped_heading)
        poses = np.column_stack((xy, heading)).astype(np.float32)
        return Trajectory(poses=poses, trajectory_sampling=target_sampling)


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #


@dataclass
class SceneMetadata:
    """Identity and window shape of one training/evaluation sample."""

    scene_dir: str  # path relative to the dataset root
    scene_id: str  # the scene's own identifier
    center_frame: int  # index of the current frame within the scene
    num_history_frames: int
    num_future_frames: int
    vehicle: Optional[str] = None
    date: Optional[str] = None
    timestamps_us: Optional[npt.NDArray[np.int64]] = None
    global_center_pose: Optional[npt.NDArray[np.float64]] = None  # [4] x, y, cos, sin
    # External taxonomy is optional and stays metadata-only. It is never fed
    # into the model feature contract unless an agent explicitly requests it.
    scene_tags: Tuple["T4SceneTag", ...] = field(default_factory=tuple)
    route_metadata: Optional["T4RouteMetadata"] = None

    @property
    def token(self) -> str:
        """A stable identifier for this window."""
        return f"{self.scene_dir}@{self.center_frame}"


@dataclass
class T4Frame:
    """One frame of a T4 window."""

    frame_index: int
    timestamp_us: int
    ego_status: EgoStatus
    map_tensors: Optional[MapTensors] = None
    annotations: Optional[Annotations] = None
    cameras: Optional[Cameras] = None
    lidar: Optional[Lidar] = None


@dataclass
class T4Scene:
    """A full window: history, current frame, and recorded future.

    This is the privileged view -- it holds the future GT that an agent must
    not see.  Agents receive :class:`T4AgentInput` from
    :meth:`get_agent_input`; the scene itself goes to the target builders and
    the scorer.
    """

    scene_metadata: SceneMetadata
    frames: List[T4Frame]
    future_ego_poses: Optional[npt.NDArray[np.float32]] = None  # [F, 3] in centre frame
    future_annotations: Optional[List[Annotations]] = None  # per future frame
    goal_pose: Optional[npt.NDArray[np.float32]] = None  # [4] x, y, cos, sin
    pdm_progress: Optional[float] = None  # PDM-Closed reference progress
    # The PDM-Closed path that earned that progress, [51, 3] (x, y, heading) in
    # the centre frame -- PDM simulates from a rear axle at the origin, so it
    # needs no transform. Present when reference loading is requested.
    pdm_reference_poses: Optional[npt.NDArray[np.float32]] = None

    @property
    def current_frame_index(self) -> int:
        """Index of the current frame within :attr:`frames`."""
        return self.scene_metadata.num_history_frames - 1

    @property
    def current_frame(self) -> T4Frame:
        """:return: the frame the agent is planning from."""
        return self.frames[self.current_frame_index]

    def get_history_poses(self) -> npt.NDArray[np.float32]:
        """The ego's own recorded past, in current-frame coordinates.

        Useful next to a prediction: a plan that ignores where the vehicle just
        came from usually shows up as a kink at the origin, which is invisible
        without the history drawn.

        :return: ``[num_history_frames, 3]`` of ``(x, y, heading)``, oldest first
            and ending at the origin.
        """
        return np.stack(
            [
                np.asarray(frame.ego_status.ego_pose, dtype=np.float32)
                for frame in self.frames[: self.current_frame_index + 1]
            ]
        )

    def get_pdm_reference_trajectory(self) -> Optional[Trajectory]:
        """The PDM-Closed reference path, as a trajectory.

        This is the ego-progress denominator's own path -- what the rule-based
        planner did on this window. Drawing it beside a model's plan is how an EP
        of 0.5 becomes legible: it says the model travelled half as far as
        PDM-Closed managed, and shows where the two diverged.

        :return: the reference trajectory, or ``None`` when reference loading is disabled.
        """
        if self.pdm_reference_poses is None:
            return None
        poses = np.asarray(self.pdm_reference_poses, dtype=np.float32)
        # The cached path includes its origin; a Trajectory does not.
        poses = poses[1:] if len(poses) > 1 else poses
        return Trajectory(
            poses=poses,
            trajectory_sampling=TrajectorySampling(
                num_poses=len(poses), interval_length=T4_INTERVAL_LENGTH
            ),
        )

    def get_agent_input(self) -> T4AgentInput:
        """:return: the non-privileged view an agent is allowed to see."""
        history = self.frames[: self.current_frame_index + 1]
        current = self.current_frame
        return T4AgentInput(
            ego_statuses=[frame.ego_status for frame in history],
            cameras=[frame.cameras for frame in history],
            lidars=[frame.lidar for frame in history],
            map_tensors=current.map_tensors,
            goal_pose=self.goal_pose,
            scene_metadata=self.scene_metadata,
        )

    def get_future_trajectory(
        self,
        num_poses: Optional[int] = None,
        stride: Optional[int] = None,
        trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Trajectory:
        """Recorded future of the human driver, in current-frame coordinates.

        ``trajectory_sampling`` selects the output grid.  If it is omitted,
        ``num_poses`` and ``stride`` keep the default contract.  The source
        future is sampled at the dataset rate and is interpolated when the
        requested interval is not an integer number of source frames.

        :param num_poses: number of poses; defaults to the contract's eight.
        :param stride: source frames per pose; defaults to the contract's five
            (0.5 s at 10 Hz).
        :param trajectory_sampling: explicit output sampling.  It cannot be
            combined with ``num_poses`` or ``stride``.
        :return: trajectory dataclass.
        """
        if self.future_ego_poses is None:
            raise ValueError(
                f"scene {self.scene_metadata.token} carries no future poses; "
                "it was read with a filter that excluded them"
            )
        if trajectory_sampling is not None and (num_poses is not None or stride is not None):
            raise ValueError(
                "trajectory_sampling cannot be combined with num_poses or stride"
            )
        if trajectory_sampling is None:
            num_poses = TRAJECTORY_POSES if num_poses is None else int(num_poses)
            stride = FUTURE_STRIDE if stride is None else int(stride)
            if num_poses <= 0 or stride <= 0:
                raise ValueError("num_poses and stride must be positive")
            trajectory_sampling = TrajectorySampling(
                num_poses=num_poses,
                interval_length=stride * T4_INTERVAL_LENGTH,
            )

        source_poses = np.asarray(self.future_ego_poses, dtype=np.float32)
        source_sampling = TrajectorySampling(
            num_poses=source_poses.shape[0],
            interval_length=T4_INTERVAL_LENGTH,
        )
        source = Trajectory(poses=source_poses, trajectory_sampling=source_sampling)
        try:
            return source.resample(trajectory_sampling)
        except ValueError as error:
            raise ValueError(
                f"scene {self.scene_metadata.token} cannot provide "
                f"{trajectory_sampling.num_poses} poses over "
                f"{trajectory_sampling.time_horizon:g}s from "
                f"{len(source_poses)} future frames"
            ) from error


@dataclass
class T4AgentInput:
    """Everything an agent may read at inference time.

    Deliberately contains no future information.  ``ego_statuses``,
    ``cameras`` and ``lidars`` are ordered oldest-to-current and are the same
    length; a sensor the agent's :class:`SensorConfig` did not request is
    ``None`` at that step rather than absent, so indexing stays uniform.
    """

    ego_statuses: List[EgoStatus]
    cameras: List[Optional[Cameras]]
    lidars: List[Optional[Lidar]]
    map_tensors: Optional[MapTensors] = None
    goal_pose: Optional[npt.NDArray[np.float32]] = None
    scene_metadata: Optional[SceneMetadata] = None

    def __post_init__(self) -> None:
        n = len(self.ego_statuses)
        if len(self.cameras) != n or len(self.lidars) != n:
            raise ValueError(
                f"T4AgentInput streams disagree in length: {n} ego statuses, "
                f"{len(self.cameras)} camera frames, {len(self.lidars)} lidar frames"
            )

    @property
    def ego_status(self) -> EgoStatus:
        """:return: the current-frame ego status."""
        return self.ego_statuses[-1]

    @property
    def num_history_frames(self) -> int:
        """:return: number of frames including the current one."""
        return len(self.ego_statuses)


# --------------------------------------------------------------------------- #
# Filters and sensor configuration
# --------------------------------------------------------------------------- #


@dataclass
class SensorConfig:
    """Which sensor streams to decode, and at which history steps.

    Decoding T4 sensors is the dominant cost of the input pipeline -- five JPEG
    views per frame, or a LiDAR sweep -- so an agent declaring what it needs is
    what makes a camera model and a LiDAR model share one reader without either
    paying for the other's data.

    Values are ``bool`` (all history steps or none) or ``list[int]`` (specific
    history step indices, where ``-1`` is the current frame).  ``cameras`` is
    keyed by camera name; the public input boundary currently accepts only the
    JPEG-backed wide channels, while the full on-disk register remains available
    to schema inspection.
    """

    cameras: Dict[str, Union[bool, List[int]]] = field(default_factory=dict)
    lidar: Union[bool, List[int]] = False

    def __post_init__(self) -> None:
        supported = {name.upper() for name in T4_SUPPORTED_CAMERA_NAMES}
        unsupported = [name for name in self.cameras if name.upper() not in supported]
        if unsupported:
            raise ValueError(
                f"unsupported T4 cameras: {unsupported}. "
                "Only JPEG-backed wide cameras are supported."
            )

    def camera_names_at(self, iteration: int) -> List[str]:
        """
        :param iteration: history step index; ``-1`` denotes the current frame.
        :return: camera names to decode at that step, in register order.
        """
        return [
            name
            for name, include in self.cameras.items()
            if (include is True) or (isinstance(include, list) and iteration in include)
        ]

    def lidar_at(self, iteration: int) -> bool:
        """
        :param iteration: history step index; ``-1`` denotes the current frame.
        :return: whether to decode the LiDAR sweep at that step.
        """
        if isinstance(self.lidar, bool):
            return self.lidar
        return iteration in self.lidar

    @property
    def any_camera(self) -> bool:
        """:return: whether any camera is requested at any step."""
        return any(bool(include) for include in self.cameras.values())

    @property
    def any_lidar(self) -> bool:
        """:return: whether LiDAR is requested at any step."""
        return bool(self.lidar)

    @classmethod
    def build_no_sensors(cls) -> SensorConfig:
        """:return: a configuration that decodes nothing (map/ego-only agents)."""
        return cls(cameras={}, lidar=False)

    @classmethod
    def build_current_frame(
        cls,
        camera_names: Optional[Sequence[str]] = None,
        lidar: bool = False,
    ) -> SensorConfig:
        """Current frame only -- the common case for single-timestep models.

        :param camera_names: cameras to decode, in register order.  ``None``
            falls back to the wide-five profile; to take whatever a given scene
            actually stores, use
            :func:`t4_e2e_devkit.dataset.rigs.sensor_config_for_scene` instead,
            since no single register fits every rig in the fleet.
        :param lidar: whether to decode the current LiDAR sweep.
        :return: sensor configuration dataclass.
        """
        names = T4_DEFAULT_CAMERA_NAMES if camera_names is None else camera_names
        return cls(
            cameras={name: [-1] for name in names},
            lidar=[-1] if lidar else False,
        )

    @classmethod
    def build_all_sensors(
        cls, camera_names: Optional[Sequence[str]] = None
    ) -> SensorConfig:
        """
        :param camera_names: cameras to decode; the wide-five profile by default.
        :return: every named camera and LiDAR at every history step.
        """
        names = T4_DEFAULT_CAMERA_NAMES if camera_names is None else camera_names
        return cls(cameras={name: True for name in names}, lidar=True)


@dataclass
class SceneFilter:
    """Which windows a run reads, and how much of each.

    The window shape lives here rather than in the agent so that a data list, a
    metric cache and a training run can be checked against one another.
    """

    num_history_frames: int = 31
    num_future_frames: int = 80
    frame_interval: int = 5
    has_route: bool = True
    max_scenes: Optional[int] = None
    scene_dirs: Optional[List[str]] = None
    dates: Optional[List[str]] = None
    vehicles: Optional[List[str]] = None
    max_window_gap_frames: Optional[int] = None
    require_cameras: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.num_history_frames < 1:
            raise ValueError("SceneFilter.num_history_frames must be at least 1")
        if self.num_future_frames < 0:
            raise ValueError("SceneFilter.num_future_frames must be non-negative")
        if self.frame_interval < 1:
            raise ValueError("SceneFilter.frame_interval must be at least 1")

    @property
    def num_frames(self) -> int:
        """:return: total frames spanned by one window."""
        return self.num_history_frames + self.num_future_frames


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class PDMResults:
    """One window's PDM score and its components.

    The multiplicative terms gate a weighted average of the rest, so a collision
    or drivable-area violation zeroes the score regardless of the other terms::

        PDMS = NC * DAC * (5*EP + 5*TTC + 2*Comfort + 0*DDC) / 12

    ``tier4_metrics`` is retained as an optional per-window compatibility
    field.  The canonical family-separated report aggregates it through
    :func:`t4_e2e_devkit.evaluation.tier4_metrics.aggregate_tier4_metrics`;
    T4 terms never alter this PDM result or its aggregate.
    """

    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    time_to_collision_within_bound: float
    ego_progress: float
    comfort: float
    score: float

    token: Optional[str] = None
    tier4_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def components(self) -> Dict[str, float]:
        """:return: the six PDM components keyed by their short names."""
        return dict(
            zip(
                PDM_COMPONENT_ORDER,
                (
                    self.no_at_fault_collisions,
                    self.drivable_area_compliance,
                    self.driving_direction_compliance,
                    self.time_to_collision_within_bound,
                    self.ego_progress,
                    self.comfort,
                ),
                strict=True,
            )
        )

    @classmethod
    def from_components(
        cls,
        components: Sequence[float],
        score: Optional[float] = None,
        token: Optional[str] = None,
        weights: Sequence[float] = DEFAULT_PDM_WEIGHTS,
        tier4_metrics: Optional[Dict[str, float]] = None,
    ) -> PDMResults:
        """Build from a ``[nc, dac, ddc, ttc, ep, comfort]`` vector.

        :param components: the six components, in :data:`PDM_COMPONENT_ORDER`.
        :param score: aggregate; recomputed from ``weights`` when omitted.
        :param token: identifier of the scored window.
        :param weights: scoring profile; defaults to :data:`DEFAULT_PDM_WEIGHTS`.
        :param tier4_metrics: optional compatibility copy of T4 metrics.
        :return: results dataclass.
        """
        if len(components) != len(PDM_COMPONENT_ORDER):
            raise ValueError(
                f"expected {len(PDM_COMPONENT_ORDER)} components {PDM_COMPONENT_ORDER}; "
                f"got {len(components)}"
            )
        nc, dac, ddc, ttc, ep, comfort = (float(value) for value in components)
        if score is None:
            score = aggregate_pdm_score((nc, dac, ddc, ttc, ep, comfort), weights)
        return cls(
            no_at_fault_collisions=nc,
            drivable_area_compliance=dac,
            driving_direction_compliance=ddc,
            time_to_collision_within_bound=ttc,
            ego_progress=ep,
            comfort=comfort,
            score=float(score),
            token=token,
            tier4_metrics=dict(tier4_metrics or {}),
        )


def aggregate_pdm_score(
    components: Sequence[float],
    weights: Sequence[float] = DEFAULT_PDM_WEIGHTS,
) -> float:
    """Aggregate six PDM components into one score.

    NC and DAC are multiplicative gates; DDC, TTC, EP and Comfort enter a
    weighted average normalised by the sum of their weights.  Keeping the
    aggregation here prevents the GPU scorer, CPU judge and reports from drifting.

    :param components: ``[nc, dac, ddc, ttc, ep, comfort]``.
    :param weights: the same six positions; the NC and DAC entries are ignored
        because those terms multiply rather than average.
    :return: the aggregate score.
    """
    nc, dac, ddc, ttc, ep, comfort = (float(value) for value in components)
    _, _, w_ddc, w_ttc, w_ep, w_comfort = (float(value) for value in weights)

    multiplicative = nc * dac
    weight_sum = w_ddc + w_ttc + w_ep + w_comfort
    if weight_sum <= 0.0:
        return multiplicative
    weighted = (w_ddc * ddc + w_ttc * ttc + w_ep * ep + w_comfort * comfort) / weight_sum
    return multiplicative * weighted


def aggregate_pdm_results(results: Sequence[PDMResults]) -> Dict[str, float]:
    """Average only the PDM family over per-window results.

    :param results: per-window scores.
    :return: mean of every PDM component and aggregate, plus the count.
    """
    if not results:
        return {"num_scenes": 0.0}
    report: Dict[str, float] = {"num_scenes": float(len(results))}
    for name in PDM_COMPONENT_ORDER:
        report[name] = float(np.mean([result.components[name] for result in results]))
    report["score"] = float(np.mean([result.score for result in results]))
    return report


def aggregate_results(results: Sequence[PDMResults]) -> Dict[str, float]:
    """Backward-compatible name for :func:`aggregate_pdm_results`.

    This function intentionally returns PDM fields only.  Use
    :func:`t4_e2e_devkit.evaluation.report.aggregate_evaluation` when more than
    one metric family is needed.
    """

    return aggregate_pdm_results(results)
