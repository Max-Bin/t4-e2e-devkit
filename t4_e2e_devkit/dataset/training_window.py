"""Flat training-window assembly for the direct T4 scene layout.

:mod:`t4_e2e_devkit.dataset.window` assembles the rich :class:`T4Scene` object
that agents and the evaluator consume.  A training loader wants something
flatter and cheaper: one numpy dict per ``(scene_dir, center_frame)`` window,
read with one bundle pread and one LiDAR pread, with no camera decode and no
map-API objects.  This module owns that product.  It was moved here verbatim
from the OnePlanner training repository so both repositories read T4 scenes
through ONE implementation — the window a model trains on and the window the
scorer replays are assembled by the same code.

The contract of one extracted window (all numpy, no batch dimension):

===========================  ==========================================
``ego_agent_past``           ``[PAST_FRAMES, 4]`` x, y, cos, sin
``ego_current_state``        ``[10]`` x,y,cos,sin,vx,vy,ax,ay,steer,yaw_rate
``ego_agent_future``         ``[FUTURE_FRAMES, 3]`` dx, dy, dheading
``lanes`` / ``route_lanes``  fixed map tensors at the centre frame
``polygons`` / ``line_strings`` / speed-limit companions
``ego_shape``                ``[3]`` wheel_base, length, width
``turn_indicators``          ``[PAST_FRAMES]``
``goal_pose``                ``[4]`` in the centre frame
``points``                   ``[N, 5]`` centre-frame LiDAR (optional)
``gt_bboxes_3d/gt_labels_3d`` centre-frame agent GT (optional)
``scene_key`` / ``scene_center``
===========================  ==========================================

Everything is expressed in the centre frame's ego coordinates; the transforms
below are the reference implementation the golden windows pin bit-for-bit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from t4_e2e_devkit.common.constants import (
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    SEGMENT_POINT_DIM,
)
from t4_e2e_devkit.common.temporal import DEFAULT_TEMPORAL_SPEC, TemporalSpec
from t4_e2e_devkit.dataset.scene import T4BundleReader, T4LidarPackReader

log = logging.getLogger(__name__)

#: Per-frame map fields the training window reads from ``derived/frames.pack``.
TRAINING_FRAME_FIELDS = (
    "lanes",
    "lanes_speed",
    "lanes_has_speed",
    "route",
    "route_speed",
    "route_has_speed",
    "polygons",
    "lines",
)

#: Whole-scene arrays the training window reads from ``derived/scalars.npz``.
TRAINING_SCALAR_FIELDS = ("trajectory", "velocity", "turn", "goal", "shape")


def expected_map_shapes() -> dict[str, tuple[int, ...]]:
    """Shapes of the required fixed-size fields in the current T4 bundle."""

    return {
        "lanes": (NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM),
        "lanes_speed": (NUM_SEGMENTS_IN_LANE, 1),
        "lanes_has_speed": (NUM_SEGMENTS_IN_LANE, 1),
        "route": (NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM),
        "route_speed": (NUM_SEGMENTS_IN_ROUTE, 1),
        "route_has_speed": (NUM_SEGMENTS_IN_ROUTE, 1),
        "polygons": (NUM_POLYGONS, POINTS_PER_POLYGON, 3),
        "lines": (NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 4),
    }


def valid_window_centers(
    n_frames: int,
    valid_mask: np.ndarray | None = None,
    stride: int = 1,
    *,
    spec: TemporalSpec = DEFAULT_TEMPORAL_SPEC,
) -> range | tuple[()]:
    """Return structurally valid center frames for one T4 scene.

    Current T4 scenes are contiguous.  A ``valid_mask`` is still accepted to
    honor trimmed edges and to reject a corrupt mask with an interior hole.

    ``stride`` subsamples the centers (a data-list construction choice,
    distinct from ``spec.frame_stride``, which is the model-rate sampling
    *within* one window).  At the default of 1 every frame is a window
    center, so consecutive samples are one frame apart and share almost all
    of their future — they are very nearly the same training pair.  Coverage
    is unchanged by striding: every part of every scene is still reachable,
    just sampled every ``stride`` frames; what drops is the count of
    near-duplicate pairs.
    """

    if n_frames < spec.min_source_frames:
        return ()
    mask = np.ones(n_frames, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if len(mask) != n_frames:
        raise ValueError(f"valid_mask has {len(mask)} entries, expected n_frames={n_frames}")
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return ()
    first_valid, last_valid = int(valid[0]), int(valid[-1])
    if not mask[first_valid : last_valid + 1].all():
        holes = int((~mask[first_valid : last_valid + 1]).sum())
        raise ValueError(
            f"T4 valid_mask contains {holes} interior invalid frame(s) in "
            f"[{first_valid}, {last_valid}]"
        )

    first = max(spec.history_span, first_valid + spec.history_span)
    last = min(n_frames - 1, last_valid) - spec.future_span
    if last < first:
        return ()
    step = max(1, int(stride))
    return range(first, last + 1, step)


class _FrameView:
    """Array-like view of ONE field inside a scene's frame bundle.

    The window builder indexes per-frame map fields with the centre index and
    checks ``ndim``/``shape``, so this exposes exactly that surface (shape is
    reported as the full ``[T, ...]`` scene array).  All views of a scene
    share one :class:`T4BundleReader`, which caches the last decoded frame —
    so reading every field of a window costs ONE pread + ONE decode.
    """

    def __init__(self, bundle: T4BundleReader, field: str, n_frames: int):
        self._b = bundle
        self._f = field
        spec = bundle.field_spec[field]
        self.shape = (n_frames,) + tuple(spec["shape"])
        self.ndim = len(self.shape)
        self.dtype = np.dtype(spec["dtype"])

    def __getitem__(self, i):
        if not isinstance(i, (int, np.integer)):
            raise TypeError("_FrameView supports single-frame indexing only")
        return self._b.frame(int(i))[self._f]

    def __len__(self):
        return self.shape[0]


class _GTByFrame:
    """``scene["gt_boxes"]["frame_0042"]`` -> that frame's boxes."""

    def __init__(self, view: _FrameView):
        self._v = view

    def __contains__(self, key) -> bool:
        try:
            return 0 <= int(str(key).split("_")[-1]) < len(self._v)
        except ValueError:
            return False

    def __getitem__(self, key):
        return self._v[int(str(key).split("_")[-1])]

    def get(self, key, default=None):
        return self[key] if key in self else default


class _EmptyByFrame:
    def get(self, key, default=None):
        return default

    def __contains__(self, key) -> bool:
        return False


class _LazyLidarFrames:
    """Open and decode T4 LiDAR frames on demand for one scene.

    Wraps :class:`T4LidarPackReader` with the scene's ``meta.json`` frame
    range: ``lidar_first_frame``/``lidar_frames`` bound which scene indices
    carry a sweep, and ``frame_offset`` maps a scene index to its pack index.
    A scene index outside the range returns ``None`` (the caller substitutes
    an empty sweep) rather than raising — trimmed scene edges are data, not
    errors, on the training path.

    On demand means on demand: the pack is opened, and its range checked,
    at the first sweep actually read.  Opening it in the constructor made a
    camera/map-only run fail on a scene that ships no pack, even though
    ``TrainingWindowBuilder(load_points=False)`` would never have read one --
    and every scene handle held an open descriptor it might not use.  A scene
    whose metadata declares no pack at all has no sweeps, which is a fact about
    the export rather than an error.
    """

    def __init__(self, meta: dict, root: Path):
        self._reader: T4LidarPackReader | None = None
        pack = meta.get("lidar_pack")
        self._path: Path | None = None
        if pack is not None:
            path = Path(pack)
            self._path = path if path.is_absolute() else root / path
        self._first = int(meta.get("lidar_first_frame", 0))
        self._count = int(meta.get("lidar_frames") or 0)
        self._offset = int(meta.get("frame_offset", 0))
        if self._first < 0 or (self._path is not None and self._count <= 0):
            raise ValueError("T4 metadata has an invalid LiDAR frame range")

    def _open(self) -> T4LidarPackReader:
        """Open the pack and check the scene's range against it, once.

        :raises ValueError: when the scene declares no pack.  ``frame`` returns
            ``None`` before reaching here, so this states the invariant rather
            than handling a case -- and a future caller that opens directly gets
            a sentence instead of a TypeError inside shapely.
        """

        if self._path is None:
            raise ValueError("this scene declares no LiDAR pack; there is nothing to open")
        if self._reader is None:
            reader = T4LidarPackReader(self._path)
            low = self._first + self._offset
            high = self._first + self._count - 1 + self._offset
            if low < 0 or high >= reader.n_frames:
                reader.close()
                raise ValueError(
                    f"{self._path}: scene frames [{self._first}.."
                    f"{self._first + self._count - 1}] + offset {self._offset} "
                    f"outside pack range (n_frames={reader.n_frames})"
                )
            self._reader = reader
        return self._reader

    def frame(self, scene_idx: int) -> np.ndarray | None:
        """Return one scene frame, or ``None`` when the scene has no frame."""

        if self._path is None:
            return None
        if not (self._first <= scene_idx < self._first + self._count):
            return None
        return self._open().read_frame(scene_idx + self._offset)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self) -> None:
        self.close()


class TrainingSceneHandles:
    """Open readers + small whole-scene arrays for one scene (worker-cached).

    Validates the scene against the training contract at open time — a schema
    error fails here, at the reader boundary, not as a shape error deep in a
    training step.
    """

    def __init__(self, scene_dir: Path, load_gt: bool, t4_root: Path):
        self._load_gt = bool(load_gt)
        d = Path(scene_dir) / "derived"
        try:
            self.meta = json.loads((d / "meta.json").read_text())
            n = int(self.meta["n_frames"])
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid T4 scene metadata in {d / 'meta.json'}") from exc
        if n <= 0:
            raise ValueError(f"T4 scene {scene_dir} has invalid n_frames={n}")
        with np.load(d / "scalars.npz", allow_pickle=False) as z:
            self.scalars = {k: z[k] for k in z.files}
        missing_scalars = sorted(set(TRAINING_SCALAR_FIELDS) - self.scalars.keys())
        if missing_scalars:
            raise ValueError(f"{scene_dir}: scalars.npz missing {missing_scalars}")
        for name in ("trajectory", "velocity", "turn"):
            if len(self.scalars[name]) != n:
                raise ValueError(
                    f"{scene_dir}: scalar {name!r} has "
                    f"{len(self.scalars[name])} frames, expected {n}"
                )
        if self.scalars["trajectory"].shape[1:] != (4,):
            raise ValueError(f"{scene_dir}: trajectory must have shape [n_frames, 4]")
        if self.scalars["velocity"].shape[1:] != (2,):
            raise ValueError(f"{scene_dir}: velocity must have shape [n_frames, 2]")
        if self.scalars["turn"].ndim != 1:
            raise ValueError(f"{scene_dir}: turn must have shape [n_frames]")
        if self.scalars["goal"].shape != (4,):
            raise ValueError(f"{scene_dir}: goal must have shape [4]")
        if self.scalars["shape"].shape != (3,):
            raise ValueError(f"{scene_dir}: shape must have shape [3]")

        self.bundle = T4BundleReader(d / "frames.pack")
        if self.bundle.n_frames != n:
            raise ValueError(
                f"{scene_dir}: frames.pack has {self.bundle.n_frames} frames, expected {n}"
            )
        missing_frames = sorted(set(TRAINING_FRAME_FIELDS) - self.bundle.field_spec.keys())
        if missing_frames:
            raise ValueError(f"{scene_dir}: frames.pack missing {missing_frames}")
        for field, shape in expected_map_shapes().items():
            spec = self.bundle.field_spec[field]
            if tuple(spec["shape"]) != shape or spec["variable"]:
                raise ValueError(
                    f"{scene_dir}: frames.pack field {field!r} has shape/spec "
                    f"{spec}, expected fixed trailing shape {shape}"
                )
        self.views = {f: _FrameView(self.bundle, f, n) for f in TRAINING_FRAME_FIELDS}
        self.gt: Dict[str, _FrameView] = {}
        if self._load_gt:
            present = {name for name in ("gt_boxes", "gt_labels") if name in self.bundle.field_spec}
            if present and present != {"gt_boxes", "gt_labels"}:
                raise ValueError(f"{scene_dir}: gt_boxes and gt_labels must be present together")
            if present:
                boxes_spec = self.bundle.field_spec["gt_boxes"]
                labels_spec = self.bundle.field_spec["gt_labels"]
                if tuple(boxes_spec["shape"]) != (9,) or tuple(labels_spec["shape"]) != ():
                    raise ValueError(
                        f"{scene_dir}: GT fields must have trailing shapes (9,) and (), "
                        f"got {boxes_spec['shape']} and {labels_spec['shape']}"
                    )
            for name in present:
                self.gt[name] = _FrameView(self.bundle, name, n)
        # meta.json stores lidar_pack relative to the T4 dataset root.  The
        # root is passed explicitly so DataLoader workers do not depend on a
        # process environment variable and train/val roots can coexist safely.
        self.lidar = _LazyLidarFrames(self.meta, Path(t4_root))
        self.n_frames = n

    def scene_dict(self, scene_key: str) -> dict:
        """The lazily-backed mapping :class:`TrainingWindowBuilder` consumes."""

        sc: dict = {
            "meta": self.meta,
            "_lazy_lidar": self.lidar,
            "_scene_key": scene_key,
        }
        sc.update(self.scalars)
        sc.update(self.views)
        if self.gt:
            sc["gt_boxes"] = _GTByFrame(self.gt["gt_boxes"])
            sc["gt_labels"] = _GTByFrame(self.gt["gt_labels"])
        elif self._load_gt:
            sc["gt_boxes"] = _EmptyByFrame()
            sc["gt_labels"] = _EmptyByFrame()
        return sc

    def close(self) -> None:
        self.bundle.close()
        self.lidar.close()


def _mapping_frame(mapping, key: str, default=None):
    if mapping is None:
        return default
    getter = getattr(mapping, "get", None)
    if getter is not None:
        return getter(key, default)
    try:
        return mapping[key]
    except (KeyError, IndexError):
        return default


class TrainingWindowBuilder:
    """Build flat planner samples from one lazily-backed T4 scene mapping.

    :param goal_clamp_m: optional radius cap on the goal pose, in metres.
    :param point_range: optional ``[x_min, y_min, z_min, x_max, y_max, z_max]``
        LiDAR crop.  This is the consuming model's voxel range, so it is a
        parameter here rather than a constant; ``None`` keeps the full sweep.
    :param load_points: read the centre-frame LiDAR sweep.  Off, ``points``
        is still emitted — empty — so the batch contract is unchanged and
        only the sweep read disappears (a camera-only backbone reads no
        LiDAR, and the sweep is by far the largest thing in the window).
    :param spec: the window's temporal contract — history/future span and
        model rate.  Sampling is stride-based on the 10 Hz source, so any
        extracted value is bit-identical to a source frame value.
    """

    def __init__(
        self,
        *,
        goal_clamp_m: float | None = None,
        point_range: Sequence[float] | None = None,
        load_points: bool = True,
        spec: TemporalSpec = DEFAULT_TEMPORAL_SPEC,
    ):
        self.goal_clamp_m = float(goal_clamp_m) if goal_clamp_m is not None else None
        self.point_range = tuple(float(v) for v in point_range) if point_range else None
        if self.point_range is not None and len(self.point_range) != 6:
            raise ValueError("point_range must be [x_min, y_min, z_min, x_max, y_max, z_max]")
        self.load_points = bool(load_points)
        self.spec = spec

    @staticmethod
    def _load_lidar_frame(scene: dict, frame_idx: int) -> np.ndarray:
        points = scene.get("points")
        key = f"frame_{frame_idx:04d}"
        if isinstance(points, dict) and key in points:
            return points[key]
        lazy = scene.get("_lazy_lidar")
        if lazy is not None:
            frame = lazy.frame(frame_idx)
            if frame is not None:
                return frame
        return np.zeros((0, 5), dtype=np.float32)

    def extract_window(self, scene: dict, center_idx: int) -> Optional[dict]:
        """Extract one spec-shaped window around ``center_idx``.

        Everything is returned in the centre frame's ego coordinates, sampled
        at the spec's model rate by striding the 10 Hz source.  Returns
        ``None`` when the window does not fit inside the scene.
        """

        spec = self.spec
        fs = spec.frame_stride
        n_frames = int(scene["meta"]["n_frames"])
        past_start = center_idx - spec.history_span
        future_end = center_idx + spec.future_span + 1
        if past_start < 0 or future_end > n_frames:
            return None

        trajectory = np.asarray(scene["trajectory"])
        velocity = np.asarray(scene["velocity"])
        center_x, center_y, center_cos, center_sin = trajectory[center_idx]
        heading = np.arctan2(center_sin, center_cos)
        cos_neg, sin_neg = np.cos(-heading), np.sin(-heading)

        past = trajectory[past_start : center_idx + 1 : fs]
        dx, dy = past[:, 0] - center_x, past[:, 1] - center_y
        relative_heading = np.arctan2(past[:, 3], past[:, 2]) - heading
        ego_past = np.stack(
            [
                dx * cos_neg - dy * sin_neg,
                dx * sin_neg + dy * cos_neg,
                np.cos(relative_heading),
                np.sin(relative_heading),
            ],
            axis=1,
        ).astype(np.float32)

        future = trajectory[center_idx + fs : future_end : fs]
        dx, dy = future[:, 0] - center_x, future[:, 1] - center_y
        ego_future = np.stack(
            [
                dx * cos_neg - dy * sin_neg,
                dx * sin_neg + dy * cos_neg,
                np.arctan2(future[:, 3], future[:, 2]) - heading,
            ],
            axis=1,
        ).astype(np.float32)

        vx, vy = velocity[center_idx]
        acceleration = scene.get("acceleration")
        ax, ay = acceleration[center_idx] if acceleration is not None else (0.0, 0.0)
        steering = scene.get("steering")
        yaw_rate = scene.get("yaw_rate")
        ego_current = np.array(
            [
                0.0,
                0.0,
                1.0,
                0.0,
                vx,
                vy,
                ax,
                ay,
                steering[center_idx] if steering is not None else 0.0,
                yaw_rate[center_idx] if yaw_rate is not None else 0.0,
            ],
            dtype=np.float32,
        )

        turn = np.asarray(scene["turn"])[past_start : center_idx + 1 : fs].astype(np.int32)
        goal = np.asarray(scene["goal"])
        goal_dx, goal_dy = goal[0] - center_x, goal[1] - center_y
        goal_x = goal_dx * cos_neg - goal_dy * sin_neg
        goal_y = goal_dx * sin_neg + goal_dy * cos_neg
        if self.goal_clamp_m is not None:
            distance = float(np.hypot(goal_x, goal_y))
            if distance > self.goal_clamp_m:
                scale = self.goal_clamp_m / distance
                goal_x, goal_y = goal_x * scale, goal_y * scale
        goal_heading = np.arctan2(goal[3], goal[2]) - heading
        goal_pose = np.array(
            [goal_x, goal_y, np.cos(goal_heading), np.sin(goal_heading)], dtype=np.float32
        )

        points = (
            self._load_lidar_frame(scene, center_idx)
            if self.load_points
            else np.zeros((0, 5), dtype=np.float32)
        )
        if points.shape[0]:
            if self.point_range is not None:
                xyz = points[:, :3]
                bounds = self.point_range
                mask = (
                    (xyz[:, 0] >= bounds[0])
                    & (xyz[:, 0] < bounds[3])
                    & (xyz[:, 1] >= bounds[1])
                    & (xyz[:, 1] < bounds[4])
                    & (xyz[:, 2] >= bounds[2])
                    & (xyz[:, 2] < bounds[5])
                )
                points = points[mask]
            else:
                points = points.copy()
            points[:, 4] = 0.0

        frame_key = f"frame_{center_idx:04d}"
        gt_boxes = _mapping_frame(scene.get("gt_boxes"), frame_key)
        gt_labels = _mapping_frame(scene.get("gt_labels"), frame_key)
        if gt_boxes is None:
            gt_boxes = np.zeros((0, 9), dtype=np.float32)
        if gt_labels is None:
            gt_labels = np.zeros((0,), dtype=np.int64)

        def frame_field(name: str):
            return np.asarray(scene[name][center_idx])

        return {
            "ego_agent_past": ego_past,
            "ego_current_state": ego_current,
            "ego_agent_future": ego_future,
            "lanes": frame_field("lanes"),
            "lanes_speed_limit": frame_field("lanes_speed"),
            "lanes_has_speed_limit": frame_field("lanes_has_speed"),
            "route_lanes": frame_field("route"),
            "route_lanes_speed_limit": frame_field("route_speed"),
            "route_lanes_has_speed_limit": frame_field("route_has_speed"),
            "polygons": frame_field("polygons"),
            "line_strings": frame_field("lines"),
            "ego_shape": np.asarray(scene["shape"], dtype=np.float32),
            "turn_indicators": turn,
            "goal_pose": goal_pose,
            "points": points,
            "gt_bboxes_3d": np.asarray(gt_boxes, dtype=np.float32),
            "gt_labels_3d": np.asarray(gt_labels, dtype=np.int64),
            "scene_key": str(scene.get("_scene_key", "")),
            "scene_center": int(center_idx),
        }
