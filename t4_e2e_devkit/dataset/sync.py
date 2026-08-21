"""Sensor synchronisation: putting boxes where the camera actually saw them.

``derived/`` exposes one frame index for every sensor, which reads as though the
LiDAR sweep and the camera frame were simultaneous. They are not: every camera
trails its LiDAR frame, and by how much depends on the channel **and on the
vehicle**.

Measured over 8 vehicles across ``prd_jt``, ``prd_jt_val`` and ``x2_dev``:

=========================  ===================  ==================
channel                    range over vehicles  spread
=========================  ===================  ==================
``CAM_FRONT_RIGHT_WIDE``     +49.3 .. +52.0 ms             2.7 ms
``CAM_FRONT_WIDE``           +42.4 .. +51.5 ms             9.1 ms
``CAM_BACK_RIGHT_WIDE``      +91.5 .. +92.8 ms             1.2 ms
``CAM_BACK_LEFT_WIDE``      +109.0 .. +110.3 ms            1.3 ms
``CAM_FRONT_LEFT_WIDE``     +115.3 .. +118.7 ms            3.4 ms
``CAM_BACK``                  -7.6 .. +110.2 ms          117.8 ms
``CAM_TOP_LEFT_CENTER``      +17.4 .. +118.9 ms          101.4 ms
=========================  ===================  ==================

Three things follow, and together they rule out every shortcut:

* **The offsets exceed a frame.** At 10 Hz a frame is 100 ms, so
  ``CAM_FRONT_LEFT_WIDE`` is more than one LiDAR frame behind. Pairing box frame
  ``i`` with image frame ``i`` is not a sub-frame refinement there, it is off by a
  whole frame.
* **They differ by up to 66 ms between channels of one rig**, so no single global
  shift is correct.
* **They differ between vehicles** -- 9 ms on ``CAM_FRONT_WIDE``, over 100 ms on
  ``CAM_BACK`` and the roof views. So a **hardcoded offset table is wrong**, even
  one keyed by channel. Every offset here is read from the scene being processed,
  never from a constant: :meth:`SensorSync.open` parses that scene's own
  ``sample_data.json``, so the correction is per scene and therefore per vehicle
  by construction. The numbers above are for understanding the magnitude, not for
  use.

At 6.3 m/s -- an ordinary urban speed in this data -- 116 ms is 0.73 m of ego
motion, which is a visible box displacement on the image.

The numbers come from ``annotation/sample_data.json`` and
``annotation/ego_pose.json``, which sit in every scene directory next to
``derived/``. Verified: ``sample_data`` carries 524 records per channel keyed by
the image filename, so file ``00042.jpg`` resolves to its own timestamp; each
record's ``ego_pose_token`` resolves to a pose whose timestamp equals the record's
exactly (max deviation 0 us), so the ego pose is already sampled per sensor and
needs no interpolation.

APPROACH
--------

Follows TIER IV's ``t4-devkit``, whose ``get_box3ds`` interpolates an annotation
to the image's own timestamp and uses the ego pose at that timestamp. The
transform machinery here is theirs, vendored -- see
:mod:`t4_e2e_devkit.common.tier4.transform` -- so the frame conventions match
their reference by construction rather than by re-derivation.

One difference, forced by the data: ``prd_jt``'s ``sample_annotation.json`` is
empty, because its labels are online-tracker output stored in
``derived/frames.pack`` per LiDAR frame rather than as annotation records. So
instead of interpolating between sparse keyframe annotations, this interpolates
between the **dense 10 Hz LiDAR frames** that bracket the camera timestamp, which
is strictly more information than t4-devkit has to work with.

SCOPE
-----

JPEG-backed cameras only, for now. The video-backed channels have degenerate
leading timestamps -- the first ~54 records of ``CAM_FRONT`` share one value and
are flagged ``is_key_frame: false`` -- so a correction there needs a validity
policy that does not exist yet. :func:`camera_offsets` reports what it can for any
channel; :meth:`SensorSync.corrected_annotations` refuses a channel whose
timestamps do not pass :func:`validate_channel`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.constants import T4_LIDAR_PACK_NAME
from t4_e2e_devkit.common.enums import T4BoxIndex

#: Where the per-sensor tables live, relative to a scene directory.
SAMPLE_DATA = "annotation/sample_data.json"
EGO_POSE = "annotation/ego_pose.json"
SENSOR = "annotation/sensor.json"
CALIBRATED_SENSOR = "annotation/calibrated_sensor.json"

#: A correction larger than this is refused rather than applied. Two frames of
#: skew is far beyond the measured 50-116 ms and means the channel is misaligned
#: in a way extrapolation would paper over.
MAX_CORRECTION_S = 0.25


class SyncUnavailable(RuntimeError):
    """A scene or channel cannot be synchronised, with the reason."""


@dataclass
class ChannelTimes:
    """One sensor channel's per-frame timestamps and ego poses."""

    channel: str
    frames: Dict[int, int]  # frame index -> timestamp, microseconds
    poses: Dict[int, np.ndarray]  # frame index -> [x, y, z] in map coordinates
    rotations: Dict[int, np.ndarray]  # frame index -> quaternion [w, x, y, z]
    key_frame: Dict[int, bool]

    def timestamp_s(self, frame_index: int) -> Optional[float]:
        """
        :param frame_index: frame index.
        :return: the capture time in seconds, or ``None`` when absent.
        """
        raw = self.frames.get(int(frame_index))
        return None if raw is None else raw * 1e-6

    @property
    def sorted_frames(self) -> List[int]:
        """:return: frame indices in order."""
        return sorted(self.frames)


@dataclass
class SensorSync:
    """Per-sensor timing for one scene, and the corrections it enables."""

    scene_dir: Path
    channels: Dict[str, ChannelTimes] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def open(cls, scene_dir: str | Path, require: bool = False) -> Optional[SensorSync]:
        """Read a scene's sensor tables.

        :param scene_dir: the T4 scene directory.
        :param require: raise instead of returning ``None`` when absent.
        :return: the sync, or ``None`` when the scene has no annotation tables.
        :raises SyncUnavailable: when ``require`` and the tables are missing or
            unreadable.
        """
        scene_dir = Path(scene_dir)
        paths = {
            name: scene_dir / name for name in (SAMPLE_DATA, EGO_POSE, SENSOR, CALIBRATED_SENSOR)
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            if require:
                raise SyncUnavailable(
                    f"{scene_dir}: cannot synchronise, missing {missing}. The per-sensor "
                    "timestamps live in the scene's annotation/ tables, not in derived/."
                )
            return None

        try:
            sample_data = json.loads(paths[SAMPLE_DATA].read_text())
            ego_poses = {
                record["token"]: record for record in json.loads(paths[EGO_POSE].read_text())
            }
            sensors = {record["token"]: record for record in json.loads(paths[SENSOR].read_text())}
            calibrated = {
                record["token"]: record
                for record in json.loads(paths[CALIBRATED_SENSOR].read_text())
            }
        except (OSError, ValueError, KeyError) as error:
            if require:
                raise SyncUnavailable(
                    f"{scene_dir}: unreadable annotation tables: {error!r}"
                ) from error
            return None

        channels: Dict[str, ChannelTimes] = {}
        for record in sample_data:
            calibration = calibrated.get(record.get("calibrated_sensor_token"))
            sensor = sensors.get(calibration.get("sensor_token")) if calibration else None
            if sensor is None:
                continue
            channel = str(sensor.get("channel"))
            # The filename is what pins a record to a frame index; the record
            # order does not, and for the video-backed channels it disagrees.
            filename = str(record.get("filename", ""))
            stem = filename.rsplit("/", 1)[-1].split(".")[0]
            if not stem.isdigit():
                continue
            frame_index = int(stem)

            entry = channels.setdefault(channel, ChannelTimes(channel, {}, {}, {}, {}))
            entry.frames[frame_index] = int(record["timestamp"])
            entry.key_frame[frame_index] = bool(record.get("is_key_frame", True))
            pose = ego_poses.get(record.get("ego_pose_token"))
            if pose is not None:
                entry.poses[frame_index] = np.asarray(pose["translation"], dtype=np.float64)
                entry.rotations[frame_index] = np.asarray(pose["rotation"], dtype=np.float64)

        return cls(scene_dir=scene_dir, channels=channels)

    # ------------------------------------------------------------------ #
    # Timing
    # ------------------------------------------------------------------ #

    @property
    def lidar(self) -> Optional[ChannelTimes]:
        """:return: the LiDAR channel's timing, if present."""
        return self.channels.get(T4_LIDAR_PACK_NAME)

    def offset_s(self, channel: str, frame_index: int) -> Optional[float]:
        """Seconds by which a camera frame trails its LiDAR frame.

        :param channel: camera channel name.
        :param frame_index: frame index.
        :return: the offset, positive when the camera is later; ``None`` when
            either timestamp is missing.
        """
        camera = self.channels.get(channel)
        lidar = self.lidar
        if camera is None or lidar is None:
            return None
        camera_t = camera.timestamp_s(frame_index)
        lidar_t = lidar.timestamp_s(frame_index)
        if camera_t is None or lidar_t is None:
            return None
        return camera_t - lidar_t

    def validate_channel(self, channel: str, max_frames: int = 64) -> Tuple[bool, str]:
        """Whether a channel's timestamps support a correction.

        Rejects the degenerate case found in the video-backed channels, where the
        leading records share one timestamp and are flagged non-key-frame: a
        correction computed from a repeated timestamp is not a correction.

        :param channel: camera channel name.
        :param max_frames: how many frames to inspect.
        :return: ``(usable, reason)``.
        """
        camera = self.channels.get(channel)
        if camera is None:
            return False, f"channel {channel!r} is absent from sample_data.json"
        if self.lidar is None:
            return False, "scene has no LiDAR channel to synchronise against"

        frames = camera.sorted_frames[:max_frames]
        if len(frames) < 2:
            return False, f"channel {channel!r} has fewer than two timestamped frames"

        timestamps = [camera.frames[index] for index in frames]
        duplicates = len(timestamps) - len(set(timestamps))
        if duplicates:
            return False, (
                f"channel {channel!r} repeats {duplicates} of its first {len(frames)} "
                "timestamps, so those frames carry no distinct capture time "
                "(seen on the video-backed channels)"
            )
        if not all(camera.key_frame.get(index, True) for index in frames):
            return False, f"channel {channel!r} has non-key-frame records in its leading frames"

        offsets = [self.offset_s(channel, index) for index in frames]
        offsets = [value for value in offsets if value is not None]
        if not offsets:
            return False, f"channel {channel!r} shares no frame indices with the LiDAR"
        if max(abs(value) for value in offsets) > MAX_CORRECTION_S:
            return False, (
                f"channel {channel!r} is offset by up to "
                f"{max(abs(v) for v in offsets) * 1e3:.0f} ms, beyond the "
                f"{MAX_CORRECTION_S * 1e3:.0f} ms this correction will apply"
            )
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Ego motion
    # ------------------------------------------------------------------ #

    def ego_transform(self, channel: str, frame_index: int) -> Optional["object"]:
        """The transform from the LiDAR-frame ego pose to the camera-frame one.

        Built with TIER IV's own :class:`HomogeneousMatrix`, so the frame
        conventions match their reference rather than a local re-derivation.

        :param channel: camera channel name.
        :param frame_index: frame index.
        :return: a ``HomogeneousMatrix`` mapping LiDAR-time ego coordinates into
            camera-time ego coordinates, or ``None`` when either pose is missing.
        """
        from t4_e2e_devkit.common.tier4.transform import HomogeneousMatrix
        from t4_e2e_devkit.common.tier4.typing import Quaternion

        camera, lidar = self.channels.get(channel), self.lidar
        if camera is None or lidar is None:
            return None
        if frame_index not in camera.poses or frame_index not in lidar.poses:
            return None

        # Both poses are ego -> map at their own capture time, and both are
        # already sampled at that time (checked: ego_pose.timestamp equals
        # sample_data.timestamp exactly), so no interpolation is needed.
        lidar_to_map = HomogeneousMatrix(
            lidar.poses[frame_index],
            Quaternion(lidar.rotations[frame_index]),
            src="ego_lidar",
            dst="map",
        )
        camera_to_map = HomogeneousMatrix(
            camera.poses[frame_index],
            Quaternion(camera.rotations[frame_index]),
            src="ego_camera",
            dst="map",
        )
        # The inverse is built with explicit frame labels rather than via
        # ``.inv()``: that method inverts the matrix but leaves ``src``/``dst``
        # unchanged, so relying on it to express "map -> ego_camera" would compose
        # a numerically correct transform under a mislabelled pair -- and
        # ``dot`` validates the labels, not the matrices.
        map_to_camera = HomogeneousMatrix.from_matrix(
            np.linalg.inv(camera_to_map.matrix), src="map", dst="ego_camera"
        )
        # ego_lidar -> map -> ego_camera
        return map_to_camera.dot(lidar_to_map)

    # ------------------------------------------------------------------ #
    # Corrected boxes
    # ------------------------------------------------------------------ #

    def corrected_boxes(
        self,
        boxes: npt.NDArray[np.floating],
        channel: str,
        frame_index: int,
        next_boxes: Optional[npt.NDArray[np.floating]] = None,
        next_labels: Optional[npt.NDArray[np.integer]] = None,
        labels: Optional[npt.NDArray[np.integer]] = None,
    ) -> npt.NDArray[np.float32]:
        """Move boxes from their LiDAR frame to a camera's capture time and frame.

        Two corrections, in this order:

        1. **Object motion.** Each box is advanced by the offset. When the next
           LiDAR frame is supplied, associated boxes are *interpolated* between
           the two frames, which is what ``t4-devkit`` does and is more faithful
           than extrapolating; unassociated boxes fall back to constant velocity
           from their own ``vx, vy``.
        2. **Ego motion.** The result is transformed from the ego frame at the
           LiDAR timestamp into the ego frame at the camera timestamp, since the
           camera extrinsic is relative to the latter.

        :param boxes: ``[M, 9]`` boxes at ``frame_index``, following
            :class:`~t4_e2e_devkit.common.enums.T4BoxIndex`.
        :param channel: camera channel name.
        :param frame_index: frame index the boxes belong to.
        :param next_boxes: ``[M2, 9]`` boxes at the bracketing later LiDAR frame.
        :param next_labels: labels for ``next_boxes``, for association.
        :param labels: labels for ``boxes``, for association.
        :return: ``[M, 9]`` corrected boxes.
        :raises SyncUnavailable: when the channel cannot be corrected, rather than
            returning the input unchanged -- silently skipping the correction is
            how a caller ends up believing it was applied.
        """
        usable, reason = self.validate_channel(channel)
        if not usable:
            raise SyncUnavailable(f"{self.scene_dir}: {reason}")

        offset = self.offset_s(channel, frame_index)
        if offset is None:
            raise SyncUnavailable(
                f"{self.scene_dir}: no timestamp for {channel!r} at frame {frame_index}"
            )

        values = np.asarray(boxes, dtype=np.float64).reshape(-1, 9).copy()
        if values.shape[0] == 0:
            return values.astype(np.float32)

        moved = self._advance_objects(values, offset, frame_index, next_boxes, labels, next_labels)
        return self._apply_ego_motion(moved, channel, frame_index).astype(np.float32)

    def _advance_objects(
        self,
        values: npt.NDArray[np.float64],
        offset: float,
        frame_index: int,
        next_boxes: Optional[npt.NDArray[np.floating]],
        labels: Optional[npt.NDArray[np.integer]],
        next_labels: Optional[npt.NDArray[np.integer]],
    ) -> npt.NDArray[np.float64]:
        """Advance each object to the camera's capture time."""
        lidar = self.lidar
        interval = None
        if lidar is not None:
            here, later = lidar.timestamp_s(frame_index), lidar.timestamp_s(frame_index + 1)
            if here is not None and later is not None and later > here:
                interval = later - here

        if next_boxes is not None and interval is not None and offset <= interval:
            matched = _associate(
                values, np.asarray(next_boxes, np.float64).reshape(-1, 9), labels, next_labels
            )
            alpha = offset / interval
            for index, target in matched.items():
                # Interpolate position and heading between the bracketing frames;
                # this is t4-devkit's approach, and strictly better informed than
                # extrapolation because the later frame is actually observed.
                values[index, T4BoxIndex.POINT2D] = (1.0 - alpha) * values[
                    index, T4BoxIndex.POINT2D
                ] + alpha * target[T4BoxIndex.POINT2D]
                values[index, T4BoxIndex.HEADING] = _interpolate_heading(
                    values[index, T4BoxIndex.HEADING], target[T4BoxIndex.HEADING], alpha
                )
            unmatched = [i for i in range(values.shape[0]) if i not in matched]
        else:
            unmatched = list(range(values.shape[0]))

        # Constant velocity for anything the association could not bracket.
        if unmatched:
            rows = np.asarray(unmatched, dtype=np.int64)
            values[rows, T4BoxIndex.X] += offset * values[rows, T4BoxIndex.VELOCITY_X]
            values[rows, T4BoxIndex.Y] += offset * values[rows, T4BoxIndex.VELOCITY_Y]
        return values

    def _apply_ego_motion(
        self, values: npt.NDArray[np.float64], channel: str, frame_index: int
    ) -> npt.NDArray[np.float64]:
        """Re-express boxes in the ego frame at the camera's capture time."""
        transform = self.ego_transform(channel, frame_index)
        if transform is None:
            # No pose pair: the object-motion correction still applies, and
            # saying so beats pretending the ego stood still.
            return values

        positions = np.column_stack(
            [values[:, T4BoxIndex.X], values[:, T4BoxIndex.Y], values[:, T4BoxIndex.Z]]
        )
        moved = np.stack([transform.transform(position) for position in positions])
        values[:, T4BoxIndex.X] = moved[:, 0]
        values[:, T4BoxIndex.Y] = moved[:, 1]
        values[:, T4BoxIndex.Z] = moved[:, 2]
        # The ego rotates as well as translates, so headings turn with it.
        yaw = float(transform.yaw_pitch_roll[0])
        values[:, T4BoxIndex.HEADING] += yaw
        # Velocities are directions in the ego frame and rotate the same way.
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        vx = values[:, T4BoxIndex.VELOCITY_X].copy()
        vy = values[:, T4BoxIndex.VELOCITY_Y].copy()
        values[:, T4BoxIndex.VELOCITY_X] = cos_y * vx - sin_y * vy
        values[:, T4BoxIndex.VELOCITY_Y] = sin_y * vx + cos_y * vy
        return values


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _interpolate_heading(start: float, end: float, alpha: float) -> float:
    """Interpolate a heading the short way round the circle."""
    delta = (end - start + np.pi) % (2 * np.pi) - np.pi
    return float(start + alpha * delta)


def _associate(
    boxes: npt.NDArray[np.float64],
    next_boxes: npt.NDArray[np.float64],
    labels: Optional[npt.NDArray[np.integer]],
    next_labels: Optional[npt.NDArray[np.integer]],
    max_distance: float = 3.0,
) -> Dict[int, npt.NDArray[np.float64]]:
    """Match boxes between consecutive LiDAR frames by nearest centre.

    Class-aware and distance-gated: a match across classes, or further than a
    plausible one-frame displacement, is worse than no match, because it would
    interpolate one object toward another one's position.

    :param boxes: ``[M, 9]`` at the earlier frame.
    :param next_boxes: ``[M2, 9]`` at the later frame.
    :param labels: labels of ``boxes``, or ``None`` to skip the class gate.
    :param next_labels: labels of ``next_boxes``.
    :param max_distance: metres; beyond this a pair is not the same object.
    :return: row index in ``boxes`` -> matched row of ``next_boxes``.
    """
    if boxes.shape[0] == 0 or next_boxes.shape[0] == 0:
        return {}

    from scipy.optimize import linear_sum_assignment

    here = boxes[:, T4BoxIndex.POINT2D]
    later = next_boxes[:, T4BoxIndex.POINT2D]
    cost = np.linalg.norm(here[:, None, :] - later[None, :, :], axis=-1)

    if labels is not None and next_labels is not None:
        same = np.asarray(labels).reshape(-1, 1) == np.asarray(next_labels).reshape(1, -1)
        cost = np.where(same, cost, np.inf)

    cost = np.where(cost > max_distance, 1e6, cost)
    rows, columns = linear_sum_assignment(cost)
    return {
        int(row): next_boxes[int(column)]
        for row, column in zip(rows, columns, strict=True)
        if cost[row, column] < max_distance
    }


def data_list_timing_report(
    data_list: "Any",
    max_scenes_per_vehicle: int = 4,
    tolerance_ms: float = 5.0,
) -> Dict[str, Any]:
    """Report a data list's camera timing, grouped by vehicle.

    Grouped by vehicle because that is the unit the offset is a property of.
    Spread *across* vehicles is expected -- ``CAM_FRONT_WIDE`` ranges over 9 ms
    across the eight vehicles in these subtrees, ``CAM_BACK`` over 100 ms -- and
    flagging it would be flagging the data being what it is.

    Spread *within* one vehicle is different: the same rig should keep the same
    timing, so a channel that wanders inside one vehicle points at a converter or
    recording defect, and that is what this flags.

    None of this feeds the correction, which reads every offset from the scene it
    is correcting. The report exists so a run can know what timing spread its
    training set contains -- relevant because a model sees the offset as a
    per-vehicle nuisance variable rather than one fixed lead.

    :param data_list: a :class:`~t4_e2e_devkit.dataset.datalist.DataList`.
    :param max_scenes_per_vehicle: scenes to sample per vehicle.
    :param tolerance_ms: within-vehicle spread above which a channel is flagged.
    :return: ``{vehicles, channels, suspect, scenes_checked, tolerance_ms}``.
    """
    from collections import defaultdict

    by_vehicle: Dict[str, List[str]] = defaultdict(list)
    for scene in data_list.scene_dirs:
        parts = Path(scene).parts
        # <subtree>/<vehicle>/<date>/<time>; fall back to the whole path when a
        # list uses some other layout, so this degrades rather than mis-groups.
        vehicle = "/".join(parts[:2]) if len(parts) >= 2 else scene
        by_vehicle[vehicle].append(scene)

    per_vehicle: Dict[str, Dict[str, float]] = {}
    within: Dict[str, List[float]] = defaultdict(list)
    checked = 0
    for vehicle, scenes in sorted(by_vehicle.items()):
        gathered: Dict[str, List[float]] = defaultdict(list)
        for scene in scenes[:max_scenes_per_vehicle]:
            report = camera_offsets(data_list.absolute_scene_dir(scene), max_frames=32)
            if not report:
                continue
            checked += 1
            for channel, values in report.items():
                if values["usable"] and np.isfinite(values["mean_ms"]):
                    gathered[channel].append(values["mean_ms"])
        per_vehicle[vehicle] = {
            channel: float(np.mean(values)) for channel, values in sorted(gathered.items())
        }
        for channel, values in gathered.items():
            if len(values) > 1:
                within[channel].append(float(np.ptp(values)))

    channels: Dict[str, Any] = {}
    suspect: List[str] = []
    all_channels = sorted({c for table in per_vehicle.values() for c in table})
    for channel in all_channels:
        means = [table[channel] for table in per_vehicle.values() if channel in table]
        worst_within = max(within.get(channel, [0.0]), default=0.0)
        channels[channel] = {
            "min_ms": float(min(means)),
            "max_ms": float(max(means)),
            "across_vehicle_spread_ms": float(max(means) - min(means)),
            "worst_within_vehicle_spread_ms": worst_within,
            "vehicles": len(means),
        }
        if worst_within > tolerance_ms:
            suspect.append(channel)

    return {
        "scenes_checked": checked,
        "vehicles": per_vehicle,
        "channels": channels,
        "suspect": suspect,
        "tolerance_ms": tolerance_ms,
    }


def camera_offsets(
    scene_dir: str | Path,
    max_frames: int = 128,
) -> Dict[str, Dict[str, Any]]:
    """Report each channel's timing offset and whether it is correctable.

    :param scene_dir: the T4 scene directory.
    :param max_frames: frames to sample.
    :return: channel -> ``{mean_ms, std_ms, min_ms, max_ms, usable, reason}``.
    """
    sync = SensorSync.open(scene_dir)
    if sync is None or sync.lidar is None:
        return {}
    report: Dict[str, Dict[str, Any]] = {}
    for channel in sorted(sync.channels):
        if channel == T4_LIDAR_PACK_NAME:
            continue
        frames = sync.channels[channel].sorted_frames[:max_frames]
        offsets = [sync.offset_s(channel, index) for index in frames]
        offsets = np.array([value * 1e3 for value in offsets if value is not None])
        usable, reason = sync.validate_channel(channel)
        report[channel] = {
            "mean_ms": float(offsets.mean()) if offsets.size else float("nan"),
            "std_ms": float(offsets.std()) if offsets.size else float("nan"),
            "min_ms": float(offsets.min()) if offsets.size else float("nan"),
            "max_ms": float(offsets.max()) if offsets.size else float("nan"),
            "usable": usable,
            "reason": reason,
        }
    return report
