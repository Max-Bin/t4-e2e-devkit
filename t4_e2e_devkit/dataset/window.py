"""Assembling one T4 window into the devkit's unified :class:`T4Scene`.

This is where a LiDAR model and a camera model stop being separate pipelines.
Both address a window as ``(scene_dir, center_frame)``; both get the same
:class:`~t4_e2e_devkit.common.dataclasses.T4Scene`; what differs is only which
sensor streams their :class:`~t4_e2e_devkit.common.dataclasses.SensorConfig`
asked to decode.  Decoding is the dominant cost of the input pipeline -- five
2880x1860 JPEGs or a full sweep per frame -- so "ask for nothing, pay nothing"
is what makes one reader serve both without either subsidising the other.

Everything numeric is delegated to
:mod:`t4_e2e_devkit.dataset.scene`, the validated reader: pose-to-local
conversion, the ego-status derivation, the stationary-box bridge, and the
frame-to-centre box transform.  This module chooses *which* of those to call
and packages the result; it re-derives none of them, because the reference
judge is calibrated against exactly those functions.

Camera images come out as raw ``uint8`` ``[H, W, 3]`` at reader resolution.
Normalization lives in the feature builders, one layer up.  That split is not
cosmetic: the reader's own float path is ``asarray(image, float32) / 255``
applied to the same resized PIL image, so a builder doing ``/255 - mean / std``
on these bytes reproduces it exactly, while keeping a sample five times smaller
across the DataLoader worker boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from t4_e2e_devkit.common.constants import (
    SCORER_FUTURE_FRAMES,
    T4_DEFAULT_IMAGE_SIZE_HW,
    T4_FRAME_RATE_HZ,
)
from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    Camera,
    Cameras,
    EgoShape,
    EgoStatus,
    Lidar,
    MapObjectIds,
    MapTensors,
    SceneFilter,
    SceneMetadata,
    SensorConfig,
    T4Frame,
    T4Scene,
)
from t4_e2e_devkit.common.t4_map import T4MapAPI
from t4_e2e_devkit.dataset.camera_source import open_camera_sources
from t4_e2e_devkit.dataset.contract import BUNDLE_TO_CONTRACT
from t4_e2e_devkit.dataset.rigs import readable_camera_names, resolve_camera_names
from t4_e2e_devkit.dataset.route import T4RouteMetadata, load_t4_route
from t4_e2e_devkit.dataset.scene import (
    T4SceneReader,
    _bridge_stationary_boxes,
    _transform_agent_boxes_to_center,
    build_ego_status,
    global_to_ego,
)
from t4_e2e_devkit.dataset.scene_tags import T4SceneTag, T4SceneTagIndex

T4_FRAME_DT_S = 1.0 / T4_FRAME_RATE_HZ


def _as_bool(value: Any) -> bool:
    """Interpret config booleans without treating ``"false"`` as true."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class WindowError(ValueError):
    """A window cannot be assembled as requested.

    Distinguished from a generic ``ValueError`` so a data-list builder can drop
    the row while a training run treats the same condition as fatal -- which is
    the correct asymmetry: a builder is deciding whether a window is usable, a
    training step has already been told that it is.
    """


class T4WindowBuilder:
    """Builds :class:`T4Scene` objects from one scene directory.

    One builder wraps one :class:`T4SceneReader`, so the reader's open file
    handles, decoded-frame cache and per-scene calibration are amortised across
    every window of that scene.  Hold it for the scene, not for the window.
    """

    def __init__(
        self,
        scene_dir: str | Path,
        root: str | Path,
        sensor_config: Optional[SensorConfig] = None,
        scene_filter: Optional[SceneFilter] = None,
        reader_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param scene_dir: the T4 scene directory.
        :param root: the dataset root the scene is addressed relative to.
        :param sensor_config: which sensors to decode; nothing by default.
        :param scene_filter: the window shape; the contract default otherwise.
        :param reader_config: extra settings forwarded to :class:`T4SceneReader`
            (frame cache directory, image size, missing-camera policy, ...).
        """
        self.scene_dir = Path(scene_dir)
        self.root = Path(root)
        self.sensor_config = sensor_config or SensorConfig.build_no_sensors()
        self.scene_filter = scene_filter or SceneFilter()

        config: Dict[str, Any] = dict(reader_config or {})
        # A camera-aware reader needs an explicit fixed register because its
        # order is part of the learned contract. A no-sensor reader has no
        # camera contract at all, however: requiring even a calibrated camera
        # register would make map/LiDAR-only scoring fail on scenes that carry
        # no JPEG camera files. Keep the register genuinely empty in that
        # path; ``read_cameras`` already returns ``None`` without decoding.
        #
        # The register is resolved against THIS scene's rig rather than taken
        # from a global default.  The fleet has at least three registers -- the
        # five-wide prd_jt rig, a prd_jt variant with no wide rear views, and
        # x2_dev with a centred CAM_BACK and only one wide view -- so a fixed
        # default fails outright on two of them.  See
        # :mod:`t4_e2e_devkit.dataset.rigs`.
        if not self.sensor_config.any_camera:
            config["camera_names"] = []
        else:
            requested = config.get("camera_names")
            if requested is None:
                requested = list(self.sensor_config.cameras.keys())
            # Resolve against cameras that are BOTH calibrated and stored, not
            # against the calibration register alone: 208 prd_jt scenes
            # calibrate the five wide views while exporting only two of them.
            config["camera_names"] = resolve_camera_names(
                requested,
                readable_camera_names(self.scene_dir),
                scene_dir=self.scene_dir,
            )
        config.setdefault("t4_image_size_hw", list(T4_DEFAULT_IMAGE_SIZE_HW))
        self.reader_config = config
        self._include_history_annotations = _as_bool(
            config.get("t4_include_history_annotations", False)
        )
        self.route_metadata: Optional[T4RouteMetadata] = load_t4_route(
            self.scene_dir,
            strict=_as_bool(config.get("t4_route_required", False)),
        )
        tags_root = config.get("t4_scene_tags_root")
        self.scene_tag_index: Optional[T4SceneTagIndex] = None
        self.scene_tags: tuple[T4SceneTag, ...] = ()
        if tags_root not in (None, "", "null", "None"):
            self.scene_tag_index = T4SceneTagIndex.cached(
                tags_root,
                include_debug=_as_bool(config.get("t4_scene_tags_include_debug", False)),
                strict=_as_bool(config.get("t4_scene_tags_strict", True)),
            )
            self.scene_tags = self.scene_tag_index.tags_for_scene(self.scene_dir)
        self.map_api: Optional[T4MapAPI] = None
        if _as_bool(config.get("t4_attach_map_ids", False)):
            route_lane_ids = (
                self.route_metadata.route_lane_ids
                if self.route_metadata is not None
                else ()
            )
            self.map_api = T4MapAPI.from_scene(
                self.scene_dir,
                config.get("t4_maps_root"),
                strict=_as_bool(config.get("t4_map_required", False)),
                route_lane_ids=route_lane_ids,
            )
        self.reader = T4SceneReader(self.scene_dir, self.root, config)
        self._image_hw = tuple(int(value) for value in config["t4_image_size_hw"])
        # Camera decoders, opened as a register on first use; see camera_source.
        self._camera_sources: Dict[str, Any] = {}
        # Per-sensor timing, for moving boxes to a camera's own capture time.
        # Optional: a scene without annotation/ tables simply gets no correction,
        # and says so through Camera.annotations being None.
        self._sync_loaded = False
        self._sync: Any = None

    # ------------------------------------------------------------------ #
    # Window enumeration
    # ------------------------------------------------------------------ #

    @property
    def num_frames(self) -> int:
        """:return: number of frames in this scene."""
        return int(self.reader.n_frames)

    def valid_centers(self) -> range:
        """Centre frames whose full window fits inside this scene.

        A centre needs ``num_history_frames - 1`` frames behind it and
        ``num_future_frames`` ahead; anything else would need padding, and a
        padded history is indistinguishable to the model from a vehicle that
        was standing still.

        :return: the valid centre range, empty when the scene is too short.
        """
        first = self.scene_filter.num_history_frames - 1
        last = self.num_frames - 1 - self.scene_filter.num_future_frames
        if last < first:
            return range(0)
        return range(first, last + 1, self.scene_filter.frame_interval)

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #

    def build(self, center: int) -> T4Scene:
        """
        Assemble the window centred on ``center``.
        :param center: centre frame index within the scene.
        :return: the assembled scene.
        :raises WindowError: when the window does not fit inside the scene.
        """
        history = self.scene_filter.num_history_frames
        future = self.scene_filter.num_future_frames
        first = center - history + 1
        last = center + future

        if first < 0 or last >= self.num_frames:
            raise WindowError(
                f"{self.scene_dir}: window [{first}, {last}] for centre {center} "
                f"falls outside the scene's {self.num_frames} frames"
            )

        trajectory = self.reader.trajectory
        center_pose = trajectory[center]
        history_indices = list(range(first, center + 1))

        # One shared definition of the ego-status columns, used by training and
        # by the closed loop alike.
        ego_status_array = build_ego_status(trajectory[history_indices], center_pose, T4_FRAME_DT_S)
        ego_shape = EgoShape.from_array(self.reader.scalars["shape"])
        turn = np.asarray(self.reader.scalars.get("turn", np.zeros(self.num_frames)))

        frames: List[T4Frame] = []
        for step, frame_index in enumerate(history_indices):
            # Negative iteration indices count back from the current frame, so
            # ``-1`` means "now" regardless of how long the history is.
            iteration = step - len(history_indices)
            frames.append(
                T4Frame(
                    frame_index=frame_index,
                    timestamp_us=self._timestamp_us(frame_index),
                    ego_status=EgoStatus(
                        ego_pose=ego_status_array[step, 0:3],
                        ego_velocity=ego_status_array[step, 3:5],
                        ego_acceleration=ego_status_array[step, 5:7],
                        ego_shape=ego_shape,
                        turn_indicator=int(turn[frame_index]) if turn.size else None,
                        control_state=self.read_control_state(frame_index),
                    ),
                    map_tensors=self.read_map(frame_index) if frame_index == center else None,
                    annotations=(
                        self.read_annotations(frame_index, center)
                        if frame_index == center or self._include_history_annotations
                        else None
                    ),
                    cameras=self.read_cameras(frame_index, iteration),
                    lidar=self.read_lidar(frame_index, iteration),
                )
            )

        future_poses, future_annotations = None, None
        if future > 0:
            future_indices = list(range(center + 1, last + 1))
            future_poses = global_to_ego(trajectory[future_indices], center_pose).astype(np.float32)
            future_annotations = self.read_future_annotations(center, future)

        window_timestamps = np.asarray(
            [self._timestamp_us(frame_index) for frame_index in range(first, last + 1)],
            dtype=np.int64,
        )

        return T4Scene(
            scene_metadata=SceneMetadata(
                scene_dir=self._relative_scene_dir(),
                scene_id=str(self.reader.meta.get("scene_id", self.scene_dir.name)),
                center_frame=int(center),
                num_history_frames=history,
                num_future_frames=future,
                vehicle=self.reader.meta.get("vehicle"),
                date=self.reader.meta.get("date"),
                timestamps_us=window_timestamps,
                global_center_pose=np.asarray(center_pose, dtype=np.float64),
                scene_tags=self.scene_tags,
                route_metadata=self.route_metadata,
            ),
            frames=frames,
            future_ego_poses=future_poses,
            future_annotations=future_annotations,
            goal_pose=self.read_goal(center_pose),
        )

    # ------------------------------------------------------------------ #
    # Field readers
    # ------------------------------------------------------------------ #

    def read_map(self, frame_index: int) -> MapTensors:
        """
        Read the vector map at one frame and rename it into contract naming.
        :param frame_index: frame to read.
        :return: the map tensors.
        :raises WindowError: when a map field is absent.  This is deliberately
            fatal rather than zero-filled: a camera-only run that quietly lost
            the route would still train, and produce a model that cannot follow
            one.
        """
        raw = self.reader.frame(int(frame_index))
        fields: Dict[str, np.ndarray] = {}
        for bundle_name, contract_name in BUNDLE_TO_CONTRACT.items():
            if bundle_name not in raw:
                raise WindowError(
                    f"{self.scene_dir}: frame {frame_index} has no map field "
                    f"{bundle_name!r} (contract name {contract_name!r}); present: "
                    f"{sorted(raw)}"
                )
            fields[contract_name] = np.asarray(raw[bundle_name])
        object_ids: Optional[MapObjectIds] = None
        if self.map_api is not None:
            center_pose = np.asarray(self.reader.trajectory[int(frame_index)], dtype=np.float64)
            route_ids = (
                self.route_metadata.route_lane_ids
                if self.route_metadata is not None
                else ()
            )
            lane_matches = self.map_api.match_local_centerlines_detailed(
                fields["lanes"],
                center_pose,
                layer="lanes",
                frame_index=frame_index,
            )
            route_matches = self.map_api.match_local_centerlines_detailed(
                fields["route_lanes"],
                center_pose,
                layer="route_lanes",
                frame_index=frame_index,
                allowed_ids=route_ids or None,
            )
            polygon_matches = self.map_api.match_local_geometries_detailed(
                fields["polygons"],
                center_pose,
                layer="polygons",
                frame_index=frame_index,
            )
            line_string_matches = self.map_api.match_local_geometries_detailed(
                fields["line_strings"],
                center_pose,
                layer="line_strings",
                frame_index=frame_index,
            )
            object_ids = MapObjectIds(
                lane_ids=tuple(match.source_object_id for match in lane_matches),
                route_lane_ids=tuple(
                    match.source_object_id for match in route_matches
                ),
                polygon_ids=tuple(match.source_object_id for match in polygon_matches),
                line_string_ids=tuple(
                    match.source_object_id for match in line_string_matches
                ),
                source_path=self.map_api.source_label,
                frame_index=int(frame_index),
                matches=lane_matches + route_matches + polygon_matches + line_string_matches,
            )
        return MapTensors(**fields, object_ids=object_ids)

    def read_annotations(self, frame_index: int, center: int) -> Annotations:
        """
        Read one frame's agent GT, expressed in the centre frame.
        :param frame_index: frame to read.
        :param center: frame the boxes are expressed in.
        :return: annotations following :class:`~t4_e2e_devkit.common.enums.T4BoxIndex`.
        """
        raw = self.reader._gt_frame(int(frame_index))
        boxes = np.asarray(raw["gt_boxes"], dtype=np.float64).reshape(-1, 9)
        labels = np.asarray(raw["gt_labels"], dtype=np.int64).reshape(-1)
        if frame_index != center:
            boxes = _transform_agent_boxes_to_center(
                boxes, self.reader.trajectory[frame_index], self.reader.trajectory[center]
            )
        return Annotations(boxes=np.asarray(boxes, dtype=np.float32), labels=labels)

    def read_future_annotations(self, center: int, count: int) -> List[Annotations]:
        """Agent GT over the future window, in centre-frame coordinates.

        The stationary-box bridge runs over the *complete* window rather than
        the scorer's first 40 frames.  A parked vehicle can drop out of tracking
        and be seen again after frame 40, and that later sighting is what fills
        an interior gap inside the first four seconds -- so bridging over a
        truncated window would leave holes the scorer reads as "the obstacle
        left".

        :param center: centre frame index.
        :param count: number of future frames.
        :return: one annotation set per future frame.
        """
        center_pose = self.reader.trajectory[center]
        boxes: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for frame_index in range(center, center + count + 1):
            raw = self.reader._gt_frame(int(frame_index))
            frame_boxes = np.asarray(raw["gt_boxes"], dtype=np.float64).reshape(-1, 9)
            frame_labels = np.asarray(raw["gt_labels"], dtype=np.int64).reshape(-1)
            boxes.append(
                _transform_agent_boxes_to_center(
                    frame_boxes, self.reader.trajectory[frame_index], center_pose
                )
            )
            labels.append(frame_labels)

        boxes, labels = _bridge_stationary_boxes(boxes, labels)
        return [
            Annotations(
                boxes=np.asarray(frame_boxes, dtype=np.float32),
                labels=np.asarray(frame_labels, dtype=np.int64),
            )
            for frame_boxes, frame_labels in zip(boxes, labels, strict=True)
        ]

    def camera_source(self, name: str):
        """The decoder for one camera, opened once and reused.

        Every consumer goes through this -- the loader, the visualisation and an
        audit script -- so a rendered frame is the pixels a model saw rather than
        a second decode that happens to look similar.

        :param name: camera channel name.
        :return: the :class:`~t4_e2e_devkit.dataset.camera_source.CameraSource`.
        """
        if not self._camera_sources:
            self._camera_sources = open_camera_sources(
                self.scene_dir,
                self.reader.camera_names,
                self._image_hw,
            )
        return self._camera_sources[name]

    @property
    def sync(self):
        """Per-sensor timing for this scene, or ``None`` when unavailable.

        Read from ``annotation/sample_data.json`` and ``annotation/ego_pose.json``
        in the scene directory -- not from ``derived/``, which exposes one frame
        index for every sensor and so cannot express the offsets.

        :return: a :class:`~t4_e2e_devkit.dataset.sync.SensorSync`, or ``None``.
        """
        if not self._sync_loaded:
            from t4_e2e_devkit.dataset.sync import SensorSync

            self._sync = SensorSync.open(self.scene_dir)
            self._sync_loaded = True
        return self._sync

    def read_camera_annotations(
        self, channel: str, frame_index: int, center: int
    ) -> Optional[Annotations]:
        """Boxes moved to one camera's capture time and ego frame.

        :param channel: camera channel name.
        :param frame_index: frame the camera image belongs to.
        :param center: window centre, the frame the boxes are expressed in.
        :return: corrected annotations, or ``None`` when this scene or channel
            cannot be synchronised -- never the uncorrected boxes under a name
            that implies they were corrected.
        """
        sync = self.sync
        if sync is None:
            return None
        usable, _ = sync.validate_channel(channel)
        if not usable:
            return None
        try:
            here = self.read_annotations(frame_index, center)
            later = self.read_annotations(frame_index + 1, center)
        except (KeyError, ValueError, IndexError):
            return None
        try:
            boxes = sync.corrected_boxes(
                np.asarray(here.boxes, dtype=np.float64),
                channel,
                frame_index,
                next_boxes=np.asarray(later.boxes, dtype=np.float64),
                next_labels=later.labels,
                labels=here.labels,
            )
        except Exception:  # noqa: BLE001 - a per-channel timing defect, not fatal
            return None
        return Annotations(boxes=boxes, labels=here.labels)

    def read_cameras(self, frame_index: int, iteration: int) -> Optional[Cameras]:
        """
        Decode the cameras this step's sensor config asks for.
        :param frame_index: frame to read.
        :param iteration: history step index; ``-1`` is the current frame.
        :return: the camera register, or ``None`` when none were requested.
        """
        names = self.sensor_config.camera_names_at(iteration)
        if not names:
            return None

        presence = self.reader.scalars.get("cam_presence")
        register: Dict[str, Camera] = {}
        for name in names:
            slot = self.reader.camera_names.index(name)
            scene_index = self.reader.camera_indices[slot]
            available = bool(presence[frame_index, scene_index]) if presence is not None else True
            source = self.camera_source(name)
            image = source.read(frame_index) if available else None
            sync = self.sync
            register[name] = Camera(
                name=name,
                image=image,
                intrinsics=self._scaled_intrinsics(slot, source),
                camera2ego_rotation=self.reader._selected_extrinsics[slot][:3, :3],
                camera2ego_translation=self.reader._selected_extrinsics[slot][:3, 3],
                timestamp_us=(
                    sync.channels[name].frames.get(frame_index)
                    if sync is not None and name in sync.channels
                    else None
                ),
                annotations=self.read_camera_annotations(name, frame_index, frame_index),
            )
        return Cameras(register)

    def read_lidar(self, frame_index: int, iteration: int) -> Optional[Lidar]:
        """
        Decode the LiDAR sweep when this step's sensor config asks for it.
        :param frame_index: frame to read.
        :param iteration: history step index; ``-1`` is the current frame.
        :return: the sweep, or ``None`` when not requested.
        """
        if not self.sensor_config.lidar_at(iteration):
            return None
        return Lidar(lidar_pc=np.asarray(self.reader._read_lidar(int(frame_index))))

    def read_control_state(self, frame_index: int) -> Dict[str, Any]:
        """The recorded EKF estimate at one frame.

        This is the vehicle's own kinematic report -- ``/localization/
        kinematic_state`` and ``/localization/acceleration`` plus steering and
        yaw rate -- and it is deliberately kept OUT of ``ego_status``, whose
        columns are all differenced from the pose history instead.  The reasons
        are recorded on :class:`~t4_e2e_devkit.common.enums.EgoStatusIndex`: the
        EKF stop filter zeroes low speeds, the lateral channels are never
        populated, and the twist trails the pose by about 130 ms.

        None of that disqualifies it here.  The simulator's initial state and
        the on-vehicle controller need the vehicle's own numbers, not a
        finite-difference reconstruction of them -- so this is the *runtime*
        state, used to seed a rollout and to issue a command, never as a
        learned history feature.

        :param frame_index: frame to read.
        :return: ``velocity``, ``acceleration``, ``steering`` and ``yaw_rate``.
        """
        scalars = self.reader.scalars

        def _row(name: str, width: int) -> np.ndarray:
            values = scalars.get(name)
            if values is None:
                return np.zeros(width, dtype=np.float32)
            row = np.asarray(values[frame_index], dtype=np.float32).reshape(-1)
            return row if row.size == width else np.resize(row, width)

        return {
            "velocity": _row("velocity", 2),
            "acceleration": _row("acceleration", 2),
            "steering": float(_row("steering", 1)[0]),
            "yaw_rate": float(_row("yaw_rate", 1)[0]),
        }

    def read_goal(self, center_pose: np.ndarray) -> np.ndarray:
        """
        The scene's destination, in centre-frame coordinates.
        :param center_pose: global pose of the centre frame.
        :return: ``[4]`` of ``(x, y, cos(yaw), sin(yaw))``.
        """
        goal_global = np.asarray(self.reader.scalars["goal"], dtype=np.float64)
        x, y, heading = global_to_ego(goal_global[None], center_pose)[0]
        return np.array([x, y, np.cos(heading), np.sin(heading)], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _relative_scene_dir(self) -> str:
        try:
            return str(self.scene_dir.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(self.scene_dir)

    def _timestamp_us(self, frame_index: int) -> int:
        timestamps = self.reader.scalars.get("timestamp")
        if timestamps is None or frame_index >= len(timestamps):
            return int(frame_index * T4_FRAME_DT_S * 1e6)
        return int(timestamps[frame_index])

    def _scaled_intrinsics(self, slot: int, source) -> np.ndarray:
        """Intrinsics rescaled from native resolution to reader resolution.

        The calibration in ``scalars.npz`` is for the native image, so a model
        reading resized frames needs it rescaled -- otherwise the principal
        point sits outside the image and any projection between camera and BEV
        is wrong by the resize ratio.  Verified on a real scene: unscaled, the
        front camera's principal point read 1420 px on a 1148 px-wide image.

        The native size comes from the JPEG camera source rather than from
        ``scalars.npz``, which does not carry it. The source reads the image
        header once per camera per scene.

        :param slot: index into the resolved camera register.
        :param source: the camera's
            :class:`~t4_e2e_devkit.dataset.camera_source.CameraSource`.
        :return: the ``[3, 3]`` matrix for the resized image.
        :raises WindowError: when the native size cannot be determined, since
            returning the unscaled matrix would be a silently wrong calibration.
        """
        camera_k = self.reader._selected_intrinsics[slot].astype(np.float64, copy=True)
        native = source.native_size()
        if native is None:
            raise WindowError(
                f"{self.scene_dir}: camera {self.reader.camera_names[slot]!r} has no readable "
                "frames, so its native resolution is unknown and its intrinsics cannot be "
                "rescaled to the reader resolution"
            )
        height, width = self._image_hw
        source_width, source_height = float(native[0]), float(native[1])
        camera_k[0, 0] *= width / source_width
        camera_k[0, 2] *= width / source_width
        camera_k[1, 1] *= height / source_height
        camera_k[1, 2] *= height / source_height
        return camera_k.astype(np.float32)

    def close(self) -> None:
        """Release the reader's file handles and the camera decoders."""
        for source in self._camera_sources.values():
            source.close()
        self._camera_sources.clear()
        self.reader.close()

    def __enter__(self) -> T4WindowBuilder:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def scorer_horizon_annotations(scene: T4Scene) -> List[Annotations]:
    """The future annotations the PDM scorer actually consumes.

    The window carries 80 future frames; the scorer's horizon is 40.  This
    truncates *after* the stationary bridge has run over the full window, which
    is the whole reason the two lengths differ.

    :param scene: an assembled scene.
    :return: the first ``SCORER_FUTURE_FRAMES + 1`` annotation sets.
    """
    if scene.future_annotations is None:
        raise WindowError(
            f"scene {scene.scene_metadata.token} has no future annotations; scoring "
            "requires the recorded future, and a missing one is never interpreted as "
            "an empty traffic scene"
        )
    return scene.future_annotations[: SCORER_FUTURE_FRAMES + 1]
