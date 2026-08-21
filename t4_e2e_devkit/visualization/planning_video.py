"""Per-frame planning videos: LiDAR bird's-eye view and camera panels.

One video frame is one :class:`~t4_e2e_devkit.common.dataclasses.T4Scene`
window, drawn the way the planning team reads its runs (the layout of
fs-perc-dataset's ``export_scene_planning.py``): the left panel is the raw
LiDAR sweep from above -- grey points on black, ego marker at the centre --
with the recorded future (white) and every model plan drawn over it; to its
right, one camera panel per model shows the road ahead with the same
trajectories projected onto the road surface through the scene's own
calibration.  A model's plan is a vehicle-width ribbon with a
green-yellow-red temporal gradient; a caption at the bottom of each camera
panel carries the model's full label and its 4 s displacement error, so a
paused frame still identifies what it compares.

Model plans arrive as prediction manifests -- the same JSONL boundary the
scorer reads -- so a video compares exactly what was scored, and any number of
manifests can be overlaid, each under its own label.  Rendering works with no
manifests at all, which is the ground-truth-only replay of a scene.

The camera is chosen by **geometry**, not by name: there is no single T4 rig,
and on ``x2_dev`` the only supported wide channel, ``CAM_FRONT_WIDE``, is
pitched about 48 degrees down at the road surface -- a name preference shows
asphalt where the scene has a street.  :func:`front_camera_for_scene` reads
each stored camera's optical axis from the scene's own calibration and picks
the one that actually looks down the road (``CAM_FRONT`` on ``x2_dev``).
Since the training reader deliberately decodes only the supported wide
channels, the video reads its display camera itself through
:class:`SceneCameraReader`, at the resolution the scene stores.

Frames stream into an ``ffmpeg`` subprocess encoding H.264, because the
devkit's only animation writer is ``frames_to_gif`` and a GIF of a full scene
is an order of magnitude larger at the same quality.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.constants import T4_NON_SURROUND_CAMERA_NAMES
from t4_e2e_devkit.common.dataclasses import Camera, T4Scene, Trajectory
from t4_e2e_devkit.evaluation.prediction_manifest import PredictionManifest
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from t4_e2e_devkit.visualization.camera import project_ego_points

#: Height of every panel; the BEV panel is this square.  700 px matches the
#: reference tool and keeps both dimensions even for the yuv420p encoder.
PANEL_HEIGHT: int = 700

#: BEV extent in ego metres.  Forward-biased on x because a plan is ahead of
#: the vehicle; the panel is square, so the spans must be equal.
BEV_X_RANGE: Tuple[float, float] = (-15.0, 55.0)
BEV_Y_RANGE: Tuple[float, float] = (-35.0, 35.0)
#: Points outside this height band -- ground returns far below the road and
#: canopy far above it -- only add noise to a top-down view.
BEV_Z_RANGE: Tuple[float, float] = (-1.5, 3.0)

#: All colours are RGB.  The two series colours are a validated categorical
#: pair for a dark surface (blue #3987e5, orange #d95926; CVD delta-E 26.8);
#: the extensions keep later manifests distinguishable on black.
GT_COLOR: Tuple[int, int, int] = (255, 255, 255)
SERIES_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (57, 135, 229),
    (217, 89, 38),
    (86, 189, 106),
    (196, 132, 224),
)
BEV_POINT_COLOR: Tuple[int, int, int] = (90, 90, 90)
EGO_COLOR: Tuple[int, int, int] = (0, 160, 255)
FRAME_COUNTER_COLOR: Tuple[int, int, int] = (255, 255, 0)

#: Opacity of the prediction ribbon over the camera image.
RIBBON_ALPHA: float = 0.45

#: Ribbon width fallback for a scene whose ego shape is unreadable.
DEFAULT_VEHICLE_WIDTH: float = 1.85

#: Horizon of the per-model displacement error shown in the caption.  Four
#: seconds, matching the scorer's horizon rather than the manifest's, so the
#: number on screen is comparable across manifests with different lengths.
FDE_HORIZON_SECONDS: float = 4.0


def _cv2():
    """OpenCV, imported on first use like matplotlib in :mod:`plots`."""
    import cv2

    return cv2


# --------------------------------------------------------------------------- #
# Camera selection and the display camera reader
# --------------------------------------------------------------------------- #


def front_camera_name(camera_names: Sequence[str]) -> str:
    """The most forward-facing camera of a register, judged by **name**.

    This is the fallback for scenes whose calibration cannot be read; prefer
    :func:`front_camera_for_scene`, which measures instead of guessing --
    ``CAM_FRONT_WIDE`` points at the road surface on ``x2_dev``.

    :param camera_names: the register to choose from.
    :return: one camera name.
    :raises ValueError: when the register is empty.
    """
    if not camera_names:
        raise ValueError("cannot pick a front camera from an empty register")
    by_upper = {name.upper(): name for name in camera_names}
    for candidate in ("CAM_FRONT", "CAM_FRONT_WIDE"):
        if candidate in by_upper:
            return by_upper[candidate]
    for name in camera_names:
        if "FRONT" in name.upper():
            return name
    return camera_names[0]


def front_camera_for_scene(scene_dir: str | Path) -> str:
    """The stored camera that actually looks down the road, by geometry.

    Reads each stored channel's optical axis from ``derived/scalars.npz`` and
    picks the one pointing most along ego-forward.  Signal-head and roof
    channels are excluded even though some point forward: they frame traffic
    lights, not the road.  Falls back to :func:`front_camera_name` when the
    calibration is unreadable.

    :param scene_dir: the T4 scene directory.
    :return: one camera name.
    :raises ValueError: when the scene stores no camera at all.
    """
    from t4_e2e_devkit.dataset.camera_source import available_cameras

    scene_dir = Path(scene_dir)
    stored = set(available_cameras(scene_dir))
    if not stored:
        raise ValueError(f"{scene_dir}: no camera directory to render")
    excluded = {name.upper() for name in T4_NON_SURROUND_CAMERA_NAMES}
    try:
        register = json.loads((scene_dir / "derived" / "cam_names.json").read_text())
        with np.load(scene_dir / "derived" / "scalars.npz") as scalars:
            extrinsics = np.asarray(scalars["cam_extrinsics"], dtype=np.float64)
        if extrinsics.shape != (len(register), 4, 4):
            raise ValueError("extrinsics/register mismatch")
    except (OSError, KeyError, ValueError):
        return front_camera_name(sorted(stored))

    best_name, best_forward = None, 0.2  # a side camera must not win by default
    for index, name in enumerate(register):
        if name not in stored or name.upper() in excluded:
            continue
        # Column 2 of camera2ego is the optical axis in ego coordinates; its x
        # component is 1.0 for a camera looking straight down the road and
        # cos(pitch) for one tilted at the asphalt.
        forward = float(extrinsics[index, 0, 2])
        if forward > best_forward:
            best_name, best_forward = name, forward
    if best_name is None:
        return front_camera_name(sorted(stored))
    return best_name


class SceneCameraReader:
    """Display-only reader for one camera channel at its stored resolution.

    The training reader decodes only the supported wide channels, at the model
    resolution -- the right boundary for model input, and the wrong one for a
    video: the geometric front camera of ``x2_dev`` is the narrow
    ``CAM_FRONT``, and the stored frames are sharper than the training crop.
    Calibration comes from ``derived/scalars.npz``, which is expressed at the
    stored resolution, so the returned :class:`Camera` projects correctly with
    no further scaling.
    """

    def __init__(self, scene_dir: str | Path, name: str) -> None:
        """
        :param scene_dir: the T4 scene directory.
        :param name: camera channel to read.
        :raises ValueError: when the scene does not calibrate this channel.
        :raises FileNotFoundError: when the scene stores no frames for it.
        """
        scene_dir = Path(scene_dir)
        register = json.loads((scene_dir / "derived" / "cam_names.json").read_text())
        lookup = {str(entry).upper(): index for index, entry in enumerate(register)}
        if name.upper() not in lookup:
            raise ValueError(
                f"{scene_dir}: camera {name!r} is not calibrated; register: {register}"
            )
        index = lookup[name.upper()]
        self.name = str(register[index])
        with np.load(scene_dir / "derived" / "scalars.npz") as scalars:
            self.intrinsics = np.asarray(scalars["cam_intrinsics"][index], dtype=np.float64)
            extrinsic = np.asarray(scalars["cam_extrinsics"][index], dtype=np.float64)
        self.rotation = extrinsic[:3, :3]
        self.translation = extrinsic[:3, 3]
        directory = scene_dir / "data" / self.name
        self._paths = {
            int(path.stem): path for path in directory.glob("*.jpg") if path.stem.isdigit()
        }
        if not self._paths:
            raise FileNotFoundError(f"{directory}: no stored frames for {self.name}")
        # Probed once from the JPEG header, not decoded: a window whose frame is
        # missing still has to produce a panel of the video's fixed size, and
        # that size follows the channel's stored resolution.
        from PIL import Image

        with Image.open(self._paths[min(self._paths)]) as probe:
            self.native_size: Tuple[int, int] = (probe.width, probe.height)

    def read(self, frame_index: int) -> Camera:
        """
        :param frame_index: frame index within the scene.
        :return: the calibrated view; ``image`` is ``None`` for a missing frame.
        """
        from PIL import Image

        path = self._paths.get(int(frame_index))
        image = None if path is None else np.asarray(Image.open(path).convert("RGB"))
        return Camera(
            name=self.name,
            image=image,
            camera2ego_rotation=self.rotation,
            camera2ego_translation=self.translation,
            intrinsics=self.intrinsics,
        )


# --------------------------------------------------------------------------- #
# Manifest lookup and the caption metric
# --------------------------------------------------------------------------- #


def manifest_trajectory(manifest: PredictionManifest, scene: T4Scene) -> Optional[Trajectory]:
    """The manifest's plan for one window, on the manifest's own time grid.

    Manifest records are keyed exactly like data-list rows, so the lookup key
    is the window's relative scene directory and centre frame.  A window the
    manifest does not cover returns ``None`` rather than raising: a manifest
    written over a strided data list legitimately skips centres.

    :param manifest: a loaded prediction manifest.
    :param scene: the window to look up.
    :return: the plan as a :class:`Trajectory`, or ``None``.
    """
    metadata = scene.scene_metadata
    record = manifest.records.get((metadata.scene_dir, metadata.center_frame))
    if record is None:
        return None
    sampling = manifest.header["trajectory"]
    return Trajectory(
        poses=record.poses,
        trajectory_sampling=TrajectorySampling(
            num_poses=int(sampling["num_poses"]),
            interval_length=float(sampling["interval_seconds"]),
        ),
    )


def final_displacement_error(
    scene: T4Scene,
    prediction: Trajectory,
    horizon_seconds: float = FDE_HORIZON_SECONDS,
) -> Optional[float]:
    """Displacement between a plan and the recorded future at one horizon.

    The recorded future is resampled onto the prediction's own grid, so the
    two poses being compared are at the same instant regardless of the
    manifest's interval.  Returns ``None`` when the window carries no future
    or cannot cover the prediction's horizon, so the caption omits a number
    rather than showing a wrong one.

    :param scene: the window supplying the recorded future.
    :param prediction: the plan to measure.
    :param horizon_seconds: where to measure, clipped to the plan's horizon.
    :return: the error in metres, or ``None``.
    """
    if scene.future_ego_poses is None:
        return None
    try:
        ground_truth = scene.get_future_trajectory(
            trajectory_sampling=prediction.trajectory_sampling
        )
    except ValueError:
        return None
    interval = float(prediction.trajectory_sampling.interval_length)
    index = min(len(prediction), max(1, int(round(horizon_seconds / interval)))) - 1
    return float(np.linalg.norm(prediction.poses[index, :2] - ground_truth.poses[index, :2]))


# --------------------------------------------------------------------------- #
# Drawing primitives (OpenCV, RGB throughout)
# --------------------------------------------------------------------------- #


def _temporal_colors(count: int) -> List[Tuple[int, int, int]]:
    """Green -> yellow -> red gradient over ``count`` steps: now to horizon."""
    colors = []
    for t in np.linspace(0.0, 1.0, count):
        if t < 0.5:
            colors.append((int(255 * t * 2), 255, 0))
        else:
            colors.append((255, int(255 * (1 - (t - 0.5) * 2)), 0))
    return colors


def _with_origin(poses: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Trajectory xy with the implicit t=0 ego position prepended."""
    return np.vstack([[0.0, 0.0], np.asarray(poses, dtype=np.float64)[:, :2]])


def _bev_panel(
    scene: T4Scene,
    ground_truth: Optional[Trajectory],
    predictions: Sequence[Tuple[str, Optional[Trajectory], Tuple[int, int, int]]],
    size: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> npt.NDArray[np.uint8]:
    """The top-down panel: LiDAR points, trajectories, ego marker, legend."""
    cv2 = _cv2()
    panel = np.zeros((size, size, 3), dtype=np.uint8)

    def to_px(xy: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
        # Ego frame is x forward, y left; the panel puts forward up, left left.
        u = (y_range[1] - xy[:, 1]) / (y_range[1] - y_range[0]) * (size - 1)
        v = (x_range[1] - xy[:, 0]) / (x_range[1] - x_range[0]) * (size - 1)
        return np.column_stack([u, v]).astype(int)

    lidar = scene.current_frame.lidar
    points = None if lidar is None else lidar.lidar_pc
    if points is not None and len(points):
        inside = (
            (points[:, 0] > x_range[0])
            & (points[:, 0] < x_range[1])
            & (points[:, 1] > y_range[0])
            & (points[:, 1] < y_range[1])
            & (points[:, 2] > BEV_Z_RANGE[0])
            & (points[:, 2] < BEV_Z_RANGE[1])
        )
        pixels = to_px(points[inside, :2])
        panel[pixels[:, 1], pixels[:, 0]] = BEV_POINT_COLOR

    def draw_line(xy, color, thickness, dots=True):
        pixels = to_px(_with_origin(xy))
        for a in range(len(pixels) - 1):
            cv2.line(
                panel,
                tuple(int(v) for v in pixels[a]),
                tuple(int(v) for v in pixels[a + 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
        if dots:
            for pixel in pixels:
                cv2.circle(panel, tuple(int(v) for v in pixel), 3, color, -1, cv2.LINE_AA)

    if ground_truth is not None:
        draw_line(ground_truth.poses, GT_COLOR, 2)

    # One prediction gets the temporal rainbow, as on the camera panels.
    # Several get one flat colour each -- two rainbows on one panel cannot be
    # told apart -- with a legend so identity never rests on colour alone.
    # The single/several split is by declared model, not by which happen to
    # cover this window, so a video's colour language never changes mid-play.
    present = [
        (trajectory, color) for _, trajectory, color in predictions if trajectory is not None
    ]
    if len(predictions) == 1:
        for trajectory, _ in present:
            pixels = to_px(_with_origin(trajectory.poses))
            colors = _temporal_colors(len(pixels))
            for a in range(len(pixels) - 1):
                cv2.line(
                    panel,
                    tuple(int(v) for v in pixels[a]),
                    tuple(int(v) for v in pixels[a + 1]),
                    colors[a + 1],
                    4,
                    cv2.LINE_AA,
                )
    else:
        for trajectory, color in present:
            draw_line(trajectory.poses, color, 3)

    ego = to_px(np.zeros((1, 2)))[0]
    cv2.rectangle(
        panel,
        (int(ego[0]) - 5, int(ego[1]) - 9),
        (int(ego[0]) + 5, int(ego[1]) + 9),
        EGO_COLOR,
        2,
    )

    if len(predictions) > 1:
        entries = [("GT", GT_COLOR)] + [(label, color) for label, _, color in predictions]
        for row, (text, color) in enumerate(entries):
            y = 58 + row * 26  # below the header line drawn by the caller
            cv2.line(panel, (14, y - 5), (40, y - 5), color, 3, cv2.LINE_AA)
            cv2.putText(
                panel, text, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_COLOR, 1, cv2.LINE_AA
            )
    return panel


def _draw_camera_ground_truth(panel, camera: Camera, poses, scale: float) -> None:
    """White polyline with dots, clipped where it passes behind the camera."""
    cv2 = _cv2()
    xy = _with_origin(poses)
    points = np.column_stack([xy, np.zeros(len(xy))])
    pixels, valid = project_ego_points(camera, points)
    pixels = (pixels * scale).astype(int)
    for a in range(len(pixels) - 1):
        if valid[a] and valid[a + 1]:
            cv2.line(
                panel,
                tuple(int(v) for v in pixels[a]),
                tuple(int(v) for v in pixels[a + 1]),
                GT_COLOR,
                2,
                cv2.LINE_AA,
            )
    for pixel, in_front in zip(pixels, valid, strict=True):
        if in_front:
            cv2.circle(panel, tuple(int(v) for v in pixel), 4, GT_COLOR, -1, cv2.LINE_AA)


def _draw_camera_ribbon(panel, camera: Camera, poses, scale: float, vehicle_width: float) -> None:
    """A model plan as a vehicle-width ribbon with the temporal gradient."""
    cv2 = _cv2()
    poses = np.asarray(poses, dtype=np.float64)
    xy = _with_origin(poses)
    headings = np.concatenate([[0.0], poses[:, 2]])
    count = len(xy)
    perpendicular = np.column_stack([-np.sin(headings), np.cos(headings)]) * (vehicle_width / 2)
    ground = np.zeros(count)
    left, left_ok = project_ego_points(camera, np.column_stack([xy + perpendicular, ground]))
    right, right_ok = project_ego_points(camera, np.column_stack([xy - perpendicular, ground]))
    left = (left * scale).astype(int)
    right = (right * scale).astype(int)

    overlay = panel.copy()
    colors = _temporal_colors(count)
    for a in range(count - 1):
        b = a + 1
        if not (left_ok[a] and right_ok[a] and left_ok[b] and right_ok[b]):
            continue
        quad = np.array([left[a], left[b], right[b], right[a]])
        cv2.fillPoly(overlay, [quad], colors[b])
    panel[:] = cv2.addWeighted(overlay, RIBBON_ALPHA, panel, 1.0 - RIBBON_ALPHA, 0)


def _camera_panel(
    camera: Camera,
    ground_truth: Optional[Trajectory],
    prediction: Optional[Trajectory],
    panel_height: int,
    vehicle_width: float,
    caption: Optional[str],
    missing_size: Optional[Tuple[int, int]] = None,
) -> npt.NDArray[np.uint8]:
    """One camera panel: image, projected trajectories, bottom caption.

    ``missing_size`` is the channel's stored ``(width, height)``, used only when
    this window has no frame.  A missing frame is a fact about the data, not a
    crash -- but the placeholder has to be the size the decoded panels are, or
    the encoder refuses the frame and one dropped frame kills the whole video.
    """
    cv2 = _cv2()
    if camera.image is None:
        if missing_size is not None:
            width, height = (int(value) for value in missing_size)
            panel_width = int(round(width * panel_height / height)) // 2 * 2
        else:
            # No stored resolution to follow: a plausible aspect keeps a
            # hand-built single frame renderable.
            panel_width = panel_height * 16 // 9 // 2 * 2
        panel = np.zeros((panel_height, panel_width, 3), np.uint8)
        cv2.putText(
            panel,
            "no image",
            (panel.shape[1] // 2 - 60, panel_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            GT_COLOR,
            2,
            cv2.LINE_AA,
        )
    else:
        height, width = camera.image.shape[:2]
        scale = panel_height / height
        panel = np.ascontiguousarray(
            cv2.resize(camera.image, (int(round(width * scale)), panel_height))
        )
        if camera.is_calibrated:
            if ground_truth is not None:
                _draw_camera_ground_truth(panel, camera, ground_truth.poses, scale)
            if prediction is not None:
                _draw_camera_ribbon(panel, camera, prediction.poses, scale, vehicle_width)
    if caption:
        cv2.putText(
            panel,
            caption,
            (14, panel_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            GT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return panel


# --------------------------------------------------------------------------- #
# Frame and video assembly
# --------------------------------------------------------------------------- #


def render_planning_frame(
    scene: T4Scene,
    predictions: Optional[Mapping[str, Optional[Trajectory]]] = None,
    camera: Optional[str] = None,
    panel_height: int = PANEL_HEIGHT,
    view_range: Optional[float] = None,
    camera_size: Optional[Tuple[int, int]] = None,
) -> npt.NDArray[np.uint8]:
    """Render one window as one video frame: BEV left, camera panels right.

    The recorded future is drawn in both panels when the window carries it,
    white in both.  Every camera panel keeps the temporal-gradient ribbon --
    the panels differ by model, so gradient ambiguity cannot arise there --
    while the BEV switches from the gradient to flat legend colours as soon as
    a second prediction appears.

    :param scene: the window; its current frame must carry a camera view.
    :param predictions: label -> plan in ego coordinates of this window, or
        ``None`` for a model that skipped this centre (its panel stays).
    :param camera: camera to show; the first of the register by default.
    :param panel_height: pixel height of the frame; the BEV is this square.
    :param view_range: symmetric BEV half-extent in metres, overriding the
        forward-biased default of :data:`BEV_X_RANGE` / :data:`BEV_Y_RANGE`.
    :param camera_size: the display channel's stored ``(width, height)``.  Only
        used when this window has no frame, to keep the placeholder panel the
        size the decoded panels are; see :func:`render_planning_video`.
    :return: ``[H, W, 3]`` uint8 RGB.
    :raises ValueError: when no camera view is present, or the named one is not.
    """
    cv2 = _cv2()
    frame = scene.current_frame
    if frame.cameras is None or len(frame.cameras) == 0:
        raise ValueError(
            "this window carries no decoded cameras; render through "
            "render_scene_video, or attach a SceneCameraReader view"
        )
    name = camera if camera is not None else front_camera_name(frame.cameras.names)
    if name not in frame.cameras:
        raise ValueError(
            f"camera {name!r} was not decoded for this window; decoded: {frame.cameras.names}"
        )
    view = frame.cameras[name]

    ground_truth: Optional[Trajectory] = None
    if scene.future_ego_poses is not None:
        ground_truth = scene.get_future_trajectory()
    entries = [
        (label, trajectory, SERIES_COLORS[index % len(SERIES_COLORS)])
        for index, (label, trajectory) in enumerate((predictions or {}).items())
    ]

    if view_range is not None:
        x_range = y_range = (-float(view_range), float(view_range))
    else:
        x_range, y_range = BEV_X_RANGE, BEV_Y_RANGE
    bev = _bev_panel(scene, ground_truth, entries, panel_height, x_range, y_range)

    shape = scene.current_frame.ego_status.ego_shape
    vehicle_width = float(getattr(shape, "width", 0) or DEFAULT_VEHICLE_WIDTH)
    panels = []
    if not entries:
        panels.append(
            _camera_panel(view, ground_truth, None, panel_height, vehicle_width, None, camera_size)
        )
    for label, trajectory, _ in entries:
        # A model that skipped this centre keeps its panel -- one panel per
        # declared model, always, so the video geometry never changes -- and
        # the caption says why it is empty.
        if trajectory is None:
            caption = f"{label}   (no plan)"
        else:
            error = final_displacement_error(scene, trajectory)
            caption = label if error is None else f"{label}   FDE {error:.2f}m"
        panels.append(
            _camera_panel(
                view,
                ground_truth,
                trajectory,
                panel_height,
                vehicle_width,
                caption,
                camera_size,
            )
        )

    image = np.hstack([bev, *panels])
    header = f"{scene.scene_metadata.scene_id[:8]}  GT=white"
    cv2.putText(image, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"frame {scene.scene_metadata.center_frame:04d}",
        (12, panel_height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        FRAME_COUNTER_COLOR,
        2,
        cv2.LINE_AA,
    )
    return image


class FFmpegVideoWriter:
    """Stream RGB frames into an ``ffmpeg`` subprocess encoding H.264 mp4.

    libx264 with yuv420p output rejects odd frame dimensions, so every frame
    is cropped to even width and height before it enters the pipe -- one row
    or column at most, invisible next to a failed encode.

    Use as a context manager, or call :meth:`close` explicitly; the file is
    not playable until the encoder has been allowed to finish.
    """

    def __init__(self, path: str | Path, fps: float = 10.0, crf: int = 26) -> None:
        """
        :param path: destination ``.mp4`` path; parent directories are created.
        :param fps: frames per second.
        :param crf: libx264 quality; lower is better and larger.
        :raises FileNotFoundError: when the ``ffmpeg`` binary is not on PATH.
        """
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError(
                "ffmpeg is not on PATH; planning videos are encoded through the ffmpeg binary"
            )
        self.path = Path(path)
        self.fps = float(fps)
        self.crf = int(crf)
        self._process: Optional[subprocess.Popen] = None
        self._size: Optional[Tuple[int, int]] = None  # (width, height)

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        """
        Append one frame.  The first frame fixes the video dimensions.
        :param frame: ``[H, W, 3]`` uint8 RGB.
        :raises ValueError: on a non-RGB frame, or one whose size changed.
        """
        frame = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an [H, W, 3] RGB frame, got shape {frame.shape}")
        height = frame.shape[0] // 2 * 2
        width = frame.shape[1] // 2 * 2
        frame = frame[:height, :width]

        if self._process is None:
            self._size = (width, height)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    str(self.crf),
                    "-pix_fmt",
                    "yuv420p",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        elif (width, height) != self._size:
            raise ValueError(
                f"frame size changed from {self._size} to {(width, height)}; "
                "every frame of one video must have the size of the first"
            )
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        """Finish the encode.  Safe to call when nothing was written."""
        if self._process is None:
            return
        self._process.stdin.close()
        returncode = self._process.wait()
        self._process = None
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with status {returncode} writing {self.path}")

    def __enter__(self) -> "FFmpegVideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def render_planning_video(
    scenes: Iterable[T4Scene],
    out_path: str | Path,
    manifests: Optional[Mapping[str, PredictionManifest]] = None,
    *,
    camera: Optional[str] = None,
    fps: float = 10.0,
    view_range: Optional[float] = None,
    panel_height: int = PANEL_HEIGHT,
    crf: int = 26,
    camera_size: Optional[Tuple[int, int]] = None,
) -> Path:
    """Render a sequence of windows into one mp4.

    ``scenes`` is an iterable rather than a builder so the caller decides how
    windows are produced; :func:`render_scene_video` is the standard producer
    over one scene's data-list centres, which is also exactly the key space a
    prediction manifest covers.

    When a window carries no LiDAR sweep, the previous window's sweep is held
    instead of leaving the panel empty: some rigs run the camera at twice the
    LiDAR rate, and a cloud that blinks on alternate frames reads as a defect.

    :param scenes: the windows to render, in playback order.
    :param out_path: destination ``.mp4`` path.
    :param manifests: label -> prediction manifest to overlay; may be empty,
        which renders the recorded future only.
    :param camera: camera to show; the first of each register by default.
    :param fps: frames per second.
    :param view_range: symmetric BEV half-extent in metres; the forward-biased
        default otherwise.
    :param panel_height: pixel height of the video.
    :param crf: libx264 quality; lower is better and larger.
    :param camera_size: the display channel's stored ``(width, height)``.  A
        window whose frame is missing then gets a placeholder panel of the same
        size as the decoded ones; without it the first dropped frame changes the
        frame size, which the encoder refuses.  Falls back to the size of the
        last decoded frame.
    :return: the written path.
    :raises ValueError: when ``scenes`` yields nothing.
    """
    frames_written = 0
    last_lidar = None
    stored_size = camera_size
    with FFmpegVideoWriter(out_path, fps=fps, crf=crf) as writer:
        for scene in scenes:
            frame = scene.current_frame
            if frame.lidar is not None and frame.lidar.lidar_pc is not None:
                last_lidar = frame.lidar
            else:
                frame.lidar = last_lidar
            # Learn the panel size from the frames that do decode, so a caller
            # that assembled its own views still survives a missing frame.
            if stored_size is None and frame.cameras is not None:
                for view in frame.cameras:
                    if view.image is not None:
                        stored_size = (view.image.shape[1], view.image.shape[0])
                        break
            # Every label stays in the mapping even when this window is not
            # covered (a manifest over a strided list legitimately skips
            # centres): the panel layout is per model, not per hit.
            predictions: Dict[str, Optional[Trajectory]] = {
                label: manifest_trajectory(manifest, scene)
                for label, manifest in (manifests or {}).items()
            }
            writer.write(
                render_planning_frame(
                    scene,
                    predictions,
                    camera=camera,
                    panel_height=panel_height,
                    view_range=view_range,
                    camera_size=stored_size,
                )
            )
            frames_written += 1
    if frames_written == 0:
        raise ValueError(f"no windows to render, so {out_path} was not written")
    return Path(out_path)


def render_scene_video(
    data_list,
    scene_rel: str,
    out_path: str | Path,
    manifests: Optional[Mapping[str, PredictionManifest]] = None,
    *,
    camera: Optional[str] = None,
    fps: float = 10.0,
    view_range: Optional[float] = None,
    lidar: bool = True,
    panel_height: int = PANEL_HEIGHT,
) -> Path:
    """Render one scene's video from a data list: the standard producer.

    Windows come from a LiDAR-only :class:`T4WindowBuilder` over the scene's
    data-list centres; the display camera is read separately through
    :class:`SceneCameraReader`, because the geometric front camera is not
    necessarily a channel the training reader may decode.

    LiDAR is opt-out and a scene may simply not ship a pack (trimmed sample
    copies, camera-only exports); the BEV then renders without points -- dark
    background, ego and trajectories still drawn -- with a printed note,
    instead of failing the whole video.

    :param data_list: a loaded data list (see ``load_data_list``).
    :param scene_rel: relative scene directory, as listed in the data list.
    :param out_path: destination ``.mp4`` path.
    :param manifests: label -> prediction manifest to overlay.
    :param camera: camera to show; the geometric front camera by default.
    :param fps: frames per second.
    :param view_range: symmetric BEV half-extent in metres; the forward-biased
        default otherwise.
    :param lidar: read the LiDAR sweeps for the BEV panel.
    :param panel_height: pixel height of the video.
    :return: the written path.
    :raises ValueError: when the data list has no windows for this scene.
    """
    from t4_e2e_devkit.common.dataclasses import Cameras, SensorConfig
    from t4_e2e_devkit.dataset.window import T4WindowBuilder

    centers = sorted({center for scene, center in data_list.rows if scene == scene_rel})
    if not centers:
        raise ValueError(f"the data list has no windows for scene {scene_rel!r}")
    scene_dir = data_list.absolute_scene_dir(scene_rel)
    name = camera if camera is not None else front_camera_for_scene(scene_dir)
    reader = SceneCameraReader(scene_dir, name)

    if lidar and not (scene_dir / "data" / "LIDAR_CONCAT.pack").is_file():
        print(f"note: {scene_rel} has no LiDAR pack; rendering the BEV without points")
        lidar = False
    builder = T4WindowBuilder(
        scene_dir,
        data_list.root,
        sensor_config=SensorConfig(cameras={}, lidar=[-1] if lidar else False),
    )

    def windows() -> Iterable[T4Scene]:
        for center in centers:
            scene = builder.build(center)
            scene.current_frame.cameras = Cameras({reader.name: reader.read(center)})
            yield scene

    try:
        return render_planning_video(
            windows(),
            out_path,
            manifests,
            camera=reader.name,
            fps=fps,
            view_range=view_range,
            panel_height=panel_height,
            camera_size=reader.native_size,
        )
    finally:
        builder.close()
