"""Per-frame planning videos: one camera view and the BEV, side by side.

One video frame is one :class:`~t4_e2e_devkit.common.dataclasses.T4Scene`
window: the BEV panel shows the map, agents, recorded future and every model
plan in ego coordinates, and the camera panel shows the same trajectories
projected onto the road surface through the scene's own calibration.  Played
over a scene's windows this is the most direct way to see *where* a planner
diverges from the human driver -- a per-window score says how far, the video
says at which corner.

Model plans arrive as prediction manifests -- the same JSONL boundary the
scorer reads -- so a video compares exactly what was scored, and any number of
manifests can be overlaid, each under its own label.  Rendering works with no
manifests at all, which is the ground-truth-only replay of a scene.

Everything draws through the existing single-frame primitives
(:func:`~t4_e2e_devkit.visualization.plots.plot_bev_frame`,
:func:`~t4_e2e_devkit.visualization.camera.add_trajectory_to_camera_ax`), so a
video frame and a saved PNG of the same window cannot disagree.  Only the
mp4 encoding is new: frames stream into an ``ffmpeg`` subprocess, because the
devkit's only animation writer is :func:`frames_to_gif` and a GIF of a full
scene is an order of magnitude larger than H.264 at the same quality.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.dataclasses import T4Scene, Trajectory
from t4_e2e_devkit.evaluation.prediction_manifest import PredictionManifest
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from t4_e2e_devkit.visualization.bev import add_trajectory_to_bev_ax
from t4_e2e_devkit.visualization.camera import add_camera_ax, add_trajectory_to_camera_ax
from t4_e2e_devkit.visualization.config import TAB_10, TRAJECTORY_CONFIG
from t4_e2e_devkit.visualization.plots import (
    _pyplot,
    describe_scene,
    figure_to_rgb,
    plot_bev_frame,
)

#: Canvas of one video frame.  ``figure_size * dpi`` is the pixel size, chosen
#: even on both axes because libx264's yuv420p output requires it; the writer
#: still guards against odd sizes for callers who override this.
VIDEO_FIGURE_SIZE: Tuple[float, float] = (16.0, 6.4)
VIDEO_DPI: int = 100

#: BEV layers for a video frame.  ``lidar`` is listed but draws only when the
#: window actually carries a sweep, so a no-LiDAR run pays nothing for it.
VIDEO_BEV_LAYERS: Tuple[str, ...] = (
    "lidar",
    "polygons",
    "line_strings",
    "lanes",
    "route_lanes",
    "annotations",
)

#: Horizon of the per-model displacement error shown in the caption.  Four
#: seconds, matching the scorer's horizon rather than the manifest's, so the
#: number on screen is comparable across manifests with different lengths.
FDE_HORIZON_SECONDS: float = 4.0


def front_camera_name(camera_names: Sequence[str]) -> str:
    """The most forward-facing camera of a register.

    There is no single T4 rig, so the front view cannot be a constant: prd_jt
    stores ``CAM_FRONT_WIDE`` where x2_dev stores ``CAM_FRONT``.  Prefers the
    centred front views, then any front view, then the first camera.

    :param camera_names: the register to choose from.
    :return: one camera name.
    :raises ValueError: when the register is empty.
    """
    if not camera_names:
        raise ValueError("cannot pick a front camera from an empty register")
    by_upper = {name.upper(): name for name in camera_names}
    for candidate in ("CAM_FRONT_WIDE", "CAM_FRONT"):
        if candidate in by_upper:
            return by_upper[candidate]
    for name in camera_names:
        if "FRONT" in name.upper():
            return name
    return camera_names[0]


def manifest_trajectory(
    manifest: PredictionManifest, scene: T4Scene
) -> Optional[Trajectory]:
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


def render_planning_frame(
    scene: T4Scene,
    predictions: Optional[Mapping[str, Trajectory]] = None,
    camera: Optional[str] = None,
    view_range: Optional[float] = None,
) -> npt.NDArray[np.uint8]:
    """Render one window as one video frame: BEV left, camera right.

    The recorded future is drawn in both panels when the window carries it;
    each prediction gets one stable categorical colour in both panels and one
    caption entry carrying its **full label** and its
    :data:`FDE_HORIZON_SECONDS` displacement error.  The caption is built from
    explicit proxy artists rather than the frame's own artists, so its content
    and geometry stay constant across a video.

    :param scene: the window; its cameras must have been decoded.
    :param predictions: label -> plan, in ego coordinates of this window.
    :param camera: camera to show; the front-most of the register by default.
    :param view_range: BEV half-extent in metres; the shared default otherwise.
    :return: ``[H, W, 3]`` uint8 RGB.
    :raises ValueError: when no cameras were decoded, or the named one was not.
    """
    from matplotlib.lines import Line2D

    plt = _pyplot()
    frame = scene.current_frame
    if frame.cameras is None or len(frame.cameras) == 0:
        raise ValueError(
            "this window carries no decoded cameras; build it with a SensorConfig "
            "that requests them, e.g. rigs.sensor_config_for_scene(scene_dir)"
        )
    name = camera if camera is not None else front_camera_name(frame.cameras.names)
    if name not in frame.cameras:
        raise ValueError(
            f"camera {name!r} was not decoded for this window; "
            f"decoded: {frame.cameras.names}"
        )
    view = frame.cameras[name]

    figure = plt.figure(figsize=VIDEO_FIGURE_SIZE, dpi=VIDEO_DPI)
    spec = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.6))
    bev_ax = figure.add_subplot(spec[0, 0])
    camera_ax = figure.add_subplot(spec[0, 1])

    config: Dict[str, object] = {"layers": list(VIDEO_BEV_LAYERS), "legend": False}
    if view_range is not None:
        config["view_range"] = float(view_range)

    ground_truth: Optional[Trajectory] = None
    trajectories: Dict[str, Trajectory] = {}
    if scene.future_ego_poses is not None:
        ground_truth = scene.get_future_trajectory()
        trajectories["ground_truth"] = ground_truth
    plot_bev_frame(scene, trajectories, config, ax=bev_ax, title="")

    add_camera_ax(camera_ax, view, title=view.name)
    handles = []
    labels = []
    if ground_truth is not None:
        settings = TRAJECTORY_CONFIG["ground_truth"]
        add_trajectory_to_camera_ax(camera_ax, view, ground_truth.poses, settings["color"])
        handles.append(Line2D([0], [0], color=settings["color"], marker="o", markersize=5))
        labels.append(str(settings["label"]))

    for index, (label, prediction) in enumerate((predictions or {}).items()):
        color = TAB_10[index % len(TAB_10)]
        add_trajectory_to_bev_ax(
            bev_ax, prediction, kind="prediction", config={"color": color, "label": None}
        )
        add_trajectory_to_camera_ax(camera_ax, view, prediction.poses, color)
        error = final_displacement_error(scene, prediction)
        handles.append(Line2D([0], [0], color=color, linewidth=2.0))
        labels.append(label if error is None else f"{label}  FDE {error:.2f} m")

    if handles:
        bev_ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.85)
    figure.suptitle(describe_scene(scene), fontsize=10)
    figure.tight_layout()
    image = figure_to_rgb(figure)
    plt.close(figure)
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
                "ffmpeg is not on PATH; planning videos are encoded through the "
                "ffmpeg binary"
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
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
                    "-pix_fmt", "yuv420p", str(self.path),
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
    crf: int = 26,
) -> Path:
    """Render a sequence of windows into one mp4.

    ``scenes`` is an iterable rather than a builder so the caller decides how
    windows are produced -- typically ``(builder.build(c) for c in centers)``
    over one scene's data-list centres, which is also exactly the key space a
    prediction manifest covers.

    When a window carries no LiDAR sweep, the previous window's sweep is held
    instead of leaving the panel empty: some rigs run the camera at twice the
    LiDAR rate, and a cloud that blinks on alternate frames reads as a defect.

    :param scenes: the windows to render, in playback order.
    :param out_path: destination ``.mp4`` path.
    :param manifests: label -> prediction manifest to overlay; may be empty,
        which renders the recorded future only.
    :param camera: camera to show; the front-most of each register by default.
    :param fps: frames per second.
    :param view_range: BEV half-extent in metres; the shared default otherwise.
    :param crf: libx264 quality; lower is better and larger.
    :return: the written path.
    :raises ValueError: when ``scenes`` yields nothing.
    """
    frames_written = 0
    last_lidar = None
    with FFmpegVideoWriter(out_path, fps=fps, crf=crf) as writer:
        for scene in scenes:
            frame = scene.current_frame
            if frame.lidar is not None and frame.lidar.lidar_pc is not None:
                last_lidar = frame.lidar
            else:
                frame.lidar = last_lidar
            predictions: Dict[str, Trajectory] = {}
            for label, manifest in (manifests or {}).items():
                trajectory = manifest_trajectory(manifest, scene)
                if trajectory is not None:
                    predictions[label] = trajectory
            writer.write(
                render_planning_frame(scene, predictions, camera=camera, view_range=view_range)
            )
            frames_written += 1
    if frames_written == 0:
        raise ValueError(f"no windows to render, so {out_path} was not written")
    return Path(out_path)
