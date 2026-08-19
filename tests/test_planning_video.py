"""Planning-video tests.

Every scene here is synthetic with known geometry -- the same rig convention as
``test_visualization``'s camera tests -- so the numeric checks (FDE, manifest
lookup, even-dimension cropping) verify values rather than "it produced an
image".  The encoder tests need the ``ffmpeg`` binary and skip without it.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from t4_e2e_devkit.common.dataclasses import (
    Camera,
    Cameras,
    EgoShape,
    EgoStatus,
    Lidar,
    SceneMetadata,
    T4Frame,
    T4Scene,
)
from t4_e2e_devkit.evaluation.prediction_manifest import (
    PredictionManifestWriter,
    load_prediction_manifest,
)
from t4_e2e_devkit.visualization import (
    FFmpegVideoWriter,
    final_displacement_error,
    front_camera_name,
    manifest_trajectory,
    render_planning_frame,
    render_planning_video,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH"
)

SCENE_DIR = "prd_jt/scene/2026-01-01/10-00-00"


def _camera() -> Camera:
    # The camera-to-ego mapping measured on real T4 scenes: camera x -> ego -y,
    # camera y -> ego -z, camera z -> ego +x.  A gradient image keeps the
    # rendered frame from being uniform.
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    image = np.broadcast_to(gradient[None, :, None], (48, 64, 3)).copy()
    return Camera(
        name="CAM_FRONT_WIDE",
        image=image,
        camera2ego_rotation=rotation,
        camera2ego_translation=np.array([1.3, 0.0, 1.9]),
        intrinsics=np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1]]),
    )


def _lidar() -> Lidar:
    points = np.column_stack(
        [
            np.linspace(2.0, 30.0, 50),
            np.zeros(50),
            np.zeros(50),
            np.ones(50),
            np.zeros(50),
        ]
    ).astype(np.float32)
    return Lidar(lidar_pc=points)


def _scene(center: int = 10, with_camera: bool = True, with_lidar: bool = False) -> T4Scene:
    """A window driving straight ahead at 5 m/s with a 4 s recorded future."""
    shape = EgoShape(wheel_base=2.7, length=4.8, width=1.9)
    frames = []
    for step, x in enumerate((-3.0, -2.0, -1.0, 0.0)):
        frames.append(
            T4Frame(
                frame_index=center - 3 + step,
                timestamp_us=step * 100_000,
                ego_status=EgoStatus(
                    ego_pose=np.array([x, 0.0, 0.0], dtype=np.float32),
                    ego_velocity=np.array([5.0, 0.0], dtype=np.float32),
                    ego_acceleration=np.zeros(2, dtype=np.float32),
                    ego_shape=shape,
                ),
                cameras=Cameras({"CAM_FRONT_WIDE": _camera()}) if with_camera else None,
                lidar=_lidar() if with_lidar else None,
            )
        )
    # 40 future frames at 10 Hz: x = 5t, so the default 8-pose/0.5 s grid
    # resamples to exactly x = 2.5, 5.0, ..., 20.0.
    future = np.column_stack(
        [np.linspace(0.5, 20.0, 40), np.zeros(40), np.zeros(40)]
    ).astype(np.float32)
    return T4Scene(
        scene_metadata=SceneMetadata(
            scene_dir=SCENE_DIR,
            scene_id="synthetic",
            center_frame=center,
            num_history_frames=4,
            num_future_frames=40,
        ),
        frames=frames,
        future_ego_poses=future,
    )


def _manifest(tmp_path, lateral_offset: float = 0.0, centers=(10,)):
    """Write and reload a manifest whose plan replays the recorded future."""
    path = tmp_path / f"predictions_{lateral_offset:g}.jsonl"
    with PredictionManifestWriter(path, num_poses=8, interval_seconds=0.5) as writer:
        for center in centers:
            poses = np.column_stack(
                [np.linspace(2.5, 20.0, 8), np.full(8, lateral_offset), np.zeros(8)]
            )
            writer.write(SCENE_DIR, center, poses)
    return load_prediction_manifest(path)


class TestFrontCameraName:
    """The front view is rig-dependent, so it is chosen rather than assumed."""

    def test_prefers_the_centred_wide_front(self):
        names = ["CAM_FRONT_LEFT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT_WIDE"]
        assert front_camera_name(names) == "CAM_FRONT_WIDE"

    def test_x2_dev_rig_uses_cam_front(self):
        assert front_camera_name(["CAM_BACK", "CAM_FRONT"]) == "CAM_FRONT"

    def test_falls_back_to_any_front_then_first(self):
        assert front_camera_name(["CAM_BACK", "CAM_FRONT_LEFT"]) == "CAM_FRONT_LEFT"
        assert front_camera_name(["CAM_TRAFFIC_LIGHT_FAR"]) == "CAM_TRAFFIC_LIGHT_FAR"

    def test_empty_register_is_an_error(self):
        with pytest.raises(ValueError, match="empty register"):
            front_camera_name([])


class TestManifestTrajectory:
    """Lookups use the data-list key space and the manifest's own time grid."""

    def test_returns_the_declared_sampling_grid(self, tmp_path):
        trajectory = manifest_trajectory(_manifest(tmp_path), _scene(center=10))
        assert trajectory is not None
        assert len(trajectory) == 8
        assert trajectory.trajectory_sampling.interval_length == pytest.approx(0.5)

    def test_uncovered_window_returns_none(self, tmp_path):
        # A manifest over a strided data list legitimately skips centres, so a
        # miss is not an error.
        assert manifest_trajectory(_manifest(tmp_path), _scene(center=11)) is None


class TestFinalDisplacementError:
    def test_replaying_the_recorded_future_scores_zero(self, tmp_path):
        scene = _scene()
        prediction = manifest_trajectory(_manifest(tmp_path), scene)
        assert final_displacement_error(scene, prediction) == pytest.approx(0.0, abs=1e-5)

    def test_measures_the_lateral_offset(self, tmp_path):
        scene = _scene()
        prediction = manifest_trajectory(_manifest(tmp_path, lateral_offset=1.0), scene)
        assert final_displacement_error(scene, prediction) == pytest.approx(1.0, abs=1e-5)

    def test_none_without_a_recorded_future(self, tmp_path):
        scene = _scene()
        prediction = manifest_trajectory(_manifest(tmp_path), scene)
        scene.future_ego_poses = None
        assert final_displacement_error(scene, prediction) is None


class TestRenderPlanningFrame:
    def test_ground_truth_only_renders(self):
        # GT-only is the no-manifest replay of a scene and must work alone.
        image = render_planning_frame(_scene())
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8
        assert image.std() > 5.0

    def test_predictions_change_the_frame(self, tmp_path):
        scene = _scene()
        baseline = render_planning_frame(scene)
        prediction = manifest_trajectory(_manifest(tmp_path, lateral_offset=2.0), scene)
        compared = render_planning_frame(scene, {"model-a": prediction})
        assert not np.array_equal(baseline, compared)

    def test_without_cameras_is_an_error(self):
        with pytest.raises(ValueError, match="no decoded cameras"):
            render_planning_frame(_scene(with_camera=False))

    def test_unknown_camera_is_an_error(self):
        with pytest.raises(ValueError, match="was not decoded"):
            render_planning_frame(_scene(), camera="CAM_BACK_WIDE")


@needs_ffmpeg
class TestFFmpegVideoWriter:
    def test_odd_dimensions_are_cropped_even(self, tmp_path):
        # libx264/yuv420p rejects odd sizes; the writer must crop, not crash.
        out = tmp_path / "odd.mp4"
        with FFmpegVideoWriter(out, fps=5.0) as writer:
            for _ in range(3):
                writer.write(np.random.default_rng(0).integers(0, 255, (49, 63, 3), np.uint8))
        assert out.is_file() and out.stat().st_size > 0

    def test_changing_frame_size_is_an_error(self, tmp_path):
        writer = FFmpegVideoWriter(tmp_path / "size.mp4", fps=5.0)
        try:
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            with pytest.raises(ValueError, match="frame size changed"):
                writer.write(np.zeros((48, 32, 3), dtype=np.uint8))
        finally:
            writer.close()


@needs_ffmpeg
class TestRenderPlanningVideo:
    def test_end_to_end_with_manifests(self, tmp_path):
        manifests = {
            "baseline-model": _manifest(tmp_path, centers=(10, 11, 12)),
            "experiment-model": _manifest(tmp_path, lateral_offset=1.0, centers=(10, 11)),
        }
        scenes = [_scene(center=center, with_lidar=center == 10) for center in (10, 11, 12)]
        out = render_planning_video(scenes, tmp_path / "scene.mp4", manifests, fps=5.0)
        assert out.is_file() and out.stat().st_size > 0
        # Windows without a sweep hold the previous one instead of strobing.
        assert scenes[1].current_frame.lidar is scenes[0].current_frame.lidar

    def test_no_windows_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="no windows"):
            render_planning_video([], tmp_path / "empty.mp4")
