"""Planning-video tests.

Every scene here is synthetic with known geometry -- the same rig convention as
``test_visualization``'s camera tests -- so the numeric checks (FDE, manifest
lookup, BEV pixel positions, even-dimension cropping) verify values rather than
"it produced an image".  The encoder tests need the ``ffmpeg`` binary and skip
without it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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
    Trajectory,
)
from t4_e2e_devkit.evaluation.prediction_manifest import (
    PredictionManifestWriter,
    load_prediction_manifest,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from t4_e2e_devkit.visualization import (
    FFmpegVideoWriter,
    SceneCameraReader,
    final_displacement_error,
    front_camera_for_scene,
    front_camera_name,
    manifest_trajectory,
    render_planning_frame,
    render_planning_video,
)
from t4_e2e_devkit.visualization.planning_video import BEV_POINT_COLOR

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")

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
    # Points 5 m to the ego's left, so they never share a BEV pixel column with
    # the straight-ahead ground truth and their grey stays checkable.
    points = np.column_stack(
        [
            np.linspace(2.0, 30.0, 50),
            np.full(50, 5.0),
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
    future = np.column_stack([np.linspace(0.5, 20.0, 40), np.zeros(40), np.zeros(40)]).astype(
        np.float32
    )
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


def _manifest(tmp_path, lateral_offset: float = 0.0, centers=(10,), data_list=None):
    """Write and reload a manifest whose plan replays the recorded future."""
    path = tmp_path / f"predictions_{lateral_offset:g}.jsonl"
    with PredictionManifestWriter(
        path, data_list=data_list, num_poses=8, interval_seconds=0.5
    ) as writer:
        for center in centers:
            poses = np.column_stack(
                [np.linspace(2.5, 20.0, 8), np.full(8, lateral_offset), np.zeros(8)]
            )
            writer.write(SCENE_DIR, center, poses)
    return load_prediction_manifest(path)


class TestFrontCameraName:
    """The name-based fallback for scenes whose calibration cannot be read."""

    def test_prefers_the_centred_narrow_front(self):
        # On the one rig storing both, the narrow CAM_FRONT is the level view;
        # CAM_FRONT_WIDE points at the asphalt there.
        assert front_camera_name(["CAM_FRONT_WIDE", "CAM_FRONT"]) == "CAM_FRONT"

    def test_uses_the_centred_wide_front_otherwise(self):
        names = ["CAM_FRONT_LEFT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT_WIDE"]
        assert front_camera_name(names) == "CAM_FRONT_WIDE"

    def test_falls_back_to_any_front_then_first(self):
        assert front_camera_name(["CAM_BACK", "CAM_FRONT_LEFT"]) == "CAM_FRONT_LEFT"
        assert front_camera_name(["CAM_TRAFFIC_LIGHT_FAR"]) == "CAM_TRAFFIC_LIGHT_FAR"

    def test_empty_register_is_an_error(self):
        with pytest.raises(ValueError, match="empty register"):
            front_camera_name([])


def _display_scene_dir(tmp_path, tilt_front_down: bool = False):
    """A scene directory with just enough calibration for camera selection.

    Camera-to-ego rotation follows the measured T4 convention (camera z is the
    optical axis).  CAM_FRONT looks down the road, CAM_FRONT_WIDE is pitched
    about 48 degrees at the asphalt, and the traffic-light camera also points
    forward -- exactly the x2_dev situation that name preferences get wrong.
    """
    from PIL import Image

    scene = tmp_path / "scene"
    names = ["CAM_FRONT", "CAM_FRONT_WIDE", "CAM_TRAFFIC_LIGHT_FAR"]
    level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    tilted = level.copy()
    tilted[:, 2] = [0.669, 0.0, -0.743]  # optical axis 48 degrees down

    extrinsics = np.stack([np.eye(4)] * 3)
    extrinsics[0, :3, :3] = tilted if tilt_front_down else level
    extrinsics[1, :3, :3] = level if tilt_front_down else tilted
    extrinsics[2, :3, :3] = level
    extrinsics[:, :3, 3] = [1.3, 0.0, 1.9]
    intrinsics = np.stack([np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1]])] * 3)

    derived = scene / "derived"
    derived.mkdir(parents=True)
    (derived / "cam_names.json").write_text(json.dumps(names))
    np.savez(derived / "scalars.npz", cam_intrinsics=intrinsics, cam_extrinsics=extrinsics)

    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    image = np.broadcast_to(gradient[None, :, None], (48, 64, 3))
    for name in names:
        (scene / "data" / name).mkdir(parents=True)
        Image.fromarray(image).save(scene / "data" / name / "00007.jpg")
    return scene


class TestFrontCameraForScene:
    """The front view is measured from the calibration, not guessed by name."""

    def test_picks_the_level_camera_over_the_tilted_one(self, tmp_path):
        assert front_camera_for_scene(_display_scene_dir(tmp_path)) == "CAM_FRONT"
        assert (
            front_camera_for_scene(_display_scene_dir(tmp_path / "b", tilt_front_down=True))
            == "CAM_FRONT_WIDE"
        )

    def test_signal_cameras_never_win(self, tmp_path):
        # The traffic-light camera points forward too; geometry alone would
        # tie it with CAM_FRONT, so it is excluded by role.
        scene = _display_scene_dir(tmp_path)
        shutil.rmtree(scene / "data" / "CAM_FRONT")
        assert front_camera_for_scene(scene) == "CAM_FRONT_WIDE"

    def test_signal_cameras_never_win_through_the_name_fallback_either(self, tmp_path):
        # The prd_jt_val scenes that store only roof channels used to come back
        # as CAM_TOP_LEFT_CENTER: excluded by role in the geometric pass, handed
        # back by the fallback.  A trajectory over a signal-head view is worse
        # than a refusal.
        scene = _display_scene_dir(tmp_path)
        for name in ("CAM_FRONT", "CAM_FRONT_WIDE"):
            shutil.rmtree(scene / "data" / name)
        with pytest.raises(ValueError, match="only non-road channels"):
            front_camera_for_scene(scene)

    def test_falls_back_to_names_without_calibration(self, tmp_path):
        (tmp_path / "data" / "CAM_FRONT_WIDE").mkdir(parents=True)
        assert front_camera_for_scene(tmp_path) == "CAM_FRONT_WIDE"

    def test_no_stored_camera_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="no camera"):
            front_camera_for_scene(tmp_path)


class TestSceneCameraReader:
    def test_reads_a_calibrated_native_resolution_view(self, tmp_path):
        reader = SceneCameraReader(_display_scene_dir(tmp_path), "CAM_FRONT")
        view = reader.read(7)
        assert view.name == "CAM_FRONT"
        assert view.image is not None and view.image.shape == (48, 64, 3)
        assert view.is_calibrated
        # The optical axis must be ego-forward, as written by the fixture.
        assert view.camera2ego_rotation[:, 2] == pytest.approx([1.0, 0.0, 0.0])

    def test_missing_frame_keeps_the_calibration(self, tmp_path):
        reader = SceneCameraReader(_display_scene_dir(tmp_path), "CAM_FRONT")
        view = reader.read(999)
        assert view.image is None and view.is_calibrated

    def test_uncalibrated_camera_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="not calibrated"):
            SceneCameraReader(_display_scene_dir(tmp_path), "CAM_BACK")


def _forward_mounted_camera() -> Camera:
    """A camera 5.35 m ahead of the ego origin and 2.78 m up, as x2_dev mounts it.

    The whole first half of a 4 s plan is then behind the pinhole, and what is
    in front projects below the frame.  No synthetic case covered this before,
    which is how a panel that silently showed nothing got shipped.
    """
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    image = np.broadcast_to(gradient[None, :, None], (48, 64, 3)).copy()
    return Camera(
        name="CAM_FRONT_WIDE",
        image=image,
        camera2ego_rotation=rotation,
        camera2ego_translation=np.array([5.35, 0.0, 2.78]),
        intrinsics=np.array([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1]]),
    )


def _slow_plan() -> Trajectory:
    """A 4 s plan of a vehicle crawling: eight poses out to 8 m ahead of ego.

    Speed is the other half of the geometry.  A camera mounted 5.35 m ahead of
    the ego origin sees a fast plan (which reaches far past the pinhole) and not
    a slow one, so a synthetic case has to be slow to reproduce what the fleet
    actually shows at a junction.
    """
    poses = np.column_stack([np.linspace(1.0, 8.0, 8), np.zeros(8), np.zeros(8)])
    return Trajectory(
        poses=poses.astype(np.float32),
        trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=0.5),
    )


class TestOutOfViewPlans:
    """A plan the view does not contain must say so, not read as no plan."""

    def test_a_forward_mounted_camera_sees_none_of_a_slow_plan(self):
        from t4_e2e_devkit.visualization.planning_video import _points_in_panel

        poses = _slow_plan().poses
        shape = (48, 64, 3)
        # Same intrinsics, same plan: only the mounting moved.
        assert _points_in_panel(_camera(), poses, 1.0, shape) > 0
        assert _points_in_panel(_forward_mounted_camera(), poses, 1.0, shape) == 0

    def test_the_panel_is_annotated_rather_than_left_blank(self):
        scene = _scene()
        scene.current_frame.cameras = Cameras({"CAM_FRONT_WIDE": _forward_mounted_camera()})
        frame = render_planning_frame(scene, {"model": _slow_plan()}, panel_height=48)
        panel = frame[:, frame.shape[0] :]
        # Nothing of the plan projects into this panel, so without the note the
        # panel would be the bare image and a reader would see a model that
        # planned nothing.
        assert not np.array_equal(panel, np.asarray(_forward_mounted_camera().image))

    def test_nothing_crashes_when_every_pose_is_behind_the_camera(self):
        scene = _scene()
        scene.current_frame.cameras = Cameras({"CAM_FRONT_WIDE": _forward_mounted_camera()})
        image = render_planning_frame(scene, None, panel_height=48)
        assert image.ndim == 3 and image.dtype == np.uint8


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

    def test_lidar_points_land_on_the_dark_bev(self):
        image = render_planning_frame(_scene(with_lidar=True), panel_height=700)
        # The BEV is the left 700 px square on black; a sweep point at
        # (x=10, y=5) maps to column (35-5)/70*699 and row (55-10)/70*699.
        assert tuple(image[-1, 0]) == (0, 0, 0)
        assert tuple(image[449, 299]) == BEV_POINT_COLOR

    def test_without_lidar_the_bev_stays_dark_but_renders(self):
        # Graceful degradation: no sweep means no points, not a crash.
        image = render_planning_frame(_scene(with_lidar=False), panel_height=700)
        assert tuple(image[449, 299]) == (0, 0, 0)

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


class TestGroundTruthGrid:
    """The white line is drawn on the grid the plans use, not the default."""

    def test_a_longer_manifest_horizon_extends_the_recorded_future(self, tmp_path):
        from t4_e2e_devkit.visualization.planning_video import _longest_sampling

        long_plan = Trajectory(
            poses=np.column_stack([np.linspace(2.5, 30.0, 12), np.zeros(12), np.zeros(12)]).astype(
                np.float32
            ),
            trajectory_sampling=TrajectorySampling(num_poses=12, interval_length=0.5),
        )
        assert _longest_sampling([("model", long_plan)]).num_poses == 12
        # Drawn on the plan's 6 s grid, the two lines end at the same instant; on
        # the 4 s default the model would look like it overshot by 10 m.
        scene = _scene()
        frame = render_planning_frame(scene, {"model": long_plan}, panel_height=48)
        assert frame.ndim == 3

    def test_no_plans_keeps_the_contract_default(self):
        from t4_e2e_devkit.visualization.planning_video import _longest_sampling

        assert _longest_sampling([]) is None
        assert _longest_sampling([("model", None)]) is None


class TestWindowShapeFollowsTheList:
    def test_the_list_manifest_supplies_the_window_shape(self):
        from t4_e2e_devkit.common.dataclasses import SceneFilter
        from t4_e2e_devkit.dataset.datalist import DataList
        from t4_e2e_devkit.visualization.planning_video import data_list_scene_filter

        data_list = DataList(
            root=Path("/nowhere"),
            rows=[(SCENE_DIR, 40)],
            manifest={"history_frames": 21, "gt_future_frames": 60, "center_stride": 2},
        )
        scene_filter = data_list_scene_filter(data_list, SceneFilter)
        assert scene_filter.num_history_frames == 21
        assert scene_filter.num_future_frames == 60
        assert scene_filter.frame_interval == 2

    def test_an_unrecorded_shape_falls_back_to_the_contract(self):
        from t4_e2e_devkit.common.dataclasses import SceneFilter
        from t4_e2e_devkit.dataset.datalist import DataList
        from t4_e2e_devkit.visualization.planning_video import data_list_scene_filter

        default = SceneFilter()
        resolved = data_list_scene_filter(
            DataList(root=Path("/nowhere"), rows=[(SCENE_DIR, 40)]), SceneFilter
        )
        assert resolved.num_history_frames == default.num_history_frames
        assert resolved.num_future_frames == default.num_future_frames


class TestManifestProvenance:
    def test_a_manifest_from_another_list_warns(self, tmp_path, caplog):
        from t4_e2e_devkit.dataset.datalist import DataList
        from t4_e2e_devkit.visualization.planning_video import warn_on_data_list_mismatch

        listed = DataList(root=tmp_path, rows=[(SCENE_DIR, 10)])
        path = listed.write(tmp_path / "rendered.datalist.json")
        other = DataList(root=tmp_path, rows=[(SCENE_DIR, 10), (SCENE_DIR, 11)])
        other_path = other.write(tmp_path / "other.datalist.json")
        manifest = _manifest(tmp_path, data_list=other_path)

        rendered = DataList(root=tmp_path, rows=listed.rows, path=path)
        with caplog.at_level("WARNING"):
            warn_on_data_list_mismatch({"model": manifest}, rendered)
        assert "different data list" in caplog.text

    def test_the_matching_list_is_silent(self, tmp_path, caplog):
        from t4_e2e_devkit.dataset.datalist import DataList
        from t4_e2e_devkit.visualization.planning_video import warn_on_data_list_mismatch

        listed = DataList(root=tmp_path, rows=[(SCENE_DIR, 10)])
        path = listed.write(tmp_path / "rendered.datalist.json")
        manifest = _manifest(tmp_path, data_list=path)
        with caplog.at_level("WARNING"):
            warn_on_data_list_mismatch(
                {"model": manifest}, DataList(root=tmp_path, rows=listed.rows, path=path)
            )
        assert caplog.text == ""


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

    def test_a_dropped_camera_frame_does_not_end_the_video(self, tmp_path):
        """One missing frame must cost one panel, not the whole render.

        libx264 refuses a frame whose size changed, so a placeholder panel at a
        different aspect than the decoded ones aborts everything after the first
        gap -- and camera gaps are ordinary in the exports.
        """
        scenes = [_scene(center=center) for center in (10, 11, 12)]
        scenes[1].current_frame.cameras["CAM_FRONT_WIDE"].image = None
        out = render_planning_video(scenes, tmp_path / "dropped.mp4", fps=5.0)
        assert out.is_file() and out.stat().st_size > 0

    def test_the_placeholder_panel_matches_the_decoded_panels(self):
        # Same width, so the frames concatenate to one size.  The fixture camera
        # is 64x48, which is not the 16:9 the old placeholder assumed.
        scene = _scene()
        with_image = render_planning_frame(scene, panel_height=48)
        scene.current_frame.cameras["CAM_FRONT_WIDE"].image = None
        without = render_planning_frame(scene, panel_height=48, camera_size=(64, 48))
        assert without.shape == with_image.shape
