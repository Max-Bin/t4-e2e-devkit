"""Dataset tests, against real T4 scenes.

Synthetic arrays cannot catch the failures that matter here -- a field read from
the wrong bundle key, intrinsics left at native resolution, a future window that
is one frame short.  Every test in this file therefore reads the dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.common.dataclasses import SceneFilter, SensorConfig
from t4_e2e_devkit.dataset.contract import BUNDLE_TO_CONTRACT, CONTRACT_MAP_FIELDS
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.dataset.window import T4WindowBuilder, WindowError

pytestmark = pytest.mark.data


class TestWindowGeometry:
    """Window bounds and the ego-status derivation."""

    def test_window_has_contract_length(self, scene):
        assert len(scene.frames) == C.PAST_FRAMES
        assert scene.current_frame_index == C.PAST_FRAMES - 1
        assert scene.future_ego_poses.shape == (C.FUTURE_FRAMES, 3)

    def test_current_frame_is_the_origin(self, scene):
        # Everything is expressed in the current frame, so the current pose is
        # the origin by construction.  A non-zero value here means the window
        # was built around a different centre than it reports.
        np.testing.assert_allclose(scene.current_frame.ego_status.ego_pose, 0.0, atol=1e-6)

    def test_history_is_ordered_oldest_to_current(self, scene):
        indices = [frame.frame_index for frame in scene.frames]
        assert indices == sorted(indices)
        assert indices[-1] == scene.scene_metadata.center_frame

    def test_ego_status_start_is_not_fabricated_as_stationary(self, scene):
        # The oldest rows have no predecessor to difference against and repeat
        # the nearest well-defined row.  Reading zero there would claim the ego
        # was standing still at the start of every window.
        speeds = [frame.ego_status.speed for frame in scene.frames]
        if max(speeds) > 1.0:  # only meaningful on a moving window
            assert speeds[0] == pytest.approx(speeds[1], rel=1e-6)

    def test_out_of_range_centre_raises(self, window_builder):
        with pytest.raises(WindowError, match="outside the scene"):
            window_builder.build(0)

    def test_valid_centers_all_build(self, window_builder):
        centers = window_builder.valid_centers()
        for center in (centers[0], centers[len(centers) // 2], centers[-1]):
            window_builder.build(center)


class TestTrajectoryHorizon:
    """The recorded trajectory must span the scorer horizon, not the GT window."""

    def test_stride_comes_from_the_contract(self, scene):
        trajectory = scene.get_future_trajectory()
        assert trajectory.poses.shape == (C.TRAJECTORY_POSES, 3)
        assert trajectory.trajectory_sampling.interval_length == pytest.approx(
            C.TRAJECTORY_INTERVAL
        )
        # 8 poses at 0.5 s = 4 s, NOT 8 poses spread over the 80-frame window.
        assert trajectory.trajectory_sampling.time_horizon == pytest.approx(4.0)

    def test_poses_match_the_dense_future_at_stride(self, scene):
        trajectory = scene.get_future_trajectory()
        expected = scene.future_ego_poses[C.FUTURE_STRIDE - 1 :: C.FUTURE_STRIDE][
            : C.TRAJECTORY_POSES
        ]
        np.testing.assert_allclose(trajectory.poses, expected, atol=1e-6)

    def test_too_few_future_frames_raises(self, t4_scene_dir, t4_root):
        builder = T4WindowBuilder(
            t4_scene_dir, t4_root, scene_filter=SceneFilter(num_future_frames=10)
        )
        try:
            centers = builder.valid_centers()
            scene = builder.build(centers[len(centers) // 2])
            # 10 frames is 1.0 s; the default contract asks for 8 poses at 0.5 s.
            # Assert on the counts rather than on a wording: the refusal must say
            # what was asked for and what the scene has, because a short window
            # must never be padded or extrapolated into a full one, and "too few"
            # alone does not tell the caller which end to fix.
            with pytest.raises(
                ValueError,
                match=rf"cannot provide {C.TRAJECTORY_POSES} poses .*from 10 future frames",
            ):
                scene.get_future_trajectory()
        finally:
            builder.close()


class TestMapFields:
    """Map naming and shapes, and the refusal to zero-fill."""

    def test_every_contract_field_is_present(self, scene):
        fields = scene.current_frame.map_tensors.as_dict()
        assert set(fields) == set(CONTRACT_MAP_FIELDS)

    def test_shapes_match_the_contract(self, scene):
        m = scene.current_frame.map_tensors
        assert m.lanes.shape == (C.NUM_SEGMENTS_IN_LANE, C.POINTS_PER_LANELET, C.SEGMENT_POINT_DIM)
        assert m.route_lanes.shape == (
            C.NUM_SEGMENTS_IN_ROUTE, C.POINTS_PER_LANELET, C.SEGMENT_POINT_DIM,
        )
        assert m.polygons.shape == (C.NUM_POLYGONS, C.POINTS_PER_POLYGON, 3)
        assert m.line_strings.shape == (C.NUM_LINE_STRINGS, C.POINTS_PER_LINE_STRING, 4)

    def test_bundle_names_are_renamed_not_dropped(self):
        # The disk vocabulary and the model vocabulary differ; the mapping is
        # the only place they are related.
        assert BUNDLE_TO_CONTRACT["route"] == "route_lanes"
        assert BUNDLE_TO_CONTRACT["lines"] == "line_strings"
        assert BUNDLE_TO_CONTRACT["lanes_speed"] == "lanes_speed_limit"
        assert len(BUNDLE_TO_CONTRACT) == len(CONTRACT_MAP_FIELDS)


class TestSensorLaziness:
    """A model must not pay for sensors it did not ask for."""

    def test_no_sensors_decodes_nothing(self, scene):
        assert all(frame.cameras is None for frame in scene.frames)
        assert all(frame.lidar is None for frame in scene.frames)

    def test_current_frame_only_decodes_one_step(self, t4_scene_dir, t4_root):
        builder = T4WindowBuilder(
            t4_scene_dir, t4_root, sensor_config=SensorConfig.build_current_frame(lidar=True)
        )
        try:
            centers = builder.valid_centers()
            scene = builder.build(centers[len(centers) // 2])
            with_cameras = [frame for frame in scene.frames if frame.cameras is not None]
            assert len(with_cameras) == 1
            assert with_cameras[0].frame_index == scene.scene_metadata.center_frame
            assert sum(frame.lidar is not None for frame in scene.frames) == 1
        finally:
            builder.close()

    def test_camera_images_are_raw_uint8_at_reader_resolution(self, t4_scene_dir, t4_root):
        builder = T4WindowBuilder(
            t4_scene_dir, t4_root, sensor_config=SensorConfig.build_current_frame()
        )
        try:
            centers = builder.valid_centers()
            cameras = builder.build(centers[len(centers) // 2]).current_frame.cameras
            present = [camera for camera in cameras if camera.is_present]
            if not present:
                pytest.skip("fixture scene has no camera images")
            height, width = C.T4_DEFAULT_IMAGE_SIZE_HW
            for camera in present:
                assert camera.image.dtype == np.uint8
                assert camera.image.shape == (height, width, 3)
        finally:
            builder.close()

    def test_intrinsics_are_rescaled_to_reader_resolution(self, t4_scene_dir, t4_root):
        builder = T4WindowBuilder(
            t4_scene_dir, t4_root, sensor_config=SensorConfig.build_current_frame()
        )
        try:
            centers = builder.valid_centers()
            cameras = builder.build(centers[len(centers) // 2]).current_frame.cameras
            height, width = C.T4_DEFAULT_IMAGE_SIZE_HW
            for camera in cameras:
                if not camera.is_present:
                    continue
                # Calibration is stored for the native image.  Left unscaled,
                # the principal point sits outside the resized frame and any
                # camera/BEV projection is wrong by the resize ratio.
                assert 0.2 * width < camera.intrinsics[0, 2] < 0.8 * width
                assert 0.2 * height < camera.intrinsics[1, 2] < 0.8 * height
        finally:
            builder.close()

    def test_lidar_points_have_five_columns(self, t4_scene_dir, t4_root):
        builder = T4WindowBuilder(
            t4_scene_dir, t4_root,
            sensor_config=SensorConfig(cameras={}, lidar=[-1]),
        )
        try:
            centers = builder.valid_centers()
            lidar = builder.build(centers[len(centers) // 2]).current_frame.lidar
            assert lidar.lidar_pc.shape[-1] == C.T4_LIDAR_POINT_DIM
            assert lidar.lidar_pc.shape[0] > 0
        finally:
            builder.close()


class TestAnnotationTransform:
    """Future boxes are expressed in the centre frame, and bridged."""

    def test_future_annotations_cover_the_window(self, scene):
        # One extra frame: index 0 is the current frame, so the list spans
        # [centre, centre + future].
        assert len(scene.future_annotations) == C.FUTURE_FRAMES + 1

    def test_current_frame_annotations_match_future_index_zero(self, scene):
        current = scene.current_frame.annotations
        first_future = scene.future_annotations[0]
        # Both describe the same frame in the same coordinates; the bridge may
        # add objects to the future list but must not move the shared ones.
        assert len(first_future) >= len(current)

    def test_boxes_use_the_t4_nine_column_layout(self, scene):
        assert scene.current_frame.annotations.boxes.shape[1] == 9


class TestAgentInputIsUnprivileged:
    """The privilege boundary is structural, not a convention."""

    def test_agent_input_has_no_future(self, scene):
        agent_input = scene.get_agent_input()
        assert not hasattr(agent_input, "future_ego_poses")
        assert not hasattr(agent_input, "future_annotations")

    def test_agent_input_stops_at_the_current_frame(self, scene):
        agent_input = scene.get_agent_input()
        assert agent_input.num_history_frames == C.PAST_FRAMES
        np.testing.assert_allclose(agent_input.ego_status.ego_pose, 0.0, atol=1e-6)

    def test_streams_stay_index_aligned(self, scene):
        agent_input = scene.get_agent_input()
        n = agent_input.num_history_frames
        assert len(agent_input.cameras) == n
        assert len(agent_input.lidars) == n


class TestDataList:
    """The list is the record of which windows a run saw."""

    def test_round_trip(self, tmp_path, t4_root):
        rows = [("prd_jt/a/b/c", 100), ("prd_jt/a/b/c", 105), ("prd_jt/d/e/f", 50)]
        original = DataList(root=t4_root, rows=rows, manifest={"note": "test"})
        path = original.write(tmp_path / "list.json")
        loaded = load_data_list(path)
        assert loaded.rows == rows
        assert loaded.root == t4_root
        assert loaded.manifest["note"] == "test"
        assert loaded.scene_dirs == ["prd_jt/a/b/c", "prd_jt/d/e/f"]

    def test_annotated_subtree_is_refused(self, tmp_path, t4_root):
        # An annotated path in an E2E list is a real mistake, not a preference:
        # standalone perception has its own GT-bearing boundary.
        DataList(root=t4_root, rows=[("annotated_data/t4dataset/x", 10)]).write(
            tmp_path / "bad.json"
        )
        with pytest.raises(ValueError, match="annotation-free"):
            load_data_list(tmp_path / "bad.json")

    def test_empty_list_is_refused(self, tmp_path, t4_root):
        # Stamped from the constants, not spelled out: a hand-written header goes
        # stale the moment the format or version moves, and then the header guard
        # fires first and this test silently stops covering the empty-rows guard.
        (tmp_path / "empty.json").write_text(
            json.dumps(
                {
                    "format": C.DATA_LIST_FORMAT,
                    "version": C.DATA_LIST_VERSION,
                    "root": "/x",
                    "rows": [],
                }
            )
        )
        with pytest.raises(ValueError, match="no rows"):
            load_data_list(tmp_path / "empty.json")

    def test_runtime_filter_is_recorded(self, t4_root):
        rows = [("prd_jt/a/b/c", index) for index in range(10)]
        narrowed = DataList(root=t4_root, rows=rows).filtered(max_rows=3)
        assert len(narrowed) == 3
        # A subsetted run must still be able to say what it ran on.
        assert narrowed.manifest["runtime_filter"]["rows_before"] == 10
        assert narrowed.manifest["runtime_filter"]["rows_after"] == 3
