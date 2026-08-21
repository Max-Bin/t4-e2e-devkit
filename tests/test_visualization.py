"""Visualization tests.

The geometric ones matter most. A plot that is merely *wrong* still renders, so
"it produced a figure" proves almost nothing -- these check the projection and the
map reconstruction against facts measured from the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.common.dataclasses import Camera
from t4_e2e_devkit.common.enums import T4BoxIndex
from t4_e2e_devkit.visualization import (
    boundaries_from_segments,
    box_corners_3d,
    camera_grid_layout,
    centerlines_from_segments,
    figure_to_rgb,
    project_with_distortion,
    render_prediction_bev,
    traffic_light_states,
)
from t4_e2e_devkit.visualization.lidar import (
    filter_lidar_pc,
    get_lidar_pc_color,
    subsample_lidar_pc,
)


class TestBoxCorners:
    """3D corners from a T4 box row."""

    def test_axis_aligned_box(self):
        # length 4, width 2, height 1.5, centred at the origin, heading 0.
        box = np.array([0, 0, 0, 2.0, 4.0, 1.5, 0.0, 0, 0], dtype=np.float64)
        corners = box_corners_3d(box)
        assert corners.shape == (8, 3)
        assert corners[:, 0].max() == pytest.approx(2.0)  # half length
        assert corners[:, 1].max() == pytest.approx(1.0)  # half width
        assert corners[:, 2].max() == pytest.approx(0.75)  # half height

    def test_width_and_length_are_not_swapped(self):
        # The whole reason T4BoxIndex exists: column 3 is width, column 4 length.
        # Read the other way round, every vehicle renders rotated 90 degrees.
        box = np.zeros(9)
        box[T4BoxIndex.WIDTH] = 2.0
        box[T4BoxIndex.LENGTH] = 5.0
        box[T4BoxIndex.HEIGHT] = 1.0
        corners = box_corners_3d(box)
        longitudinal = corners[:, 0].max() - corners[:, 0].min()
        lateral = corners[:, 1].max() - corners[:, 1].min()
        assert longitudinal == pytest.approx(5.0)
        assert lateral == pytest.approx(2.0)

    def test_heading_rotates_the_footprint(self):
        box = np.array([0, 0, 0, 2.0, 4.0, 1.5, np.pi / 2, 0, 0], dtype=np.float64)
        corners = box_corners_3d(box)
        # Rotated 90 degrees: the long axis now lies along y.
        assert corners[:, 1].max() == pytest.approx(2.0)
        assert corners[:, 0].max() == pytest.approx(1.0)

    def test_z_is_the_centre(self):
        box = np.zeros(9)
        box[T4BoxIndex.Z] = 1.0
        box[T4BoxIndex.HEIGHT] = 2.0
        box[T4BoxIndex.WIDTH] = box[T4BoxIndex.LENGTH] = 1.0
        corners = box_corners_3d(box)
        assert corners[:, 2].min() == pytest.approx(0.0)
        assert corners[:, 2].max() == pytest.approx(2.0)


class TestProjection:
    """Pinhole and the OpenCV distortion model."""

    @staticmethod
    def _intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def test_principal_axis_maps_to_principal_point(self):
        pixels = project_with_distortion(np.array([[0.0, 0.0, 10.0]]), self._intrinsics())
        np.testing.assert_allclose(pixels[0], [320.0, 240.0])

    def test_pinhole_scales_with_inverse_depth(self):
        matrix = self._intrinsics()
        near = project_with_distortion(np.array([[1.0, 0.0, 5.0]]), matrix)
        far = project_with_distortion(np.array([[1.0, 0.0, 10.0]]), matrix)
        # Twice the depth, half the offset from the principal point.
        assert (near[0, 0] - 320.0) == pytest.approx(2 * (far[0, 0] - 320.0))

    def test_zero_distortion_equals_pinhole(self):
        matrix = self._intrinsics()
        points = np.array([[1.0, 2.0, 7.0], [-3.0, 0.5, 12.0]])
        plain = project_with_distortion(points, matrix)
        zeroed = project_with_distortion(points, matrix, np.zeros(5))
        np.testing.assert_allclose(plain, zeroed)

    def test_distortion_is_applied_to_normalised_coordinates(self):
        # The model is defined on (X/Z, Y/Z). Applying it to raw (X, Y) instead --
        # which is what t4-devkit's view_points does -- agrees only at Z == 1, so
        # a point at another depth is what distinguishes the two.
        matrix = self._intrinsics()
        coefficients = np.array([0.1, 0.01, 0.0, 0.0, 0.001])
        at_unit_depth = np.array([[0.5, 0.5, 1.0]])
        scaled = np.array([[5.0, 5.0, 10.0]])  # same normalised coordinates
        np.testing.assert_allclose(
            project_with_distortion(at_unit_depth, matrix, coefficients),
            project_with_distortion(scaled, matrix, coefficients),
            atol=1e-9,
        )

    def test_distortion_moves_off_axis_points_only(self):
        matrix = self._intrinsics()
        coefficients = np.array([0.2, 0.0, 0.0, 0.0, 0.0])
        on_axis = np.array([[0.0, 0.0, 5.0]])
        np.testing.assert_allclose(
            project_with_distortion(on_axis, matrix, coefficients),
            project_with_distortion(on_axis, matrix),
        )
        off_axis = np.array([[1.0, 0.0, 5.0]])
        assert not np.allclose(
            project_with_distortion(off_axis, matrix, coefficients),
            project_with_distortion(off_axis, matrix),
        )

    def test_accepts_every_opencv_coefficient_length(self):
        matrix = self._intrinsics()
        points = np.array([[1.0, 1.0, 6.0]])
        for length in (4, 5, 8, 12, 14):
            pixels = project_with_distortion(points, matrix, np.zeros(length))
            np.testing.assert_allclose(pixels, project_with_distortion(points, matrix))


class TestCameraTransforms:
    """The camera-to-ego convention, on a synthetic rig with known geometry."""

    @staticmethod
    def _camera():
        # Camera x -> ego -y, camera y -> ego -z, camera z -> ego +x. This is the
        # mapping measured on real T4 scenes.
        rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        return Camera(
            name="CAM_FRONT_WIDE",
            image=np.zeros((672, 1148, 3), dtype=np.uint8),
            camera2ego_rotation=rotation,
            camera2ego_translation=np.array([1.3, 0.0, 1.9]),
            intrinsics=np.array([[500.0, 0, 574.0], [0, 500.0, 336.0], [0, 0, 1]]),
        )

    def test_point_ahead_is_in_front(self):
        camera = self._camera()
        camera_points = camera.ego_to_camera(np.array([[11.3, 0.0, 1.9]]))
        # 10 m ahead of the camera, on its optical axis.
        np.testing.assert_allclose(camera_points[0], [0.0, 0.0, 10.0], atol=1e-9)

    def test_point_behind_is_rejected(self):
        camera = self._camera()
        _, valid = camera.project_to_image(np.array([[-5.0, 0.0, 1.9]]))
        assert not valid[0]

    def test_ground_projects_below_the_horizon(self):
        camera = self._camera()
        pixels, valid = camera.project_to_image(np.array([[21.3, 0.0, 0.0]]))
        assert valid[0]
        # A point on the road 20 m ahead must appear below the principal point.
        assert pixels[0, 1] > camera.intrinsics[1, 2]

    def test_left_of_ego_projects_left_of_centre(self):
        camera = self._camera()
        pixels, valid = camera.project_to_image(np.array([[11.3, 5.0, 1.9]]))
        assert valid[0]
        assert pixels[0, 0] < camera.intrinsics[0, 2]

    def test_uncalibrated_camera_refuses_to_project(self):
        camera = Camera(name="CAM_FRONT")
        assert not camera.is_calibrated
        with pytest.raises(ValueError, match="no calibration"):
            camera.ego_to_camera(np.zeros((1, 3)))


class TestMapReconstruction:
    """Lane geometry from the segment tensor."""

    @staticmethod
    def _segments(num_lanes=3, points=5, valid_lanes=2):
        segments = np.zeros((num_lanes, points, C.SEGMENT_POINT_DIM), dtype=np.float32)
        for lane in range(valid_lanes):
            segments[lane, :, C.X] = np.arange(points) + 1.0
            segments[lane, :, C.Y] = float(lane) * 3.0 + 1.0
            segments[lane, :, C.LB_Y] = 1.5  # left boundary offset
            segments[lane, :, C.RB_Y] = -1.5  # right boundary offset
        return segments

    def test_padding_rows_are_dropped(self):
        polylines = centerlines_from_segments(self._segments(valid_lanes=2))
        assert len(polylines) == 2

    def test_boundaries_are_offsets_from_the_centreline(self):
        # Measured on real data: |LB| = |RB| = 1.497 m, opposite signs in 100% of
        # rows, implied lane width median exactly 3.00 m. Read as absolute
        # coordinates instead, every lane collapses to a blob at the origin.
        segments = self._segments(valid_lanes=1)
        centres = centerlines_from_segments(segments)[0]
        left, right = boundaries_from_segments(segments)
        np.testing.assert_allclose(left[0][:, 1], centres[:, 1] + 1.5)
        np.testing.assert_allclose(right[0][:, 1], centres[:, 1] - 1.5)
        width = np.linalg.norm(left[0] - right[0], axis=-1)
        np.testing.assert_allclose(width, 3.0)

    def test_traffic_light_state_is_read_from_the_one_hot(self):
        segments = self._segments(valid_lanes=1)
        segments[0, :, C.TRAFFIC_LIGHT_RED] = 1.0
        assert traffic_light_states(segments) == ["red"]
        segments[0, :, C.TRAFFIC_LIGHT_RED] = 0.0
        segments[0, :, C.TRAFFIC_LIGHT_GREEN] = 1.0
        assert traffic_light_states(segments) == ["green"]

    def test_states_align_with_centrelines(self):
        segments = self._segments(num_lanes=4, valid_lanes=3)
        assert len(traffic_light_states(segments)) == len(centerlines_from_segments(segments))

    def test_rejects_wrong_tensor_rank(self):
        with pytest.raises(ValueError, match=r"\[N, P, >=8\]"):
            centerlines_from_segments(np.zeros((5, 8)))


class TestLidarHelpers:
    def test_filter_crops_to_the_view_box(self):
        points = np.array(
            [[0.0, 0.0, 0.0, 1, 0], [500.0, 0.0, 0.0, 1, 0], [0.0, 0.0, 99.0, 1, 0]],
            dtype=np.float32,
        )
        assert len(filter_lidar_pc(points)) == 1

    def test_subsample_reports_what_it_dropped(self):
        points = np.zeros((1000, 5), dtype=np.float32)
        kept, dropped = subsample_lidar_pc(points, max_points=100)
        assert len(kept) == 100
        assert dropped == 900

    def test_subsample_is_deterministic(self):
        points = np.random.default_rng(0).random((500, 5)).astype(np.float32)
        first, _ = subsample_lidar_pc(points, 50, seed=7)
        second, _ = subsample_lidar_pc(points, 50, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_id_colouring_is_refused(self):
        # NAVSIM clouds carry a per-point instance id and T4's do not; accepting
        # "id" and colouring by zeros would render a uniform cloud that looks like
        # a successful plot.
        with pytest.raises(ValueError, match="valid choices"):
            get_lidar_pc_color(np.zeros((4, 5), np.float32), {"color_element": "id"})

    def test_colour_per_point(self):
        points = np.random.default_rng(0).random((32, 5)).astype(np.float32)
        assert len(get_lidar_pc_color(points)) == 32


class TestCameraGridLayout:
    """The layout has to follow the rig, since the rigs genuinely differ."""

    def test_wide_five_rig_has_no_centre_rear_cell(self):
        grid = camera_grid_layout(
            [
                "CAM_FRONT_WIDE",
                "CAM_FRONT_LEFT_WIDE",
                "CAM_FRONT_RIGHT_WIDE",
                "CAM_BACK_LEFT_WIDE",
                "CAM_BACK_RIGHT_WIDE",
            ]
        )
        assert grid[0] == ["CAM_FRONT_LEFT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT_WIDE"]
        assert grid[1] == ["CAM_BACK_LEFT_WIDE", None, "CAM_BACK_RIGHT_WIDE"]

    def test_x2_dev_rig_fills_the_centre_rear_cell(self):
        # x2_dev has a real CAM_BACK; prd_jt does not. A fixed layout gets one of
        # the two wrong.
        grid = camera_grid_layout(
            [
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
        )
        assert grid[0] == ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        assert grid[1] == ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

    def test_non_surround_views_go_on_their_own_row(self):
        grid = camera_grid_layout(["CAM_FRONT", "CAM_TRAFFIC_LIGHT_FAR"])
        assert grid[0] == [None, "CAM_FRONT", None]
        assert "CAM_TRAFFIC_LIGHT_FAR" in grid[-1]

    def test_every_camera_is_placed(self):
        names = ["CAM_FRONT", "CAM_BACK", "CAM_TOP_LEFT_CENTER", "CAM_TRAFFIC_LIGHT_NEAR"]
        placed = {cell for row in camera_grid_layout(names) for cell in row if cell}
        assert placed == set(names)


@pytest.mark.data
class TestRenderingRealScenes:
    """The plots run end to end on real windows and produce non-trivial images."""

    def test_bev_renders(self, scene):
        from t4_e2e_devkit.visualization import plot_bev_frame

        figure, _ = plot_bev_frame(scene, {"ground_truth": scene.get_future_trajectory()})
        image = figure_to_rgb(figure)
        assert image.ndim == 3 and image.shape[2] == 3
        # A blank canvas would be uniform; anything drawn gives it variance.
        assert image.std() > 5.0

    def test_score_panel_renders(self, scene):
        from t4_e2e_devkit.visualization import plot_bev_with_score

        results = {
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
            "ego_progress": 0.6,
            "time_to_collision_within_bound": 0.8,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
            "score": 0.8,
        }
        figure, _ = plot_bev_with_score(scene, scene.get_future_trajectory(), results)
        assert figure_to_rgb(figure).std() > 5.0

    def test_cameras_render_with_overlays(self, t4_scene_dir, t4_root):
        from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene
        from t4_e2e_devkit.dataset.window import T4WindowBuilder
        from t4_e2e_devkit.visualization import plot_cameras_frame

        builder = T4WindowBuilder(
            t4_scene_dir,
            t4_root,
            sensor_config=sensor_config_for_scene(t4_scene_dir, lidar=True),
        )
        try:
            centers = builder.valid_centers()
            window = builder.build(centers[len(centers) // 2])
            figure, axes = plot_cameras_frame(window, with_annotations=True, with_lidar=True)
            assert figure_to_rgb(figure).std() > 5.0
        finally:
            builder.close()

    def test_cameras_without_sensors_is_an_error(self, scene):
        from t4_e2e_devkit.visualization import plot_cameras_frame

        with pytest.raises(ValueError, match="no decoded cameras"):
            plot_cameras_frame(scene)


class TestTrajectoryKinds:
    """Every kind the palette declares must be producible, or not offered."""

    def test_config_kinds_are_all_reachable(self):
        from t4_e2e_devkit.visualization.config import TRAJECTORY_CONFIG

        # `prediction` comes from an agent; the other two come from the window.
        # A declared kind nothing can produce is dead config -- it promises a
        # legend entry that never appears.
        assert set(TRAJECTORY_CONFIG) == {
            "prediction",
            "ground_truth",
            "history",
        }

    def test_fixed_bev_legend_has_stable_t4_vocabulary(self):
        import matplotlib.pyplot as plt

        from t4_e2e_devkit.visualization import add_fixed_bev_legend

        figure, ax = plt.subplots()
        try:
            legend = add_fixed_bev_legend(
                ax,
                trajectory_roles=("history", "prediction"),
            )
            assert [text.get_text() for text in legend.get_texts()] == [
                "history",
                "prediction",
                "ego",
                "car",
                "truck",
                "bus",
                "bicycle",
                "pedestrian",
                "Goal Pose",
            ]
        finally:
            plt.close(figure)


@pytest.mark.data
class TestWindowSuppliedTrajectories:
    def test_history_ends_at_the_origin(self, scene):
        history = scene.get_history_poses()
        assert history.shape == (C.PAST_FRAMES, 3)
        np.testing.assert_allclose(history[-1], 0.0, atol=1e-6)

    def test_history_is_ordered_oldest_first(self, scene):
        history = scene.get_history_poses()
        # The ego approaches the origin, so distance from it should not increase.
        distances = np.linalg.norm(history[:, :2], axis=-1)
        assert distances[0] >= distances[-1]

    def test_reference_trajectories_skips_what_is_missing(self, scene):
        from t4_e2e_devkit.visualization import reference_trajectories

        trajectories = reference_trajectories(scene)
        assert "history" in trajectories
        assert "ground_truth" in trajectories
        assert set(trajectories) == {"history", "ground_truth"}

    def test_history_is_not_given_a_second_origin(self, scene):
        from t4_e2e_devkit.visualization import plot_bev_frame

        # plot_bev_frame passes include_origin=False for history, because the
        # history already ends at the origin; prepending one would draw a spur
        # back to a point the path already contains.
        figure, ax = plot_bev_frame(scene, {"history": scene.get_history_poses()})
        history_lines = [line for line in ax.get_lines() if line.get_label() == "history"]
        assert history_lines
        assert len(history_lines[0].get_xdata()) == C.PAST_FRAMES


class TestVizCallbackIsNonFatal:
    """A plotting bug must never end a long training run."""

    def test_failure_is_swallowed_and_eventually_disables(self, tmp_path):
        from t4_e2e_devkit.planning.training.callbacks import TrajectoryVizCallback

        callback = TrajectoryVizCallback(data_list=tmp_path / "does_not_exist.json", max_failures=2)

        class _Trainer:
            is_global_zero = True
            sanity_checking = False
            current_epoch = 0
            global_step = 0
            logger = None

        trainer = _Trainer()
        import torch

        module = torch.nn.Linear(1, 1)

        # Losing a twelve-hour run to a missing file is strictly worse than
        # losing the plot, so this returns rather than raises.
        callback.on_validation_epoch_end(trainer, module)
        assert callback._failures == 1
        trainer.current_epoch = 1
        callback.on_validation_epoch_end(trainer, module)
        assert callback._disabled is True

    def test_skips_sanity_check_and_non_zero_ranks(self, tmp_path):
        from t4_e2e_devkit.planning.training.callbacks import TrajectoryVizCallback

        callback = TrajectoryVizCallback(data_list=tmp_path / "x.json")

        class _Trainer:
            is_global_zero = False
            sanity_checking = False
            current_epoch = 0
            global_step = 0
            logger = None

        import torch

        callback.on_validation_epoch_end(_Trainer(), torch.nn.Linear(1, 1))
        assert callback._failures == 0


class TestPredictionVisualization:
    """The model-repository visualization boundary uses the shared renderer."""

    @staticmethod
    def _sample():
        lanes = np.zeros(
            (C.NUM_SEGMENTS_IN_LANE, C.POINTS_PER_LANELET, C.SEGMENT_POINT_DIM), np.float32
        )
        route = np.zeros(
            (C.NUM_SEGMENTS_IN_ROUTE, C.POINTS_PER_LANELET, C.SEGMENT_POINT_DIM), np.float32
        )
        lanes[0, :, 0] = np.linspace(0.0, 40.0, C.POINTS_PER_LANELET)
        route[0, :, 0] = np.linspace(0.0, 40.0, C.POINTS_PER_LANELET)
        route[0, :, 1] = 2.0
        gt = np.column_stack([np.linspace(0.5, 40.0, 80), np.zeros(80)]).astype(np.float32)
        pred = gt.copy()
        pred[:, 1] = 0.5
        return gt, pred, lanes, route

    def test_render_prediction_bev_returns_rgb(self):
        gt, pred, lanes, route = self._sample()
        image = render_prediction_bev(gt, pred, lanes=lanes, route=route)
        assert image.ndim == 3
        assert image.shape[-1] == 3
        assert image.dtype == np.uint8
        assert image.max() > image.min()

    def test_render_prediction_bev_accepts_geometry_only_maps(self):
        gt, pred, lanes, route = self._sample()
        image = render_prediction_bev(
            gt,
            pred,
            lanes=lanes[..., :2],
            route=route[..., :2],
        )
        assert image.ndim == 3
        assert image.shape[-1] == 3
        assert image.dtype == np.uint8
        assert image.max() > image.min()

    def test_prediction_callback_logs_model_owned_samples(self):
        from types import SimpleNamespace

        from t4_e2e_devkit.planning.training import PredictionVizCallback

        gt, pred, lanes, route = self._sample()

        class _Logger:
            def __init__(self):
                self.calls = []

            def log_image(self, **kwargs):
                self.calls.append(kwargs)

        logger = _Logger()
        trainer = SimpleNamespace(
            is_global_zero=True,
            sanity_checking=False,
            current_epoch=0,
            loggers=[logger],
        )
        module = SimpleNamespace()
        callback = PredictionVizCallback(n_samples=1)

        callback.on_validation_epoch_start(trainer, module)
        assert module._viz_capacity == 1
        module._viz_samples = [{"gt_xy": gt, "pred_xy": pred, "lanes": lanes, "route": route}]
        callback.on_validation_epoch_end(trainer, module)

        assert len(logger.calls) == 1
        assert logger.calls[0]["key"] == "val/bev_trajectory"
        assert len(logger.calls[0]["images"]) == 1
        assert module._viz_samples == []
