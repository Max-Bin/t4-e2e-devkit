"""The contract tests: the numbers and layouts everything else assumes.

Each of these pins a fact that, if it silently changed, would produce plausible
wrong results rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    EgoShape,
    Lidar,
    PDMResults,
    SensorConfig,
    Trajectory,
    aggregate_pdm_score,
)
from t4_e2e_devkit.common.enums import BoundingBoxIndex, EgoStatusIndex, T4BoxIndex
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class TestTemporalContract:
    """The four horizons, which are routinely conflated."""

    def test_history_window(self):
        assert C.PAST_FRAMES == C.INPUT_T + 1 == 31

    def test_future_gt_is_eight_seconds(self):
        assert C.FUTURE_FRAMES == 80
        assert C.FUTURE_FRAMES / C.T4_FRAME_RATE_HZ == 8.0

    def test_scorer_horizon_is_four_seconds(self):
        assert C.SCORER_FUTURE_FRAMES == 40
        assert C.SCORER_FUTURE_FRAMES / C.T4_FRAME_RATE_HZ == 4.0

    def test_model_output_covers_the_scorer_horizon(self):
        # 8 poses at 0.5 s is the same 4 s the scorer reads.  If these ever
        # disagree, a model would be supervised over one horizon and scored
        # over another.
        assert C.TRAJECTORY_POSES * C.TRAJECTORY_INTERVAL == 4.0
        assert (
            C.TRAJECTORY_POSES * C.FUTURE_STRIDE == C.SCORER_FUTURE_FRAMES
        ), "model poses at FUTURE_STRIDE must span exactly the scorer horizon"

    def test_observation_window_exceeds_scoring_horizon(self):
        # TTC projects past the last scored step, so the observation buffer must
        # be longer than the horizon it serves.
        assert C.PDM_OBSERVATION_FRAMES > C.SCORER_FUTURE_FRAMES

    def test_window_needs_history_plus_future(self):
        assert C.MIN_T4_FRAMES == C.PAST_FRAMES + C.FUTURE_FRAMES


class TestMapContract:
    """Vector map shapes and the segment column layout."""

    def test_segment_point_dim(self):
        # 8 geometry + 5 traffic light + 10 + 10 line types
        assert C.SEGMENT_POINT_DIM == 33
        assert C.LINE_TYPE_RIGHT_START + C.LINE_TYPE_NUM == C.SEGMENT_POINT_DIM

    def test_traffic_light_one_hot_is_contiguous(self):
        assert C.TRAFFIC_LIGHT_GREEN == C.TRAFFIC_LIGHT
        assert (
            C.TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT - C.TRAFFIC_LIGHT + 1
            == C.TRAFFIC_LIGHT_ONE_HOT_DIM
        )

    def test_lane_and_route_counts(self):
        assert (C.NUM_SEGMENTS_IN_LANE, C.POINTS_PER_LANELET) == (140, 20)
        assert (C.NUM_SEGMENTS_IN_ROUTE, C.POINTS_PER_LANELET) == (25, 20)
        assert (C.NUM_POLYGONS, C.POINTS_PER_POLYGON) == (10, 40)
        assert (C.NUM_LINE_STRINGS, C.POINTS_PER_LINE_STRING) == (60, 20)


class TestBoxLayout:
    """T4 boxes put width before length; NAVSIM's do not."""

    def test_t4_box_is_nine_columns(self):
        assert T4BoxIndex.size() == 9

    def test_width_precedes_length(self):
        assert T4BoxIndex.WIDTH < T4BoxIndex.LENGTH

    def test_t4_and_navsim_layouts_differ(self):
        # This is the point of having two classes.  Reading a T4 row through
        # NAVSIM's ordering swaps every footprint's axes and changes collision,
        # TTC and drivable-area outcomes without raising.
        assert (T4BoxIndex.WIDTH, T4BoxIndex.LENGTH) != (
            BoundingBoxIndex.WIDTH,
            BoundingBoxIndex.LENGTH,
        )

    def test_velocity_columns(self):
        assert (T4BoxIndex.VELOCITY_X, T4BoxIndex.VELOCITY_Y) == (7, 8)


class TestEgoShape:
    """The footprint is read from the scene, never assumed."""

    def test_round_trip(self):
        shape = EgoShape.from_array(np.array([2.75, 4.34, 1.84]))
        assert shape.wheel_base == pytest.approx(2.75)
        assert shape.length == pytest.approx(4.34)
        assert shape.width == pytest.approx(1.84)
        np.testing.assert_allclose(shape.as_array(), [2.75, 4.34, 1.84])

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="wheel_base, length, width"):
            EgoShape.from_array(np.array([1.0, 2.0]))

    def test_rear_axle_offset_is_within_the_footprint(self):
        shape = EgoShape.from_array(np.array([2.75, 4.34, 1.84]))
        assert 0.0 < shape.rear_axle_to_center < shape.length


class TestSensorConfig:
    """Sensor declaration is what lets one reader serve every model."""

    def test_no_sensors_decodes_nothing(self):
        config = SensorConfig.build_no_sensors()
        assert config.camera_names_at(-1) == []
        assert not config.lidar_at(-1)
        assert not config.any_camera and not config.any_lidar

    def test_current_frame_only(self):
        config = SensorConfig.build_current_frame(("CAM_FRONT_WIDE",), lidar=True)
        assert config.camera_names_at(-1) == ["CAM_FRONT_WIDE"]
        assert config.camera_names_at(-5) == []
        assert config.lidar_at(-1) and not config.lidar_at(-2)

    def test_register_order_is_preserved(self):
        names = ("CAM_BACK_LEFT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_LEFT_WIDE")
        config = SensorConfig.build_all_sensors(names)
        # Order is part of the learned camera-register contract, so it must not
        # be normalised or sorted anywhere.
        assert config.camera_names_at(-1) == list(names)


class TestTrajectory:
    """Shape validation, because a wrong pose count is otherwise silent."""

    def test_accepts_the_contract_shape(self):
        trajectory = Trajectory(poses=np.zeros((C.TRAJECTORY_POSES, 3), dtype=np.float32))
        assert len(trajectory) == C.TRAJECTORY_POSES

    def test_normalizes_sequence_inputs_to_contiguous_float32(self):
        trajectory = Trajectory(poses=[[0, 0, 0] for _ in range(C.TRAJECTORY_POSES)])
        assert trajectory.poses.dtype == np.float32
        assert trajectory.poses.flags.c_contiguous

    def test_rejects_pose_count_mismatch(self):
        with pytest.raises(ValueError, match="sampling declares"):
            Trajectory(poses=np.zeros((5, 3), dtype=np.float32))

    def test_rejects_wrong_pose_dim(self):
        with pytest.raises(ValueError, match=r"\(x, y, heading\)"):
            Trajectory(poses=np.zeros((C.TRAJECTORY_POSES, 4), dtype=np.float32))

    def test_resamples_with_declared_time(self):
        source_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
        target_sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
        source_times = np.arange(1, 81, dtype=np.float32) * 0.1
        source = Trajectory(
            poses=np.column_stack((source_times, source_times * 0.0, source_times * 0.1)),
            trajectory_sampling=source_sampling,
        )

        target = source.resample(target_sampling)

        np.testing.assert_allclose(target.poses[:, 0], np.arange(1, 9) * 0.5)
        np.testing.assert_allclose(target.timestamps, np.arange(1, 9) * 0.5)
        assert target.trajectory_sampling == target_sampling

    def test_resample_rejects_short_source(self):
        source = Trajectory(
            poses=np.zeros((8, 3), dtype=np.float32),
            trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=0.5),
        )
        with pytest.raises(ValueError, match="beyond its horizon"):
            source.resample(TrajectorySampling(num_poses=81, interval_length=0.1))


class TestAggregation:
    """PDMS structure: two multiplicative gates over a weighted average."""

    def test_perfect_score(self):
        assert aggregate_pdm_score([1.0] * 6) == pytest.approx(1.0)

    def test_collision_zeroes_everything(self):
        # NC is a gate, not a term: a collision cannot be averaged away by
        # excellent progress and comfort.
        assert aggregate_pdm_score([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_drivable_area_violation_zeroes_everything(self):
        assert aggregate_pdm_score([1.0, 0.0, 1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_default_weights_normalise_over_twelve(self):
        # 5*EP + 5*TTC + 2*Comfort + 0*DDC, over 12.
        score = aggregate_pdm_score([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        assert score == pytest.approx((5 + 5 + 2) / 12)

    def test_ddc_has_no_default_weight(self):
        # DDC is a separate predicted head with zero default inference weight;
        # it must not leak into the aggregate.
        with_ddc = aggregate_pdm_score([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        without_ddc = aggregate_pdm_score([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        assert with_ddc == pytest.approx(without_ddc)

    def test_results_round_trip_components(self):
        components = [1.0, 1.0, 0.0, 0.8, 0.6, 1.0]
        results = PDMResults.from_components(components, token="scene@42")
        assert list(results.components.values()) == pytest.approx(components)
        assert results.components["ep"] == pytest.approx(0.6)
        assert results.token == "scene@42"

    def test_results_reject_wrong_component_count(self):
        with pytest.raises(ValueError, match="expected 6 components"):
            PDMResults.from_components([1.0, 1.0, 1.0])


class TestAnnotations:
    """Missing GT must never be indistinguishable from an empty road."""

    def test_empty_has_nine_columns(self):
        empty = Annotations.empty()
        assert len(empty) == 0
        assert empty.boxes.shape == (0, 9)

    def test_rejects_label_box_mismatch(self):
        with pytest.raises(ValueError, match="disagree"):
            Annotations(boxes=np.zeros((3, 9), dtype=np.float32), labels=np.zeros(2, dtype=np.int64))

    def test_rejects_too_few_columns(self):
        with pytest.raises(ValueError, match="T4BoxIndex"):
            Annotations(boxes=np.zeros((2, 5), dtype=np.float32), labels=np.zeros(2, dtype=np.int64))

    def test_normalizes_sequence_inputs_and_checks_tokens(self):
        annotation = Annotations(
            boxes=[[0, 0, 0, 2, 4, 1, 0, 0, 0]],
            labels=[0],
            track_tokens=["track-0"],
            velocities=[[1, 0]],
        )
        assert annotation.boxes.dtype == np.float32
        assert annotation.labels.dtype == np.int64
        assert annotation.velocities.shape == (1, 2)
        with pytest.raises(ValueError, match="track_tokens.*disagree"):
            Annotations(boxes=np.zeros((1, 9)), labels=np.zeros(1), track_tokens=[])


class TestLidar:
    def test_normalizes_sequence_inputs(self):
        lidar = Lidar(lidar_pc=[[0, 0, 0, 1, 0]])
        assert lidar.lidar_pc.dtype == np.float32
        assert lidar.lidar_pc.shape == (1, C.T4_LIDAR_POINT_DIM)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="got shape"):
            Lidar(lidar_pc=np.zeros(C.T4_LIDAR_POINT_DIM))


class TestEgoStatusIndex:
    """The seven ego columns, all differenced from the pose history."""

    def test_layout(self):
        assert EgoStatusIndex.size() == 7
        assert EgoStatusIndex.POSE == slice(0, 3)
        assert EgoStatusIndex.VELOCITY_2D == slice(3, 5)
        assert EgoStatusIndex.ACCELERATION_2D == slice(5, 7)
