from __future__ import annotations

import numpy as np

from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    EgoShape,
    EgoStatus,
    SceneMetadata,
    T4Frame,
    T4Scene,
)
from t4_e2e_devkit.planning.scenario_builder.abstract_scenario import T4Scenario


def _scenario() -> T4Scenario:
    shape = EgoShape(wheel_base=2.7, length=4.5, width=1.8)
    frames = [
        T4Frame(
            frame_index=index,
            timestamp_us=index * 100_000,
            ego_status=EgoStatus(
                ego_pose=np.array([(index - 2) * 0.1, 0.0, 0.0], dtype=np.float32),
                ego_velocity=np.array([1.0, 0.0], dtype=np.float32),
                ego_acceleration=np.zeros(2, dtype=np.float32),
                ego_shape=shape,
            ),
            annotations=Annotations.empty(),
        )
        for index in range(3)
    ]
    future_poses = np.column_stack(
        (
            np.arange(1, 5, dtype=np.float32) * 0.1,
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )
    )
    annotations = [Annotations.empty() for _ in range(5)]
    return T4Scenario(
        T4Scene(
            scene_metadata=SceneMetadata(
                scene_dir="prd_jt/day/vehicle/scene",
                scene_id="scene",
                center_frame=2,
                num_history_frames=3,
                num_future_frames=4,
                global_center_pose=np.array([2.0, 0.0, 1.0, 0.0]),
            ),
            frames=frames,
            future_ego_poses=future_poses,
            future_annotations=annotations,
        ),
        interval_length=0.1,
    )


def test_scenario_samples_keep_the_requested_horizon():
    scenario = _scenario()

    assert scenario.token.endswith("@2")
    assert scenario.get_number_of_iterations() == 5
    assert scenario.get_time_point(2) == 400_000

    past = scenario.get_past_ego_statuses(time_horizon=0.2, num_samples=1)
    np.testing.assert_allclose([status.ego_pose[0] for status in past], [-0.2, 0.0])

    future = scenario.get_future_ego_statuses(time_horizon=0.4, num_samples=2)
    np.testing.assert_allclose([status.ego_pose[0] for status in future], [0.0, 0.2, 0.4])


def test_scenario_exposes_tracks_sensor_replay_and_map_optional():
    scenario = _scenario()

    assert len(scenario.get_tracked_objects_at_iteration(0).tracked_objects) == 0
    assert len(scenario.get_future_tracked_objects(time_horizon=0.4, num_samples=2)) == 3
    assert scenario.get_sensor_frame_at_iteration(0).frame_index == 2
    assert scenario.get_ego_state_at_iteration(0) is scenario.initial_ego_state
    assert [frame.frame_index for frame in scenario.get_past_sensor_frames(time_horizon=0.2)] == [0, 1, 2]
    assert scenario.get_route_roadblock_ids() == ()
    assert scenario.get_mission_goal() is None
    assert scenario.get_map_api() is None
