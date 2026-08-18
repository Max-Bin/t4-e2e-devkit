"""Environment-gated acceptance checks against a mounted T4 scene."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from t4_e2e_devkit.common.constants import T4_INTERVAL_LENGTH
from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene
from t4_e2e_devkit.dataset.window import T4WindowBuilder
from t4_e2e_devkit.evaluation import MetricContext, MetricEngine, compute_open_loop_metrics
from t4_e2e_devkit.planning.scenario_builder import T4Scenario


@pytest.mark.data
def test_real_t4_window_satisfies_scenario_sensor_and_metric_contracts():
    """Run the public interfaces on one mounted scene without private paths."""

    scene_value = os.environ.get("T4E2E_REAL_SCENE")
    if not scene_value:
        pytest.skip("set T4E2E_REAL_SCENE to run the T4 acceptance check")
    scene_dir = Path(scene_value).resolve()
    if not scene_dir.is_dir():
        pytest.fail(f"T4E2E_REAL_SCENE is not a directory: {scene_dir}")
    root = Path(os.environ.get("T4E2E_REAL_ROOT", scene_dir.parents[3])).resolve()

    sensor_config = sensor_config_for_scene(scene_dir, lidar=False, history=False)
    if not sensor_config.cameras:
        pytest.fail("the acceptance scene has no supported JPEG-backed wide camera")
    builder = T4WindowBuilder(
        scene_dir,
        root,
        sensor_config=sensor_config,
        scene_filter=SceneFilter(
            num_history_frames=3,
            num_future_frames=80,
            frame_interval=1,
            has_route=False,
        ),
        reader_config={
            "t4_attach_map_ids": True,
            "t4_maps_root": os.environ.get("T4E2E_MAPS_ROOT"),
        },
    )
    try:
        centers = builder.valid_centers()
        if not len(centers):
            pytest.fail("the acceptance scene has no 3-history/80-future window")
        scene = builder.build(int(centers[len(centers) // 2]))
        scenario = T4Scenario(scene, T4_INTERVAL_LENGTH, map_api=builder.map_api)

        assert scene.current_frame.cameras is not None
        assert scene.current_frame.lidar is None
        assert scenario.get_ego_state_at_iteration(0) is scenario.initial_ego_state
        assert scenario.get_mission_goal() is None or len(scenario.get_mission_goal()) == 4

        prediction = scene.get_future_trajectory(num_poses=8, stride=5)
        open_loop = compute_open_loop_metrics(prediction, scene)
        report = MetricEngine.t4_default().evaluate(
            MetricContext(
                token=scenario.token,
                prediction=prediction,
                ground_truth=scene,
                scene=scene,
            ),
            families=("open_loop",),
        )
        assert open_loop.num_poses == 8
        assert {record.family for record in report.records} == {"open_loop"}
    finally:
        builder.close()
