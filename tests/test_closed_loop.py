from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent
from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    EgoShape,
    EgoStatus,
    MapTensors,
    SceneMetadata,
    SensorConfig,
    T4Frame,
    T4Scene,
    Trajectory,
)
from t4_e2e_devkit.planning.simulation.closed_loop import (
    KinematicState,
    PerfectTracker,
    T4ClosedLoopConfig,
    T4ClosedLoopRunner,
    _densify_for_simulation,
    _rebase_map,
)
from t4_e2e_devkit.planning.simulation.closed_loop_geometry import compute_replay_geometry
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


class _ForwardAgent(AbstractT4Agent):
    def __init__(self, interval_length: float = 0.5, sensor_config=None):
        super().__init__()
        self.interval_length = interval_length
        self.sensor_config = sensor_config
        self.inputs = []

    def name(self) -> str:
        return "test_forward"

    def get_sensor_config(self):
        return self.sensor_config or SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input):
        self.inputs.append(agent_input)
        times = np.arange(1, 9, dtype=np.float32) * self.interval_length
        return Trajectory(
            poses=np.column_stack((times, np.zeros_like(times), np.zeros_like(times))),
            trajectory_sampling=TrajectorySampling(
                num_poses=len(times), interval_length=self.interval_length
            ),
        )


class _OracleAgent(_ForwardAgent):
    def __init__(self):
        super().__init__()
        self.requires_scene = True


def _scene(
    frame: int,
    history_frames: int = 3,
    annotations: Annotations | None = None,
    goal_pose: np.ndarray | None = None,
) -> T4Scene:
    shape = EgoShape(wheel_base=2.7, length=4.5, width=1.8)
    local_x = np.arange(-history_frames + 1, 1, dtype=np.float32) * 0.1
    frames = [
        T4Frame(
            frame_index=frame - history_frames + 1 + index,
            timestamp_us=(frame - history_frames + 1 + index) * 100_000,
            ego_status=EgoStatus(
                ego_pose=np.array([x, 0.0, 0.0], dtype=np.float32),
                ego_velocity=np.array([1.0, 0.0], dtype=np.float32),
                ego_acceleration=np.zeros(2, dtype=np.float32),
                ego_shape=shape,
            ),
            annotations=annotations if index == history_frames - 1 else None,
        )
        for index, x in enumerate(local_x)
    ]
    return T4Scene(
        scene_metadata=SceneMetadata(
            scene_dir="synthetic/scene",
            scene_id="synthetic",
            center_frame=frame,
            num_history_frames=history_frames,
            num_future_frames=0,
            global_center_pose=np.array([frame * 0.1, 0.0, 1.0, 0.0]),
        ),
        frames=frames,
        goal_pose=goal_pose,
    )


def test_perfect_tracker_matches_reference_step():
    tracker = PerfectTracker(dt_s=0.1, max_speed_mps=20.0)
    state = KinematicState(0.0, 0.0, 0.0, 0.0)
    reference = np.array([[0.1, 0.0, math.pi / 2], [0.2, 0.0, math.pi / 2]])

    updated = tracker.track(state, reference)

    assert updated.x == pytest.approx(0.1)
    assert updated.y == pytest.approx(0.0)
    assert updated.speed_mps == pytest.approx(1.0)
    assert updated.heading == pytest.approx(math.pi / 2)
    assert updated.yaw_rate_radps == pytest.approx(math.pi / 2 / 0.1)

    new_pose, new_speed = tracker.track(np.array([0.0, 0.0, 0.0, 0.0]), reference)
    np.testing.assert_allclose(new_pose, [0.1, 0.0, math.pi / 2])
    assert new_speed == pytest.approx(1.0)
    assert tracker.last_accel == pytest.approx(10.0)


def test_plan_is_densified_from_model_grid():
    plan = Trajectory(
        poses=np.column_stack(
            (
                np.arange(1, 9, dtype=np.float32) * 0.5,
                np.zeros(8, dtype=np.float32),
                np.zeros(8, dtype=np.float32),
            )
        ),
        trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=0.5),
    )

    dense = _densify_for_simulation(plan, 0.1)

    assert dense.poses.shape == (40, 3)
    np.testing.assert_allclose(dense.poses[[0, 4, -1], 0], [0.1, 0.5, 4.0])


def test_closed_loop_replans_and_feeds_back_live_ego_state():
    scenes = {10: _scene(10), 11: _scene(11)}
    agent = _ForwardAgent()
    runner = T4ClosedLoopRunner(
        agent,
        scenes.__getitem__,
        T4ClosedLoopConfig(history_frames=3, replan_interval=1),
    )

    result = runner.run(start_frame=10, num_steps=2)

    assert list(result.source_frames) == [10, 11]
    assert len(agent.inputs) == 2
    assert agent.inputs[0].ego_status.speed == pytest.approx(1.0)
    assert agent.inputs[1].ego_status.speed == pytest.approx(1.0)
    assert result.states[-1].x == pytest.approx(1.2)
    assert result.realized_trajectory().poses[-1, 0] == pytest.approx(0.2)


def test_closed_loop_can_hold_a_plan_between_replans():
    scenes = {10: _scene(10), 11: _scene(11)}
    agent = _ForwardAgent()
    runner = T4ClosedLoopRunner(
        agent,
        scenes.__getitem__,
        T4ClosedLoopConfig(history_frames=3, replan_interval=2),
    )

    result = runner.run(start_frame=10, num_steps=2)

    assert len(agent.inputs) == 1
    assert result.plans[0] is not None
    assert result.plans[1] is None
    assert result.states[-1].x == pytest.approx(1.2)


def test_closed_loop_rejects_privileged_oracle():
    with pytest.raises(ValueError, match="privileged scene"):
        T4ClosedLoopRunner(_OracleAgent(), lambda _: _scene(10))


def test_closed_loop_attaches_replay_events_to_result_and_metrics():
    annotations = Annotations(
        boxes=np.array([[2.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
        track_tokens=["replayed-agent"],
    )
    scenes = {
        10: _scene(
            10,
            annotations=annotations,
            goal_pose=np.array([0.2, 0.0, 1.0, 0.0], dtype=np.float32),
        ),
        11: _scene(
            11,
            annotations=annotations,
            goal_pose=np.array([0.2, 0.0, 1.0, 0.0], dtype=np.float32),
        ),
    }
    result = T4ClosedLoopRunner(
        _ForwardAgent(),
        scenes.__getitem__,
        T4ClosedLoopConfig(history_frames=3, replan_interval=1),
    ).run(start_frame=10, num_steps=2)

    assert result.collision_steps == (0, 1)
    assert result.goal_pose_world is not None
    assert result.termination_reason == "collision"
    from t4_e2e_devkit.evaluation.closed_loop import compute_closed_loop_metrics

    metrics = compute_closed_loop_metrics(result)
    assert metrics.collision == pytest.approx(1.0)
    assert metrics.goal_reached == pytest.approx(1.0)
    assert metrics.timeout == pytest.approx(0.0)
    assert metrics.termination_reason == "collision"
    assert metrics.min_agent_clearance_m is not None
    assert metrics.min_agent_clearance_m < 0.0
    assert metrics.ttc_violation == pytest.approx(1.0)
    assert metrics.trace is not None
    assert metrics.trace.ttc_violation is not None


def test_closed_loop_geometry_reports_map_events_only_when_map_is_available():
    scene = _scene(10)
    lanes = np.zeros((1, 4, 33), dtype=np.float32)
    lanes[0, :, 0] = [-10.0, -3.0, 3.0, 10.0]
    lanes[0, :, 4] = 0.0
    lanes[0, :, 5] = 2.0
    lanes[0, :, 6] = 0.0
    lanes[0, :, 7] = -2.0
    line_strings = np.zeros((1, 4, 4), dtype=np.float32)
    line_strings[0, :, 0] = [-10.0, -3.0, 3.0, 10.0]
    line_strings[0, :, 1] = 2.0
    line_strings[0, :, 3] = 1.0
    scene.current_frame.map_tensors = MapTensors(
        lanes=lanes,
        lanes_speed_limit=np.zeros((1, 1), dtype=np.float32),
        lanes_has_speed_limit=np.zeros((1, 1), dtype=bool),
        route_lanes=lanes.copy(),
        route_lanes_speed_limit=np.zeros((1, 1), dtype=np.float32),
        route_lanes_has_speed_limit=np.zeros((1, 1), dtype=bool),
        polygons=np.zeros((1, 4, 3), dtype=np.float32),
        line_strings=line_strings,
    )

    inside = compute_replay_geometry(KinematicState(1.0, 0.0, 0.0, 1.0), scene)
    assert inside.drivable_violation is False
    assert inside.road_border_violation is False
    assert inside.road_border_distance_m == pytest.approx(1.1, abs=0.05)

    outside = compute_replay_geometry(KinematicState(1.0, 5.0, 0.0, 1.0), scene)
    assert outside.drivable_violation is True
    assert outside.road_border_violation is False

    scene.current_frame.map_tensors = None
    unavailable = compute_replay_geometry(KinematicState(1.0, 0.0, 0.0, 1.0), scene)
    assert unavailable.drivable_violation is None
    assert unavailable.road_border_distance_m is None


@pytest.mark.data
def test_closed_loop_real_t4_wide_camera_smoke():
    """Acceptance test for a mounted T4 scene; skipped without dataset paths."""

    scene_value = os.environ.get("T4E2E_REAL_SCENE")
    if not scene_value:
        pytest.skip("set T4E2E_REAL_SCENE to run the real-data closed-loop smoke test")
    scene_dir = Path(scene_value).resolve()
    if not scene_dir.is_dir():
        pytest.fail(f"T4E2E_REAL_SCENE is not a directory: {scene_dir}")
    root = Path(os.environ.get("T4E2E_REAL_ROOT", scene_dir.parents[3])).resolve()

    from t4_e2e_devkit.common.dataclasses import SceneFilter
    from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene
    from t4_e2e_devkit.dataset.window import T4WindowBuilder

    sensor_config = sensor_config_for_scene(scene_dir, lidar=False, history=False)
    assert sensor_config.cameras, "the real-data smoke test requires JPEG-backed wide cameras"
    agent = _ForwardAgent(sensor_config=sensor_config)
    filter_config = SceneFilter(
        num_history_frames=3,
        num_future_frames=0,
        frame_interval=1,
    )
    builder = T4WindowBuilder(
        scene_dir,
        root,
        sensor_config=sensor_config,
        scene_filter=filter_config,
    )
    try:
        centers = builder.valid_centers()
        if not len(centers):
            pytest.fail(f"scene has no valid history window: {scene_dir}")
        start_frame = int(centers[len(centers) // 2])
        scene = builder.build(start_frame)
        assert scene.current_frame.cameras is not None
        assert scene.current_frame.lidar is None
    finally:
        builder.close()

    result = T4ClosedLoopRunner.from_scene_dir(
        agent,
        scene_dir,
        root,
        config=T4ClosedLoopConfig(history_frames=3, replan_interval=2),
    )
    try:
        rollout = result.run(start_frame=start_frame, num_steps=3)
    finally:
        result.close()
    assert len(rollout.source_frames) == 3
    assert len(agent.inputs) == 2
    # The agent input's ``lidars`` are the sweeps themselves, one per history
    # step -- not frames -- and this run asked for cameras only.
    assert all(sweep is None for sweep in agent.inputs[0].lidars)


def test_map_rebase_uses_recorded_frame_as_the_transform_origin():
    from t4_e2e_devkit.common.dataclasses import MapTensors

    empty = np.zeros((1, 1, 33), dtype=np.float32)
    empty[0, 0, 0] = 2.0
    empty[0, 0, 1] = 1.0
    map_tensors = MapTensors(
        lanes=empty,
        lanes_speed_limit=np.zeros((1, 1), dtype=np.float32),
        lanes_has_speed_limit=np.zeros((1, 1), dtype=bool),
        route_lanes=empty.copy(),
        route_lanes_speed_limit=np.zeros((1, 1), dtype=np.float32),
        route_lanes_has_speed_limit=np.zeros((1, 1), dtype=bool),
        polygons=np.zeros((1, 1, 3), dtype=np.float32),
        line_strings=np.zeros((1, 1, 4), dtype=np.float32),
    )

    rebased = _rebase_map(
        map_tensors,
        recorded_pose=np.array([10.0, 5.0, 0.0]),
        live_pose=np.array([11.0, 5.0, 0.0]),
    )

    np.testing.assert_allclose(rebased.lanes[0, 0, :2], [1.0, 1.0])
