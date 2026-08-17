from __future__ import annotations

import numpy as np

from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    EgoShape,
    EgoStatus,
    SceneMetadata,
    SensorConfig,
    T4Frame,
    T4Scene,
    Trajectory,
)
from t4_e2e_devkit.planning.simulation.closed_loop import (
    KinematicState,
    T4ClosedLoopConfig,
    T4ClosedLoopRunner,
)
from t4_e2e_devkit.planning.simulation.interfaces import (
    ConstantVelocityTrafficPolicy,
    ReplayTrafficPolicy,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class _Agent:
    requires_scene = False

    def __init__(self):
        self.inputs = []

    def compute_trajectory(self, agent_input):
        self.inputs.append(agent_input)
        return Trajectory(
            poses=np.column_stack(
                (np.arange(1, 9, dtype=np.float32) * 0.1, np.zeros(8), np.zeros(8))
            ),
            trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=0.1),
        )


class _Controller:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def step(self, state, reference_world):
        del reference_world
        return KinematicState(
            state.x + 0.05,
            state.y,
            state.heading,
            state.speed_mps,
        )


def _scene(frame: int) -> T4Scene:
    shape = EgoShape(2.7, 4.5, 1.8)
    frames = [
        T4Frame(
            frame_index=frame - 2 + index,
            timestamp_us=(frame - 2 + index) * 100_000,
            ego_status=EgoStatus(
                ego_pose=np.array([index * 0.1, 0.0, 0.0], dtype=np.float32),
                ego_velocity=np.array([1.0, 0.0], dtype=np.float32),
                ego_acceleration=np.zeros(2, dtype=np.float32),
                ego_shape=shape,
            ),
            annotations=Annotations.empty() if index == 2 else None,
        )
        for index in range(3)
    ]
    return T4Scene(
        scene_metadata=SceneMetadata(
            scene_dir="prd_jt/scene",
            scene_id="scene",
            center_frame=frame,
            num_history_frames=3,
            num_future_frames=0,
            global_center_pose=np.array([0.0, 0.0, 1.0, 0.0]),
        ),
        frames=frames,
    )


def test_runner_accepts_controller_and_default_replay_policy():
    controller = _Controller()
    agent = _Agent()
    result = T4ClosedLoopRunner(
        agent,
        lambda frame: _scene(frame),
        T4ClosedLoopConfig(history_frames=3),
        controller=controller,
        traffic_policy=ReplayTrafficPolicy(),
    ).run(start_frame=10, num_steps=2)
    assert controller.reset_count == 1
    assert result.realized_poses_world[-1, 0] == 0.1


def test_constant_velocity_policy_does_not_mutate_replay_scene():
    scene = _scene(10)
    scene.current_frame.annotations = Annotations(
        boxes=np.array([[1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0]], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
    )
    updated = ConstantVelocityTrafficPolicy().update(
        scene,
        state=KinematicState(0.0, 0.0, 0.0, 1.0),
        step=0,
        dt_s=0.5,
    )
    assert updated is not scene
    assert updated.current_frame.annotations.boxes[0, 0] == 2.0
    assert scene.current_frame.annotations.boxes[0, 0] == 1.0
    assert SensorConfig.build_no_sensors().lidar is False
