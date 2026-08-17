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
    ConstantVelocityTrafficAgentController,
    ConstantVelocityTrafficPolicy,
    ReactiveTrafficPolicy,
    ReplayTrafficPolicy,
)
from t4_e2e_devkit.planning.simulation.manager import SimulationRequest, T4SimulationManager
from t4_e2e_devkit.planning.simulation.runtime import (
    SimulationRunner,
    SimulationSetup,
    StepSimulationTimeController,
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


def test_reactive_policy_updates_each_agent_and_preserves_replay_input():
    scene = _scene(10)
    scene.current_frame.annotations = Annotations(
        boxes=np.array([[1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0]], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
        track_tokens=["agent-1"],
    )

    updated = ReactiveTrafficPolicy(ConstantVelocityTrafficAgentController()).update(
        scene,
        state=KinematicState(0.0, 0.0, 0.0, 1.0),
        step=0,
        dt_s=0.5,
    )

    assert updated.current_frame.annotations.boxes[0, 0] == 2.0
    assert updated.current_frame.annotations.track_tokens == ["agent-1"]
    assert scene.current_frame.annotations.boxes[0, 0] == 1.0


class _Lifecycle:
    def __init__(self):
        self.starts = []
        self.steps = []
        self.ends = []

    def on_start(self, token, state):
        self.starts.append((token, state))

    def on_step(self, tick):
        self.steps.append(tick)

    def on_end(self, result):
        self.ends.append(result)


def test_simulation_manager_runs_ordered_requests_with_lifecycle_hooks():
    lifecycle = _Lifecycle()
    runner = T4ClosedLoopRunner(
        _Agent(),
        lambda frame: _scene(frame),
        T4ClosedLoopConfig(history_frames=3),
    )
    manager = T4SimulationManager(runner, callbacks=[lifecycle])

    results = manager.run_many(
        [SimulationRequest(start_frame=10, num_steps=1), SimulationRequest(10, 2)]
    )

    assert [len(result.source_frames) for result in results] == [1, 2]
    assert len(lifecycle.starts) == 2
    assert len(lifecycle.steps) == 3
    assert len(lifecycle.ends) == 2


def test_generic_simulation_keeps_state_observation_and_callbacks_aligned():
    class Scenario:
        initial_ego_state = 0

    class Observation:
        def __init__(self):
            self.reset_count = 0
            self.initialized = 0
            self.indices = []

        def reset(self):
            self.reset_count += 1

        def initialize(self):
            self.initialized += 1

        def get_observation(self, iteration, history):
            del history
            self.indices.append(iteration.index)
            return f"observation-{iteration.index}"

    class Planner:
        observation_type = str

        def __init__(self):
            self.reset_count = 0
            self.initialized = 0
            self.inputs = []

        def reset(self):
            self.reset_count += 1

        def initialize(self, initialization):
            del initialization
            self.initialized += 1

        def compute_trajectory(self, planner_input):
            self.inputs.append(planner_input)
            return planner_input.iteration.index

    class Controller:
        def __init__(self):
            self.state = 0
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            self.state = 0

        def update_state(self, trajectory, iteration):
            assert trajectory == iteration.index
            self.state += 1
            return self.state

    class Callback:
        def __init__(self):
            self.events = []

        def on_simulation_start(self, setup):
            self.events.append("start")

        def on_initialization_end(self, setup, planner):
            self.events.append("initialized")

        def on_simulation_step(self, sample):
            self.events.append((sample.ego_state, sample.observation))

        def on_simulation_end(self, setup, planner, history):
            self.events.append("end")

    observation = Observation()
    planner = Planner()
    controller = Controller()
    callback = Callback()
    setup = SimulationSetup(
        scenario=Scenario(),
        planner=planner,
        observation=observation,
        ego_controller=controller,
        time_controller=StepSimulationTimeController(0, 100_000, 2),
        callbacks=(callback,),
    )

    history = SimulationRunner().run(setup)

    assert [(sample.ego_state, sample.observation) for sample in history] == [
        (0, "observation-0"),
        (1, "observation-1"),
    ]
    assert observation.indices == [0, 1]
    assert len(planner.inputs) == 2
    assert observation.reset_count == planner.reset_count == controller.reset_count == 1
    assert observation.initialized == planner.initialized == 1
    assert callback.events == [
        "start",
        "initialized",
        (0, "observation-0"),
        (1, "observation-1"),
        "end",
    ]
