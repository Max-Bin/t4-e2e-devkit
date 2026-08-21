"""Agent interface tests.

The point of these is that they are written against ``AbstractT4Agent`` alone.
Any model registered in the devkit passes or fails them without a single
model-specific line -- which is what "unified interface" has to mean in practice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t4_e2e_devkit.agents import AbstractT4Agent, available_agents, build_agent
from t4_e2e_devkit.agents.builders import (
    EgoStatusFeatureBuilder,
    MapFeatureBuilder,
    TrajectoryTargetBuilder,
)
from t4_e2e_devkit.agents.registry import register_agent
from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.common.dataclasses import SensorConfig


class TestRegistry:
    def test_builtins_are_registered(self):
        assert {"constant_velocity", "human"} <= set(available_agents())

    def test_unknown_agent_lists_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            build_agent("no_such_agent")

    def test_duplicate_registration_is_refused(self):
        # Two packages claiming one name would make which model ran depend on
        # import order.
        with pytest.raises(ValueError, match="already registered"):
            register_agent("constant_velocity", lambda: None)

    def test_non_agent_is_refused(self):
        register_agent("_not_an_agent", lambda: object(), overwrite=True)
        with pytest.raises(TypeError, match="AbstractT4Agent"):
            build_agent("_not_an_agent")


@pytest.mark.parametrize("name", ["constant_velocity", "human"])
class TestInterfaceConformance:
    """Every registered agent must answer the interface's questions."""

    def test_is_an_agent(self, name):
        agent = build_agent(name)
        assert isinstance(agent, AbstractT4Agent)
        assert isinstance(agent, torch.nn.Module)

    def test_declares_a_name(self, name):
        assert build_agent(name).name() == name

    def test_declares_a_sensor_config(self, name):
        assert isinstance(build_agent(name).get_sensor_config(), SensorConfig)

    def test_initialize_is_idempotent(self, name):
        agent = build_agent(name)
        agent.initialize()
        agent.initialize()

    def test_declares_target_builders(self, name):
        assert len(build_agent(name).get_target_builders()) >= 1

    def test_trajectory_sampling_matches_the_contract(self, name):
        sampling = build_agent(name).trajectory_sampling
        assert sampling.num_poses == C.TRAJECTORY_POSES
        assert sampling.interval_length == pytest.approx(C.TRAJECTORY_INTERVAL)


class TestConstantVelocity:
    """The floor baseline, checked against hand-computable cases."""

    def test_straight_line_extrapolation(self):
        agent = build_agent("constant_velocity")
        # 10 m/s forward: pose k should be at 10 * 0.5 * (k+1) metres.
        ego_status = torch.zeros(1, 7)
        ego_status[0, 3] = 10.0
        trajectory = agent({"ego_status": ego_status})["trajectory"]
        assert trajectory.shape == (1, C.TRAJECTORY_POSES, 3)
        expected_x = 10.0 * C.TRAJECTORY_INTERVAL * torch.arange(1, C.TRAJECTORY_POSES + 1)
        torch.testing.assert_close(trajectory[0, :, 0], expected_x)
        torch.testing.assert_close(trajectory[0, :, 1], torch.zeros(C.TRAJECTORY_POSES))

    def test_stationary_ego_stays_put(self):
        agent = build_agent("constant_velocity")
        trajectory = agent({"ego_status": torch.zeros(1, 7)})["trajectory"]
        torch.testing.assert_close(trajectory, torch.zeros_like(trajectory))

    def test_lateral_velocity_is_used(self):
        # The ego status is already in the current frame, so vy is real lateral
        # motion, not an artefact to be dropped.
        agent = build_agent("constant_velocity")
        ego_status = torch.zeros(1, 7)
        ego_status[0, 3:5] = torch.tensor([3.0, 4.0])
        trajectory = agent({"ego_status": ego_status})["trajectory"]
        assert trajectory[0, -1, 1] == pytest.approx(4.0 * C.TRAJECTORY_INTERVAL * 8)

    def test_history_input_uses_the_current_row(self):
        agent = build_agent("constant_velocity")
        history = torch.zeros(1, C.PAST_FRAMES, 7)
        history[0, -1, 3] = 5.0  # only the current row moves
        trajectory = agent({"ego_status": history})["trajectory"]
        assert trajectory[0, 0, 0] == pytest.approx(5.0 * C.TRAJECTORY_INTERVAL)


class TestHumanAgentIsAnOracle:
    def test_declares_that_it_needs_the_scene(self):
        assert build_agent("human").requires_scene is True

    def test_refuses_to_plan_from_an_agent_input(self):
        # Returning something plausible here would make an oracle look like a
        # very poor model rather than a misuse.
        agent = build_agent("human")
        with pytest.raises(NotImplementedError, match="privileged scene"):
            agent.compute_trajectory(object())


class TestBuilders:
    def test_map_builder_refuses_to_zero_fill(self):
        class NoMap:
            ego_statuses = []
            cameras = []
            lidars = []
            map_tensors = None
            goal_pose = None

        with pytest.raises(ValueError, match="never zero-filled"):
            MapFeatureBuilder().compute_features(NoMap())

    def test_builder_names_are_distinct(self):
        builders = [
            EgoStatusFeatureBuilder(),
            EgoStatusFeatureBuilder(include_history=True),
            MapFeatureBuilder(),
            TrajectoryTargetBuilder(),
        ]
        names = [builder.get_unique_name() for builder in builders]
        assert len(names) == len(set(names))


@pytest.mark.data
class TestAgentsOnRealScenes:
    """The loop closes: a scene in, a valid trajectory out, for every agent."""

    def test_constant_velocity_plans_from_an_agent_input(self, scene):
        agent = build_agent("constant_velocity")
        agent.initialize()
        trajectory = agent.compute_trajectory(scene.get_agent_input())
        assert trajectory.poses.shape == (C.TRAJECTORY_POSES, 3)
        assert np.isfinite(trajectory.poses).all()

    def test_human_replays_the_recorded_future(self, scene):
        agent = build_agent("human")
        trajectory = agent.compute_trajectory_from_scene(scene)
        np.testing.assert_allclose(trajectory.poses, scene.get_future_trajectory().poses, atol=1e-6)

    def test_feature_and_target_builders_run(self, scene):
        agent = build_agent("constant_velocity")
        agent_input = scene.get_agent_input()
        features = {}
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        targets = {}
        for builder in agent.get_target_builders():
            targets.update(builder.compute_targets(scene))
        assert features["ego_status"].shape == (7,)
        assert targets["trajectory"].shape == (C.TRAJECTORY_POSES, 3)
