"""Reference agents.

These exist to calibrate everything else.  A PDM score is only meaningful next
to what trivial and perfect behaviour score on the same windows:

* :class:`ConstantVelocityAgent` -- the floor.  Extrapolates the current
  velocity, uses no map and no sensors.  A learned model that does not clear it
  has learned nothing about the scene.
* :class:`HumanAgent` -- the ceiling, and a test of the scorer rather than of a
  model.  It replays the recorded human trajectory, so its score should be near
  the maximum; when it is not, the deficit locates a disagreement between the
  scorer and the data -- a map field that does not cover where the vehicle
  actually drove, a track that the bridge failed to interpolate -- not a
  driving mistake.

They are also the cheapest possible check that the interface holds end to end:
both run through the same dataset, builders and scorer as a full model.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent
from t4_e2e_devkit.agents.builders import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
    EgoStatusFeatureBuilder,
    TrajectoryTargetBuilder,
)
from t4_e2e_devkit.common.constants import TRAJECTORY_INTERVAL, TRAJECTORY_POSES
from t4_e2e_devkit.common.dataclasses import (
    SensorConfig,
    T4AgentInput,
    T4Scene,
    Trajectory,
)
from t4_e2e_devkit.common.enums import EgoStatusIndex
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class ConstantVelocityAgent(AbstractT4Agent):
    """Extrapolates the current velocity along the current heading."""

    def __init__(
        self,
        num_poses: int = TRAJECTORY_POSES,
        interval_length: float = TRAJECTORY_INTERVAL,
    ) -> None:
        """
        :param num_poses: poses to emit.
        :param interval_length: seconds between poses.
        """
        super().__init__()
        self.num_poses = num_poses
        self.interval_length = interval_length

    @property
    def trajectory_sampling(self) -> TrajectorySampling:
        return TrajectorySampling(
            num_poses=self.num_poses,
            interval_length=self.interval_length,
        )

    def name(self) -> str:
        """:return: agent name."""
        return "constant_velocity"

    def get_sensor_config(self) -> SensorConfig:
        """:return: no sensors; this agent reads only the ego row."""
        return SensorConfig.build_no_sensors()

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """:return: the ego status builder."""
        return [EgoStatusFeatureBuilder()]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """:return: the trajectory target builder."""
        return [
            TrajectoryTargetBuilder(trajectory_sampling=self.trajectory_sampling)
        ]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        :param features: must contain ``ego_status`` ``[B, 7]``.
        :return: ``trajectory`` ``[B, num_poses, 3]``.
        """
        ego_status = features["ego_status"]
        if ego_status.dim() == 3:  # a history was requested; take the current row
            ego_status = ego_status[:, -1]

        # The ego status is already expressed in the current frame, so the pose
        # is the origin and the velocity is the body-frame direction of travel.
        velocity = ego_status[:, EgoStatusIndex.VELOCITY_2D]
        times = torch.arange(
            1, self.num_poses + 1, device=ego_status.device, dtype=ego_status.dtype
        ) * self.interval_length

        displacement = velocity.unsqueeze(1) * times.view(1, -1, 1)  # [B, P, 2]
        heading = torch.zeros_like(displacement[..., :1])
        return {"trajectory": torch.cat([displacement, heading], dim=-1)}

    def compute_loss(self, features, targets, predictions):
        """:raises NotImplementedError: this agent has no parameters to train."""
        raise NotImplementedError("ConstantVelocityAgent has no parameters")


class HumanAgent(AbstractT4Agent):
    """Replays the recorded human trajectory.

    This is an oracle: it reads the future, so ``requires_scene`` is set and it
    can never be deployed.  Its trajectory comes from
    :meth:`T4Scene.get_future_trajectory`, the same call the target builder
    makes, so its score is the scorer's own upper bound on these windows.
    """

    def __init__(
        self,
        num_poses: int = TRAJECTORY_POSES,
        interval_length: float = TRAJECTORY_INTERVAL,
    ) -> None:
        """
        :param num_poses: poses to emit.
        :param interval_length: seconds between poses.
        """
        super().__init__(requires_scene=True)
        self.num_poses = num_poses
        self.interval_length = interval_length

    @property
    def trajectory_sampling(self) -> TrajectorySampling:
        return TrajectorySampling(
            num_poses=self.num_poses,
            interval_length=self.interval_length,
        )

    def name(self) -> str:
        """:return: agent name."""
        return "human"

    def get_sensor_config(self) -> SensorConfig:
        """:return: no sensors."""
        return SensorConfig.build_no_sensors()

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """:return: no feature builders; this agent reads the scene directly."""
        return []

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """:return: the trajectory target builder."""
        return [
            TrajectoryTargetBuilder(trajectory_sampling=self.trajectory_sampling)
        ]

    def compute_trajectory_from_scene(self, scene: T4Scene) -> Trajectory:
        """
        :param scene: the privileged window view.
        :return: the recorded human trajectory.
        """
        return scene.get_future_trajectory(
            trajectory_sampling=self.trajectory_sampling
        )

    def compute_trajectory(self, agent_input: T4AgentInput) -> Trajectory:
        """:raises NotImplementedError: always -- use
        :meth:`compute_trajectory_from_scene`.

        An oracle cannot plan from an agent input by definition, and silently
        returning a straight line here would make it look like a very poor model
        instead of a misuse.
        """
        raise NotImplementedError(
            "HumanAgent is an oracle and needs the privileged scene; call "
            "compute_trajectory_from_scene(scene) instead of compute_trajectory(agent_input)."
        )
