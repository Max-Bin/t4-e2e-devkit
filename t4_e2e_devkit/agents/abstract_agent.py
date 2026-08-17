"""The one interface every T4 end-to-end model implements.

An agent answers four questions and the rest of the devkit follows from them:

1. **What sensors do I need?**  :meth:`~AbstractT4Agent.get_sensor_config` --
   the dataset decodes exactly that and nothing else, which is what lets a
   LiDAR model and a camera model share one reader.
2. **How do I see the data?**  :meth:`~AbstractT4Agent.get_feature_builders` and
   :meth:`~AbstractT4Agent.get_target_builders`.
3. **What do I predict?**  :meth:`~AbstractT4Agent.forward`, returning a
   ``trajectory`` of ``[B, num_poses, 3]``.
4. **How am I trained?**  :meth:`~AbstractT4Agent.compute_loss` and
   :meth:`~AbstractT4Agent.get_optimizers`.

Everything downstream -- training, PDM scoring, submission, visualisation --
speaks only this interface.  Adding a model means implementing it; it does not
mean touching the dataset, the scorer or the scripts.

``compute_trajectory`` is device- and dtype-aware, and
:meth:`~AbstractT4Agent.compute_control` lets a deployable agent expose a
one-step actuator command through the same object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import torch

from t4_e2e_devkit.agents.builders import AbstractFeatureBuilder, AbstractTargetBuilder
from t4_e2e_devkit.common.dataclasses import (
    DEFAULT_TRAJECTORY_SAMPLING,
    SensorConfig,
    T4AgentInput,
    Trajectory,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class AbstractT4Agent(torch.nn.Module, ABC):
    """Interface for an end-to-end planning agent on T4 data."""

    def __init__(self, requires_scene: bool = False) -> None:
        """
        :param requires_scene: whether this agent needs the privileged scene at
            inference.  Only oracles may set it; a deployable agent cannot,
            because the future it would read does not exist on the vehicle.
        """
        super().__init__()
        self.requires_scene = requires_scene

    # ------------------------------------------------------------------ #
    # Identity and data requirements
    # ------------------------------------------------------------------ #

    @abstractmethod
    def name(self) -> str:
        """:return: the name this agent is registered and reported under."""

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: which camera and LiDAR streams to decode, at which history steps.
        """

    def initialize(self) -> None:
        """Load weights and prepare for inference.

        Called once before the first :meth:`compute_trajectory`.  Kept separate
        from ``__init__`` so an agent can be constructed from config -- for a
        dataset build or a dry run -- without paying for a checkpoint load.
        """

    # ------------------------------------------------------------------ #
    # Training surface
    # ------------------------------------------------------------------ #

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """:return: builders producing this agent's input tensors."""
        raise NotImplementedError(f"{type(self).__name__} has no feature builders")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """:return: builders producing this agent's supervision targets."""
        raise NotImplementedError(f"{type(self).__name__} has no target builders")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        :param features: batched feature tensors.
        :return: predictions; must contain ``trajectory`` of ``[B, num_poses, 3]``.
        """
        raise NotImplementedError(f"{type(self).__name__} implements no forward pass")

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param features: the batch's features.
        :param targets: the batch's targets.
        :param predictions: this agent's output for the batch.
        :return: the scalar to backpropagate, or a dict whose ``loss`` key holds
            it and whose other keys are logged.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support training")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        """
        :return: an optimizer, or ``{"optimizer": ..., "lr_scheduler": ...}``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support training")

    def get_training_callbacks(self) -> List[Any]:
        """:return: Lightning callbacks this agent contributes."""
        return []

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @property
    def trajectory_sampling(self) -> TrajectorySampling:
        """:return: the sampling of the trajectory this agent emits."""
        return DEFAULT_TRAJECTORY_SAMPLING

    @property
    def device(self) -> torch.device:
        """:return: the device this agent's parameters live on."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def compute_trajectory(self, agent_input: T4AgentInput) -> Trajectory:
        """Plan from one window.

        The batch dimension, device placement and detach are handled here so
        every agent's inference path is identical -- and so a subclass cannot
        accidentally leave a tensor on the GPU or skip ``no_grad``, which is how
        evaluation runs quietly become memory-bound.

        :param agent_input: the non-privileged view of one window.
        :return: the planned trajectory, in ego-frame local coordinates.
        """
        self.eval()
        device = self.device

        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }

        with torch.no_grad():
            predictions = self.forward(features)

        if "trajectory" not in predictions:
            raise KeyError(
                f"{type(self).__name__}.forward must return a 'trajectory' key; "
                f"got {sorted(predictions)}"
            )
        poses = predictions["trajectory"].squeeze(0).float().cpu().numpy()
        return Trajectory(poses=np.asarray(poses, dtype=np.float32),
                          trajectory_sampling=self.trajectory_sampling)

    def compute_control(self, agent_input: T4AgentInput) -> Dict[str, float]:
        """One-step actuator command for deployment.

        Optional.  An agent that implements it can be driven directly on the
        vehicle through the same object that was trained, rather than through a
        parallel runtime class that has to be kept in sync with it.

        :param agent_input: the non-privileged view of one window.
        :return: at least ``acceleration`` in m/s^2 and ``steering_rate`` in rad/s.
        """
        raise NotImplementedError(f"{type(self).__name__} exposes no control output")
