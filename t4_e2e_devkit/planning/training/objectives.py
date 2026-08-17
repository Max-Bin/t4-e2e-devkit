"""Training objective and metric interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch


class AbstractObjective(ABC):
    @abstractmethod
    def compute(self, predictions: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> torch.Tensor | Mapping[str, torch.Tensor]:
        """Compute a differentiable objective."""


class AbstractTrainingMetric(ABC):
    @abstractmethod
    def compute(self, predictions: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor] | torch.Tensor:
        """Compute a detached/loggable training metric."""


class MeanSquaredTrajectoryObjective(AbstractObjective):
    """Generic trajectory regression objective."""

    def __init__(self, prediction_key: str = "trajectory", target_key: str = "trajectory") -> None:
        self.prediction_key = prediction_key
        self.target_key = target_key

    def compute(self, predictions: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predictions[self.prediction_key], targets[self.target_key])


class TrajectoryErrorMetric(AbstractTrainingMetric):
    def __init__(self, prediction_key: str = "trajectory", target_key: str = "trajectory") -> None:
        self.prediction_key = prediction_key
        self.target_key = target_key

    def compute(self, predictions: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        error = torch.linalg.vector_norm(predictions[self.prediction_key] - targets[self.target_key], dim=-1)
        return {"mean_l2": error.mean().detach(), "final_l2": error[..., -1].mean().detach()}


__all__ = [
    "AbstractObjective",
    "AbstractTrainingMetric",
    "MeanSquaredTrajectoryObjective",
    "TrajectoryErrorMetric",
]
