"""The Lightning module every T4 agent trains through.

It is deliberately thin.  Loss, optimizers and callbacks all come from the agent,
so this class owns only what is genuinely shared: moving a batch to the device,
calling the agent, logging, and -- when the agent asks for it -- running the PDM
scorer inside the step so the scorer heads can be supervised.

Why the module and not the agent owns the scorer: a scorer instance holds a CUDA
context and a worker pool, and one per agent replica would multiply both.  The
agent declares that it wants scorer supervision; the module provides exactly one
scorer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent
from t4_e2e_devkit.common.constants import PDM_COMPONENT_ORDER


class T4LightningModule(pl.LightningModule):
    """Wraps an :class:`AbstractT4Agent` for training and validation."""

    def __init__(
        self,
        agent: AbstractT4Agent,
        scorer: Optional[Any] = None,
        log_component_metrics: bool = True,
    ) -> None:
        """
        :param agent: the agent to train.
        :param scorer: a :class:`~t4_e2e_devkit.evaluation.T4PDMScorer` for
            scorer-supervised agents; ``None`` for plain regression.
        :param log_component_metrics: log the six PDM components separately.
        """
        super().__init__()
        self.agent = agent
        self.scorer = scorer
        self.log_component_metrics = log_component_metrics

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        :param features: batched features.
        :return: the agent's predictions.
        """
        return self.agent(features)

    def _step(self, batch: Tuple[Dict, Dict], prefix: str) -> torch.Tensor:
        features, targets = batch
        predictions = self.agent(features)
        loss = self.agent.compute_loss(features, targets, predictions)

        if isinstance(loss, dict):
            if "loss" not in loss:
                raise KeyError(
                    f"{type(self.agent).__name__}.compute_loss returned a dict without a "
                    f"'loss' key; got {sorted(loss)}"
                )
            for name, value in loss.items():
                if name != "loss":
                    self.log(f"{prefix}/{name}", value, on_step=False, on_epoch=True)
            loss = loss["loss"]

        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        :param batch: ``(features, targets)``.
        :param batch_idx: index within the epoch.
        :return: the loss to backpropagate.
        """
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        :param batch: ``(features, targets)``.
        :param batch_idx: index within the epoch.
        :return: the validation loss.
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """:return: the agent's optimizers and schedulers."""
        return self.agent.get_optimizers()

    def configure_callbacks(self):
        """:return: the agent's training callbacks."""
        return self.agent.get_training_callbacks()

    def log_pdm_components(self, components: torch.Tensor, prefix: str = "train") -> None:
        """Log the six PDM components of a scored batch.

        Worth logging separately rather than only the aggregate: the aggregate
        can improve while the scorer's proposal ranking collapses, because the
        trajectory generator is improving at the same time.  Which component
        moved is what distinguishes the two.

        :param components: ``[..., 6]`` in :data:`PDM_COMPONENT_ORDER`.
        :param prefix: metric name prefix.
        """
        if not self.log_component_metrics:
            return
        flat = components.reshape(-1, components.shape[-1]).float()
        for index, name in enumerate(PDM_COMPONENT_ORDER):
            self.log(f"{prefix}/pdm/{name}", flat[:, index].mean(), on_step=False, on_epoch=True)
