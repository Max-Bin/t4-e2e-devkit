"""The Lightning module every T4 agent trains through.

It is deliberately thin.  Loss, optimizers and callbacks all come from the agent,
so this class owns only what is genuinely shared: moving a batch to the device,
calling the agent, logging, and optionally running the detached PDM evaluator
for training-time reporting.

The evaluator is opt-in because it is expensive and its result is a metric, not a
differentiable training objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent


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
        :param scorer: an optional PDM/NavSim scorer
            used for detached training-time reporting.
        :param log_component_metrics: log PDM components separately.
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
        if not isinstance(batch, (tuple, list)) or len(batch) not in {2, 3}:
            raise TypeError("T4 training batches must be (features, targets[, scenes])")
        features, targets = batch[:2]
        scenes = batch[2] if len(batch) == 3 else None
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

        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise TypeError("agent loss must be a scalar torch.Tensor")
        if not torch.isfinite(loss.detach()):
            raise ValueError(f"agent loss for {prefix} is not finite")

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        self._log_training_metrics(predictions, targets, prefix)
        if self.scorer is not None:
            if scenes is None:
                raise ValueError(
                    "a scorer was configured, but the dataloader did not return scenes; "
                    "set return_scenes=True"
                )
            components = self.scorer.score_proposals(
                self._proposal_tensor(predictions),
                scenes,
                trajectory_sampling=self.agent.trajectory_sampling,
            )
            self.log_pdm_components(
                components.values,
                components.metric_names,
                prefix=prefix,
            )
        return loss

    def _proposal_tensor(self, predictions: Mapping[str, Any]) -> torch.Tensor:
        output = predictions.get("proposals", predictions.get("trajectory"))
        if not torch.is_tensor(output):
            raise TypeError(
                "scorer evaluation needs tensor predictions['trajectory'] or ['proposals']"
            )
        if output.ndim == 3:
            output = output.unsqueeze(1)
        if output.ndim != 4 or output.shape[-1] != 3:
            raise ValueError(
                "scorer proposals must have shape [B,N,T,3] or trajectory shape [B,T,3]; "
                f"got {tuple(output.shape)}"
            )
        if not output.is_floating_point() or not torch.isfinite(output).all():
            raise ValueError("scorer proposals must be finite floating-point values")
        return output

    def _log_training_metrics(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        prefix: str,
    ) -> None:
        for metric in self.agent.get_training_metrics():
            output = metric.compute(predictions, targets)
            values = (
                {type(metric).__name__.lower(): output}
                if torch.is_tensor(output) or isinstance(output, (float, int))
                else output
            )
            if not isinstance(values, Mapping):
                raise TypeError(f"{type(metric).__name__}.compute must return a scalar or mapping")
            for name, value in values.items():
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
                if value.numel() != 1:
                    raise ValueError(
                        f"training metric {name!r} must return one scalar, got {tuple(value.shape)}"
                    )
                self.log(
                    f"{prefix}/{name}", value.reshape(()).detach(), on_step=False, on_epoch=True
                )

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

    def on_train_epoch_start(self) -> None:
        """Advance the locality sampler before every training epoch."""

        trainer = getattr(self, "_trainer", None)
        datamodule = None if trainer is None else getattr(trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(self.current_epoch)

    def log_pdm_components(
        self,
        components: torch.Tensor,
        metric_names: tuple[str, ...],
        prefix: str = "train",
    ) -> None:
        """Log only the metrics selected on the scorer."""
        if not self.log_component_metrics:
            return
        if components.shape[-1] != len(metric_names):
            raise ValueError(
                "proposal metric tensor width does not match metric_names: "
                f"{components.shape[-1]} != {len(metric_names)}"
            )
        flat = components.reshape(-1, components.shape[-1]).float()
        for index, name in enumerate(metric_names):
            values = flat[:, index]
            values = values[torch.isfinite(values)]
            if values.numel():
                self.log(
                    f"{prefix}/pdm/{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )
