"""Generic proposal scoring for model training.

The public manifest scorer is the evaluation boundary.  Training-time
proposal selection needs the same formulas on a tensor shaped
``[batch, proposals, poses, 3]``; this module keeps that small convenience
interface in the devkit instead of duplicating it in every model repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from t4_e2e_devkit.common.dataclasses import T4Scene
from t4_e2e_devkit.evaluation.navsim_score import (
    T4NavSimProposalResult,
    T4NavSimScorer,
    T4NavSimScorerConfig,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


@dataclass(frozen=True)
class PendingProposalScore:
    """A proposal score kept under the submitted proposal identity."""

    proposals_ref: Any
    result: T4NavSimProposalResult


class T4ProposalScorer:
    """Score detached proposal tensors on the shared T4 metric implementation.

    The scorer is synchronous today.  ``submit``/``wait`` are deliberately
    small and stable so a training loop may overlap or batch this operation in
    the future without adding a model-specific scoring wrapper.
    """

    def __init__(
        self,
        scorer: T4NavSimScorer,
        trajectory_sampling: TrajectorySampling,
    ) -> None:
        self.scorer = scorer
        self.trajectory_sampling = trajectory_sampling

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        num_poses: int,
        interval_seconds: float | None = None,
    ) -> "T4ProposalScorer":
        """Build a proposal scorer from a model-neutral mapping-like config."""

        def get(name: str, default: Any = None) -> Any:
            return config.get(name, default) if hasattr(config, "get") else default

        metric_names = get("pdm_metric_names")
        if metric_names is not None and isinstance(metric_names, str):
            metric_names = (metric_names,)
        elif metric_names is not None:
            metric_names = tuple(str(name) for name in metric_names)

        requested_device = str(get("t4_oracle_device", "auto") or "auto").lower()
        if requested_device in {"cuda", "gpu"} or requested_device.startswith("cuda:"):
            backend = "gpu"
            device = None if requested_device in {"cuda", "gpu"} else requested_device
        elif requested_device == "cpu":
            backend = "cpu"
            device = None
        else:
            backend = "auto"
            device = None

        scorer = T4NavSimScorer(
            T4NavSimScorerConfig(
                version=str(get("pdm_version", "v2")),
                metric_names=metric_names,
                backend=backend,
                device=device,
                gpu_dtype=str(get("t4_oracle_gpu_dtype", "float32")),
                require_extended_comfort=bool(get("pdm_require_extended_comfort", False)),
                use_simulator=bool(get("pdm_use_simulator", True)),
            )
        )
        interval = (
            float(interval_seconds)
            if interval_seconds is not None
            else float(get("t4_trajectory_dt_s", 0.5))
        )
        return cls(
            scorer,
            TrajectorySampling(num_poses=int(num_poses), interval_length=interval),
        )

    def score(
        self,
        proposals: Any,
        scenes: Sequence[T4Scene],
    ) -> T4NavSimProposalResult:
        """Return named metrics for ``[B,T,3|4]`` or ``[B,N,T,3|4]``."""

        if getattr(proposals, "ndim", None) == 3:
            proposals = proposals[:, None]
        if getattr(proposals, "ndim", None) != 4:
            raise ValueError("proposals must have shape [B,T,3|4] or [B,N,T,3|4]")
        if proposals.shape[-1] == 4:
            import torch

            heading = torch.atan2(proposals[..., 3], proposals[..., 2]).unsqueeze(-1)
            proposals = torch.cat((proposals[..., :2], heading), dim=-1)
        return self.scorer.score_proposals(
            proposals,
            scenes,
            trajectory_sampling=self.trajectory_sampling,
        )

    def submit(self, proposals: Any, scenes: Sequence[T4Scene]) -> PendingProposalScore:
        """Submit proposals and retain their identity for a later ``wait``."""

        return PendingProposalScore(
            proposals_ref=proposals,
            result=self.score(proposals, scenes),
        )

    @staticmethod
    def wait(pending: PendingProposalScore) -> T4NavSimProposalResult:
        """Return a previously submitted proposal score."""

        return pending.result


__all__ = ["PendingProposalScore", "T4ProposalScorer"]
