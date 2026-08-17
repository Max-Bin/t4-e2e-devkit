"""Open-loop trajectory metrics.

Open-loop metrics compare a planned ego trajectory with the recorded future.
They are intentionally independent from PDM-Score and the T4 safety family:
trajectory fidelity answers whether the plan follows the driver, while the
other families answer whether it is safe or physically usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from t4_e2e_devkit.common.constants import T4_INTERVAL_LENGTH
from t4_e2e_devkit.common.dataclasses import T4Scene, Trajectory
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


@dataclass(frozen=True)
class OpenLoopMetricConfig:
    """Settings for comparing two trajectories on a common time grid."""

    miss_threshold_m: float = 2.0
    target_sampling: Optional[TrajectorySampling] = None

    def __post_init__(self) -> None:
        if self.miss_threshold_m <= 0.0:
            raise ValueError("miss_threshold_m must be positive")


@dataclass(frozen=True)
class OpenLoopMetrics:
    """Per-window open-loop errors.

    ``miss_rate`` is a per-window indicator.  Averaging it over windows gives
    the dataset miss rate, while retaining the indicator keeps per-window CSV
    output useful.
    """

    ade_m: float
    fde_m: float
    heading_mae_rad: float
    final_heading_error_rad: float
    miss_rate: float
    horizon_s: float
    num_poses: int
    token: Optional[str] = None

    @property
    def values(self) -> Mapping[str, float]:
        """Return scalar metrics without the optional window token."""

        return {
            "ade_m": self.ade_m,
            "fde_m": self.fde_m,
            "heading_mae_rad": self.heading_mae_rad,
            "final_heading_error_rad": self.final_heading_error_rad,
            "miss_rate": self.miss_rate,
            "horizon_s": self.horizon_s,
            "num_poses": float(self.num_poses),
        }


def compute_open_loop_metrics(
    prediction: Trajectory,
    ground_truth: Trajectory | T4Scene,
    *,
    config: Optional[OpenLoopMetricConfig] = None,
    token: Optional[str] = None,
) -> OpenLoopMetrics:
    """Compare a prediction with recorded future poses.

    When no target grid is specified, the ground-truth interval is retained
    and the comparison is truncated to the common horizon.  This accepts both
    dense long-horizon outputs and sparse short-horizon outputs without
    inferring time from a point count.  A configured target grid is useful when
    a benchmark requires one fixed horizon across all runs.

    :param prediction: planned ego trajectory in the current ego frame.
    :param ground_truth: recorded trajectory or a scene carrying one.
    :param config: miss threshold and optional explicit comparison grid.
    :param token: optional window identifier for reports.
    :return: one independent open-loop result.
    """

    config = config or OpenLoopMetricConfig()
    scene_token = None
    if isinstance(ground_truth, T4Scene):
        scene_token = ground_truth.scene_metadata.token
        if config.target_sampling is not None:
            ground_truth = ground_truth.get_future_trajectory(
                trajectory_sampling=config.target_sampling
            )
        else:
            if ground_truth.future_ego_poses is None:
                raise ValueError(
                    f"scene {scene_token} carries no future poses for open-loop evaluation"
                )
            ground_truth = Trajectory(
                poses=ground_truth.future_ego_poses,
                trajectory_sampling=TrajectorySampling(
                    num_poses=len(ground_truth.future_ego_poses),
                    interval_length=T4_INTERVAL_LENGTH,
                ),
            )

    target = config.target_sampling or _overlap_sampling(prediction, ground_truth)
    prediction_horizon = float(prediction.trajectory_sampling.time_horizon)
    ground_truth_horizon = float(ground_truth.trajectory_sampling.time_horizon)
    if target.time_horizon > prediction_horizon + 1e-9:
        raise ValueError(
            "comparison horizon exceeds prediction horizon: "
            f"target={target.time_horizon:g}s, prediction={prediction_horizon:g}s"
        )
    if target.time_horizon > ground_truth_horizon + 1e-9:
        raise ValueError(
            "comparison horizon exceeds ground-truth horizon: "
            f"target={target.time_horizon:g}s, ground_truth={ground_truth_horizon:g}s"
        )

    predicted = prediction.resample(target)
    recorded = ground_truth.resample(target)
    position_error = np.linalg.norm(predicted.poses[:, :2] - recorded.poses[:, :2], axis=-1)
    heading_delta = np.arctan2(
        np.sin(predicted.poses[:, 2] - recorded.poses[:, 2]),
        np.cos(predicted.poses[:, 2] - recorded.poses[:, 2]),
    )
    return OpenLoopMetrics(
        ade_m=float(np.mean(position_error)),
        fde_m=float(position_error[-1]),
        heading_mae_rad=float(np.mean(np.abs(heading_delta))),
        final_heading_error_rad=float(abs(heading_delta[-1])),
        miss_rate=float(position_error[-1] > config.miss_threshold_m),
        horizon_s=float(target.time_horizon),
        num_poses=int(target.num_poses),
        token=token or scene_token,
    )


def aggregate_open_loop_metrics(results: Sequence[OpenLoopMetrics]) -> dict[str, float]:
    """Average open-loop metrics without mixing in another metric family."""

    if not results:
        return {"num_scenes": 0.0}
    names = tuple(results[0].values.keys())
    report: dict[str, float] = {"num_scenes": float(len(results))}
    for name in names:
        report[name] = float(np.mean([result.values[name] for result in results]))
    return report


def _overlap_sampling(prediction: Trajectory, ground_truth: Trajectory) -> TrajectorySampling:
    """Use the GT interval over the largest horizon both trajectories cover."""

    interval = float(ground_truth.trajectory_sampling.interval_length)
    horizon = min(
        float(prediction.trajectory_sampling.time_horizon),
        float(ground_truth.trajectory_sampling.time_horizon),
    )
    num_poses = int(np.floor(horizon / interval + 1e-9))
    if num_poses < 1:
        raise ValueError(
            "prediction and ground truth have no common positive-time sample"
        )
    return TrajectorySampling(num_poses=num_poses, interval_length=interval)


__all__ = [
    "OpenLoopMetricConfig",
    "OpenLoopMetrics",
    "aggregate_open_loop_metrics",
    "compute_open_loop_metrics",
]
