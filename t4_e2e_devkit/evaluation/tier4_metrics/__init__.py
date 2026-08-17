"""TIER IV's own scoring family, reported alongside PDM-Score.

This is a separate metric family. It covers questions that PDM-Score does not:

* ``red_light`` -- traffic-light compliance. PDM-Score has no traffic-light
  term, while the T4 batch carries a one-hot signal in every
  ``route_lanes`` point.
* ``static_collision`` -- PDM's NC scores at-fault collisions with *dynamic*
  agents; driving into a stopped obstacle is a different failure.
* ``lane_departure`` / ``road_border`` -- graded departure, where DAC is a
  binary area check.
* ``feasibility`` / ``kinematic_gate`` -- whether the plan is executable at all,
  which no PDM-Score term asks.
* ``temporal_stability`` -- agreement between consecutive frames' plans.  Every
  PDM-Score term is single-frame.

PDM-Score keeps a different comfort formulation and collision semantics. The
families are reported side by side and are not summed.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from t4_e2e_devkit.common.constants import SCORER_FUTURE_FRAMES, T4_INTERVAL_LENGTH
from t4_e2e_devkit.common.dataclasses import Trajectory
from t4_e2e_devkit.evaluation.tier4_metrics.config import RewardConfig
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

__all__ = [
    "RewardConfig",
    "aggregate_tier4_metrics",
    "compute_tier4_metrics",
    "densify_trajectory",
]


def densify_trajectory(
    poses: np.ndarray | Trajectory,
    num_frames: int = SCORER_FUTURE_FRAMES,
    source_interval: Optional[float] = None,
    target_interval: float = T4_INTERVAL_LENGTH,
    source_sampling: Optional[TrajectorySampling] = None,
) -> torch.Tensor:
    """Resample a trajectory to the metric family's 10 Hz grid.

    The metrics differentiate the path to get accelerations and yaw rates, and
    their thresholds are calibrated at ``dt = 0.1``.  Feeding 0.5 s poses
    directly would divide by the wrong ``dt`` and make every acceleration read
    five times too small -- so the resampling is part of the contract, not a
    convenience.

    :param poses: ``[P, 3]`` of ``(x, y, heading)``, or a
        :class:`Trajectory` carrying its sampling.
    :param num_frames: output length.
    :param source_interval: seconds between input poses when ``poses`` is an
        array.  Arrays default to 0.5 seconds.
    :param target_interval: seconds between output poses.
    :param source_sampling: explicit sampling for an array input.
    :return: ``[1, num_frames, 4]`` of ``(x, y, cos, sin)``.
    """
    if isinstance(poses, Trajectory):
        if source_sampling is not None or source_interval is not None:
            raise ValueError(
                "source_sampling/source_interval are only valid for array inputs"
            )
        source_sampling = poses.trajectory_sampling
        values = poses.poses
    else:
        values = poses
        if source_sampling is not None and source_interval is not None:
            raise ValueError("source_sampling and source_interval are mutually exclusive")
        if source_sampling is None:
            source_sampling = TrajectorySampling(
                num_poses=np.asarray(values).shape[0],
                interval_length=0.5 if source_interval is None else source_interval,
            )
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"trajectory poses must have shape [P, 3], got {poses.shape}")
    if poses.shape[0] != source_sampling.num_poses:
        raise ValueError(
            f"trajectory has {poses.shape[0]} poses but its sampling declares "
            f"{source_sampling.num_poses}"
        )
    # The trajectory starts at the ego origin, which the model does not emit.
    source_times = np.arange(poses.shape[0] + 1) * float(source_sampling.interval_length)
    source = np.vstack([np.zeros((1, 3)), poses])
    target_times = np.arange(1, num_frames + 1) * target_interval

    if target_times.size and target_times[-1] > source_times[-1] + 1e-9:
        raise ValueError(
            "trajectory does not cover the metric horizon: "
            f"source={source_times[-1]:g}s, target={target_times[-1]:g}s"
        )

    x = np.interp(target_times, source_times, source[:, 0])
    y = np.interp(target_times, source_times, source[:, 1])
    heading = np.interp(target_times, source_times, np.unwrap(source[:, 2]))
    dense = np.stack([x, y, np.cos(heading), np.sin(heading)], axis=-1)
    return torch.from_numpy(dense).float().unsqueeze(0)


def compute_tier4_metrics(
    trajectory,
    scene,
    config: Optional[RewardConfig] = None,
) -> Dict[str, float]:
    """Score one trajectory with the TIER IV metric family.

    Every term is computed independently and a term whose inputs are missing is
    omitted rather than defaulted.  A silent 1.0 for "no red light data" reads
    identically to "obeyed the light", which is the one thing these numbers
    exist to distinguish.

    :param trajectory: the planned :class:`~t4_e2e_devkit.common.dataclasses.Trajectory`.
    :param scene: the :class:`~t4_e2e_devkit.common.dataclasses.T4Scene` it is scored against.
    :param config: thresholds; TIER IV's defaults otherwise.
    :return: named metric values.
    """
    from t4_e2e_devkit.evaluation.tier4_metrics.subscores import (
        compute_feasibility_score_batch,
        compute_kinematic_gate,
        compute_lane_departure_penalty,
        compute_red_light_score_batch,
    )

    config = config or RewardConfig()
    frame = scene.current_frame
    if frame.map_tensors is None:
        return {}

    ego_trajs = densify_trajectory(trajectory)
    ego_shape = torch.from_numpy(frame.ego_status.ego_shape.as_array()).float()
    data: Dict[str, torch.Tensor] = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32)).unsqueeze(0)
        for name, value in frame.map_tensors.as_dict().items()
    }

    metrics: Dict[str, float] = {}

    metrics["kinematic_gate"] = float(
        compute_kinematic_gate(ego_trajs, config, ego_shape)[0]
    )

    feasibility, _ = compute_feasibility_score_batch(ego_trajs, ego_shape, data, config)
    metrics["feasibility"] = float(feasibility[0])

    metrics["red_light"] = float(compute_red_light_score_batch(ego_trajs, data, config)[0])

    crossing_gate, near_frac, wide_frac, _, cont_penalty = compute_lane_departure_penalty(
        ego_trajs, ego_shape, data, config=config
    )
    metrics["lane_departure_gate"] = float(crossing_gate[0])
    metrics["lane_departure_near_frac"] = float(near_frac[0])
    metrics["lane_departure_wide_frac"] = float(wide_frac[0])
    metrics["lane_departure_cont"] = float(cont_penalty[0])

    return metrics


def aggregate_tier4_metrics(
    results: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    """Average the T4 metric family without touching PDM-Score.

    A metric missing from one window is omitted from that metric's denominator;
    this keeps unavailable sensor/map fields distinct from a measured zero.
    """

    if not results:
        return {"num_scenes": 0.0}
    keys = sorted({key for result in results for key in result})
    report: Dict[str, float] = {"num_scenes": float(len(results))}
    for key in keys:
        values = [float(result[key]) for result in results if key in result]
        report[key] = float(np.mean(values))
    return report
