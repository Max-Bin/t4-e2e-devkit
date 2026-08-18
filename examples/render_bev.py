"""Render one map-and-annotation BEV window through the Python API.

Example::

    uv run python examples/render_bev.py \
        --scene /data/t4/prd_jt/scene/date/time \
        --root /data/t4 \
        --center 100 \
        --out results/visualization/window.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from t4_e2e_devkit.common.dataclasses import SensorConfig, Trajectory
from t4_e2e_devkit.dataset.window import T4WindowBuilder
from t4_e2e_devkit.evaluation.prediction_manifest import trajectory_to_poses
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from t4_e2e_devkit.visualization import plot_bev_frame, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--center", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--view-range", type=float, default=None)
    parser.add_argument(
        "--prediction",
        type=Path,
        default=None,
        help="optional .npy or .npz file containing [T, 3|4] poses",
    )
    parser.add_argument(
        "--prediction-interval",
        type=float,
        default=0.5,
        help="prediction sampling interval in seconds",
    )
    args = parser.parse_args()

    builder = T4WindowBuilder(
        args.scene.resolve(),
        args.root.resolve(),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    try:
        centers = builder.valid_centers()
        if not centers:
            raise SystemExit("scene is too short for the default T4 window")
        center = args.center if args.center is not None else centers[len(centers) // 2]
        scene = builder.build(center)
        trajectories = {}
        if scene.future_ego_poses is not None:
            trajectories["ground_truth"] = scene.get_future_trajectory()
        if args.prediction is not None:
            loaded = np.load(args.prediction, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    if "poses" not in loaded.files:
                        raise ValueError("prediction archive must contain a 'poses' array")
                    raw_prediction = np.asarray(loaded["poses"])
                finally:
                    loaded.close()
            else:
                raw_prediction = np.asarray(loaded)
            if raw_prediction.ndim != 2 or raw_prediction.shape[1] not in (3, 4):
                raise ValueError(
                    "prediction must have shape [T, 3] or [T, 4], "
                    f"got {raw_prediction.shape}"
                )
            prediction_poses = trajectory_to_poses(
                raw_prediction, num_poses=int(raw_prediction.shape[0])
            )
            trajectories["prediction"] = Trajectory(
                poses=prediction_poses,
                trajectory_sampling=TrajectorySampling(
                    num_poses=int(raw_prediction.shape[0]),
                    interval_length=args.prediction_interval,
                ),
            )
        config = {} if args.view_range is None else {"view_range": args.view_range}
        figure, _ = plot_bev_frame(scene, trajectories, config)
        print(save_figure(figure, args.out))
    finally:
        builder.close()


if __name__ == "__main__":
    main()
