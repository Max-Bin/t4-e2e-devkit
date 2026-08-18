"""Convert neutral NumPy predictions into the shared JSONL manifest.

The input file contains ``scene``, ``center`` and ``poses`` arrays. See
``examples/README.md`` for the exact shapes and coordinate convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from t4_e2e_devkit.dataset.datalist import load_data_list
from t4_e2e_devkit.evaluation.prediction_manifest import (
    PredictionManifestWriter,
    trajectory_to_poses,
    validate_prediction_keys,
)


def _scene_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="input .npz")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL")
    parser.add_argument("--interval-seconds", required=True, type=float)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as payload:
        required = {"scene", "center", "poses"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"input is missing arrays: {missing}")
        scenes = [_scene_name(value) for value in payload["scene"]]
        centers = np.asarray(payload["center"], dtype=np.int64)
        poses = np.asarray(payload["poses"])

    if poses.ndim != 3 or poses.shape[-1] not in (3, 4):
        raise ValueError(f"poses must have shape [N,T,3|4], got {poses.shape}")
    if len(scenes) != len(centers) or len(scenes) != poses.shape[0]:
        raise ValueError("scene, center and poses arrays must have the same length")
    if not len(scenes):
        raise ValueError("input contains no predictions")

    data_list = load_data_list(args.data_list)
    keys = [(scene, int(center)) for scene, center in zip(scenes, centers, strict=True)]
    if len(set(keys)) != len(keys):
        raise ValueError("input contains duplicate scene/center keys")
    expected = list(data_list)
    validate_prediction_keys(dict.fromkeys(keys), expected)

    with PredictionManifestWriter(
        args.output,
        data_list=args.data_list,
        num_poses=int(poses.shape[1]),
        interval_seconds=args.interval_seconds,
    ) as writer:
        for (scene, center), trajectory in zip(keys, poses, strict=True):
            writer.write(
                scene,
                center,
                trajectory_to_poses(trajectory, num_poses=int(poses.shape[1])),
            )
    print(args.output)


if __name__ == "__main__":
    main()
