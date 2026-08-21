"""Render one scene's planning video through the Python API.

Example::

    uv run python examples/render_planning_video.py \
        --data-list results/val.datalist.json \
        --scene prd_jt/scene/date/time \
        --manifest baseline=results/model/predictions.jsonl \
        --out results/visualization/scene.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from t4_e2e_devkit.dataset.datalist import load_data_list
from t4_e2e_devkit.evaluation.prediction_manifest import load_prediction_manifest
from t4_e2e_devkit.visualization import render_scene_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-list", required=True, type=Path)
    parser.add_argument("--scene", required=True, help="relative scene directory from the list")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="prediction manifest to overlay, repeatable",
    )
    parser.add_argument("--out", required=True, type=Path, help="output .mp4 path")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--no-lidar", action="store_true")
    args = parser.parse_args()

    manifests = {}
    for entry in args.manifest:
        label, separator, path = entry.partition("=")
        if not separator or not label or not path:
            raise SystemExit(f"--manifest expects LABEL=PATH, got {entry!r}")
        manifests[label] = load_prediction_manifest(path)

    data_list = load_data_list(args.data_list)
    print(
        render_scene_video(
            data_list,
            args.scene,
            args.out,
            manifests,
            fps=args.fps,
            lidar=not args.no_lidar,
        )
    )


if __name__ == "__main__":
    main()
