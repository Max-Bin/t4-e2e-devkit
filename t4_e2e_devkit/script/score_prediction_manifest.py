"""Score an external model's ``t4-e2e.predictions`` manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from t4_e2e_devkit.evaluation.prediction_scoring import score_prediction_manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments, score a manifest and print its report."""

    parser = argparse.ArgumentParser(
        prog="t4e2e score-manifest",
        description="Score model predictions without importing the model repository.",
    )
    parser.add_argument("data_list", type=Path, help="T4 data-list JSON")
    parser.add_argument("--predictions", required=True, type=Path, help="prediction JSONL")
    parser.add_argument("--output-dir", required=True, type=Path, help="report directory")
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="optional ordered metric subset; omit for the complete version",
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--device", default=None, help="CUDA device, for example cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument(
        "--scene-cache-size",
        type=int,
        default=0,
        help="open scene builders to retain; 0 disables the scene cache",
    )
    parser.add_argument(
        "--no-per-window",
        action="store_true",
        help="write only aggregate.json, not per_window.csv",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = score_prediction_manifest(
        data_list_path=args.data_list,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        version=args.version,
        metric_names=args.metrics,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        max_scenes=args.max_scenes,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        scene_cache_size=args.scene_cache_size,
        write_per_window=not args.no_per_window,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
