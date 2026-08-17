"""Merge rank manifests produced by a distributed evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from t4_e2e_devkit.evaluation.distributed import merge_worker_manifests


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="t4e2e merge-workers",
        description="Validate and merge portable worker manifests.",
    )
    parser.add_argument("--input-manifest", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="combined manifest JSON")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args(argv)
    results = merge_worker_manifests(
        args.input_manifest,
        output_path=Path(args.output),
        run_id=args.run_id,
        world_size=args.world_size,
        require_complete_world=not args.allow_incomplete,
        require_success=args.require_success,
    )
    print(json.dumps({"num_results": len(results), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
