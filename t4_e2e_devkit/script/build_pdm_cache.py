"""Build the PDM-Closed reference cache.

The cache holds one number per window that nothing else can supply:
``pdm_progress``, the ego-progress denominator.  PDM-Closed generates its
proposals from the T4 route, IDM policies, the current boxes and the scene's own
vehicle shape; the selected proposal is replayed against the recorded window and
its raw progress is gated by NC x DAC.  That is scene-level data -- it cannot be
reconstructed from a model's own trajectory, and substituting the demonstrated
future endpoint changes what EP measures rather than approximating it.

This cache is optional for GPU scoring, which computes the same reference online
on CUDA without reading or writing a cache.  Build it only for an explicit
CPU/offline reference run or for an audit artifact.  The cache is immutable and
signature-checked: a cache built with different PDM parameters is rejected
rather than silently mixed with freshly computed labels.

Usage::

    python -m t4_e2e_devkit.script.build_pdm_cache \\
        --data-list /path/to/t4_train.json \\
        --cache-root /path/to/t4-pdm-reference-cache \\
        --jobs 16

Put an offline cache on node-local NVMe, not on shared storage. Re-running is
resumable -- an existing scene is reported ``already-present`` and skipped
unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    :param argv: argument vector; ``sys.argv`` by default.
    :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-list", type=Path, help="build only the scenes this list uses")
    source.add_argument("--root", type=Path, help="build every scene under this T4 root")

    parser.add_argument("--glob", default="prd_jt/*/*/*", help="scene glob when using --root")
    parser.add_argument("--cache-root", required=True, type=Path, help="cache destination")
    parser.add_argument(
        "--drivable-area-buffer-m", type=float, default=0.0,
        help="PDM roadblock buffer; part of the cache signature (default: %(default)s)",
    )
    parser.add_argument("--frame-cache-root", type=Path, default=None, help="decoded frame cache")
    parser.add_argument("--jobs", type=int, default=1, help="parallel scene workers")
    parser.add_argument("--overwrite", action="store_true", help="rebuild scenes already cached")
    parser.add_argument("--max-scenes", type=int, default=None, help="stop after N scenes")
    return parser.parse_args(argv)


def _scene_centers(args: argparse.Namespace) -> Dict[str, Optional[List[int]]]:
    """Resolve which scenes to build, and which centres within each."""
    if args.data_list is not None:
        from t4_e2e_devkit.dataset.datalist import load_data_list

        data_list = load_data_list(args.data_list)
        centers: Dict[str, Optional[List[int]]] = {}
        for scene, center in data_list:
            centers.setdefault(scene, []).append(int(center))
        return {"__root__": data_list.root, **{k: sorted(set(v)) for k, v in centers.items()}}

    root = args.root.resolve()
    scenes = sorted(path for path in root.glob(args.glob) if path.is_dir())
    # None means "every valid centre of the scene".
    return {"__root__": root, **{str(path.relative_to(root)): None for path in scenes}}


def _build_one(
    scene: str,
    root: str,
    cache_root: str,
    centers: Optional[List[int]],
    buffer_m: float,
    frame_cache_root: Optional[str],
    overwrite: bool,
) -> Dict[str, Any]:
    """Build one scene's cache.  Runs in a worker process."""
    from t4_e2e_devkit.evaluation.reference.pdm_closed import (
        T4PDMReferenceConfig,
        build_t4_pdm_reference_cache,
    )

    try:
        result = build_t4_pdm_reference_cache(
            Path(root) / scene,
            root,
            cache_root,
            centers=centers,
            config=T4PDMReferenceConfig(roadblock_buffer_m=buffer_m),
            frame_cache_root=frame_cache_root,
            overwrite=overwrite,
        )
        return {"scene": scene, "status": "ok", **{k: v for k, v in result.items() if not hasattr(v, "shape")}}
    except Exception as error:  # noqa: BLE001
        # One unbuildable scene must not abandon the other few thousand; the
        # failure is reported and the run continues.
        return {"scene": scene, "status": "failed", "error": repr(error)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    :param argv: argument vector.
    :return: process exit code; non-zero when any scene failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    resolved = _scene_centers(args)
    root = str(resolved.pop("__root__"))
    scenes = list(resolved)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    logger.info("building PDM reference cache for %d scene(s) with %d job(s)", len(scenes), args.jobs)

    task_args = [
        (
            scene,
            root,
            str(args.cache_root.resolve()),
            resolved[scene],
            args.drivable_area_buffer_m,
            str(args.frame_cache_root) if args.frame_cache_root else None,
            args.overwrite,
        )
        for scene in scenes
    ]

    results: List[Dict[str, Any]] = []
    if args.jobs <= 1:
        for task in task_args:
            results.append(_build_one(*task))
            logger.info("%s: %s", results[-1]["scene"], results[-1]["status"])
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_build_one, *task): task[0] for task in task_args}
            for done in as_completed(futures):
                results.append(done.result())
                logger.info(
                    "[%d/%d] %s: %s",
                    len(results), len(task_args), results[-1]["scene"], results[-1]["status"],
                )

    failed = [result for result in results if result["status"] == "failed"]
    print(
        json.dumps(
            {
                "cache_root": str(args.cache_root),
                "n_scenes": len(results),
                "n_failed": len(failed),
                "failures": failed[:10],
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
