"""Build a data list from a T4 dataset root.

This is the only place a window is accepted or rejected.  Keeping the decision
here rather than in the loader is what lets one dataset serve models with
different tolerances: a model that cannot handle a dropped LiDAR frame passes
``--max-window-gap-frames 0``, one that can leaves it out, and both read the
same scenes through the same reader.

It also means every exclusion is recorded.  The written manifest reports what
each gate cost -- per label, per scene, per camera -- so a short list can be
explained later without rebuilding it.  A run whose list does not say why it is
short is a run nobody can reproduce.

Usage::

    python -m t4_e2e_devkit.script.build_datalist \\
        --root /path/to/t4_dataset \\
        --glob 'prd_jt/*/*/*' \\
        --out /path/to/t4_train.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from t4_e2e_devkit.common.constants import (
    DEFAULT_CENTER_STRIDE,
    FUTURE_FRAMES,
    PAST_FRAMES,
    T4_DEFAULT_CAMERA_NAMES,
)
from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.datalist import DataList, is_e2e_scene_path
from t4_e2e_devkit.dataset.window import T4WindowBuilder

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    :param argv: argument vector; ``sys.argv`` by default.
    :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="T4 dataset root")
    parser.add_argument(
        "--glob",
        default="prd_jt/*/*/*",
        help="scene directory glob, relative to --root (default: %(default)s)",
    )
    parser.add_argument("--out", required=True, type=Path, help="output data-list path")
    parser.add_argument("--date", action="append", default=None, help="keep only these dates; repeatable")
    parser.add_argument("--vehicle", action="append", default=None, help="keep only these vehicles; repeatable")
    parser.add_argument(
        "--history-frames", type=int, default=PAST_FRAMES,
        help="history frames including the current one (default: %(default)s)",
    )
    parser.add_argument(
        "--future-frames", type=int, default=FUTURE_FRAMES,
        help="recorded future frames per window (default: %(default)s)",
    )
    parser.add_argument(
        "--center-stride", type=int, default=DEFAULT_CENTER_STRIDE,
        help="source frames between consecutive centres (default: %(default)s = 0.5 s)",
    )
    parser.add_argument(
        "--camera-names", nargs="+", default=list(T4_DEFAULT_CAMERA_NAMES),
        help="camera register a scene must expose (default: the five wide views)",
    )
    parser.add_argument(
        "--require-cameras", nargs="*", default=None,
        help="cameras a window must have an image for at its centre; "
             "'none' requires nothing, omitted means --camera-names",
    )
    parser.add_argument(
        "--max-window-gap-frames", type=int, default=None,
        help="drop centres whose window misses more than N key frames; "
             "omit to accept any gap, 0 to require a strictly uniform window",
    )
    parser.add_argument(
        "--require-pdm-progress", type=Path, default=None, metavar="CACHE",
        help="drop centres with no valid PDM-Closed label in this cache",
    )
    parser.add_argument(
        "--drivable-area-buffer-m", type=float, default=0.0,
        help="PDM roadblock buffer the cache was built with (default: %(default)s)",
    )
    parser.add_argument("--max-scenes", type=int, default=None, help="stop after N scenes")
    parser.add_argument("--limit-per-scene", type=int, default=None, help="keep at most N rows per scene")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> DataList:
    """
    Enumerate windows and apply the gates.
    :param args: parsed arguments.
    :return: the built data list.
    """
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"T4 root is not a directory: {root}")

    scene_filter = SceneFilter(
        num_history_frames=args.history_frames,
        num_future_frames=args.future_frames,
        frame_interval=args.center_stride,
    )

    require_cameras: Optional[List[str]]
    if args.require_cameras is None:
        require_cameras = list(args.camera_names)
    elif len(args.require_cameras) == 1 and args.require_cameras[0].lower() == "none":
        require_cameras = []
    else:
        require_cameras = list(args.require_cameras)

    reader_config: Dict[str, Any] = {"camera_names": list(args.camera_names)}
    if args.require_pdm_progress is not None:
        reader_config.update(
            {
                "t4_load_oracle_targets": True,
                "t4_pdm_reference_cache_dir": str(args.require_pdm_progress.resolve()),
                "t4_drivable_area_buffer_m": args.drivable_area_buffer_m,
            }
        )

    rows: List[tuple] = []
    scenes: List[str] = []
    dropped: Dict[str, int] = {
        "scene_outside_e2e_subtree": 0,
        "scene_unreadable": 0,
        "scene_without_cameras": 0,
        "scene_too_short": 0,
        "rows_dropped_by_date": 0,
        "rows_dropped_by_vehicle": 0,
        "rows_dropped_by_window_gap": 0,
        "rows_dropped_by_camera": 0,
        "rows_dropped_by_pdm_progress": 0,
        "rows_dropped_by_window_error": 0,
    }
    scene_errors: Dict[str, str] = {}

    candidates = sorted(path for path in root.glob(args.glob) if path.is_dir())
    logger.info("%d scene directories match %s", len(candidates), args.glob)

    for scene_path in candidates:
        if args.max_scenes is not None and len(scenes) >= args.max_scenes:
            break
        relative = str(scene_path.relative_to(root))

        if not is_e2e_scene_path(relative):
            dropped["scene_outside_e2e_subtree"] += 1
            continue
        if args.date and not any(date in relative for date in args.date):
            dropped["rows_dropped_by_date"] += 1
            continue
        if args.vehicle and not any(vehicle in relative for vehicle in args.vehicle):
            dropped["rows_dropped_by_vehicle"] += 1
            continue

        try:
            builder = T4WindowBuilder(
                scene_path, root, scene_filter=scene_filter, reader_config=reader_config
            )
        except ValueError as error:
            # A missing camera directory is a whole-scene exclusion, not a
            # per-window one, and is counted separately so it does not look
            # like a corrupt scene.
            key = "scene_without_cameras" if "camera" in str(error).lower() else "scene_unreadable"
            dropped[key] += 1
            scene_errors[relative] = repr(error)
            continue
        except (OSError, KeyError) as error:
            dropped["scene_unreadable"] += 1
            scene_errors[relative] = repr(error)
            continue

        try:
            scene_rows = _scene_rows(builder, relative, require_cameras, args, dropped)
        finally:
            builder.close()

        if scene_rows:
            rows.extend(scene_rows)
            scenes.append(relative)
        else:
            dropped["scene_too_short"] += 1

        if len(scenes) % 50 == 0 and scenes:
            logger.info("%d scenes, %d rows", len(scenes), len(rows))

    manifest = {
        "glob": args.glob,
        "dates": args.date,
        "vehicles": args.vehicle,
        "camera_names": list(args.camera_names),
        "history_frames": args.history_frames,
        "gt_future_frames": args.future_frames,
        "center_stride": args.center_stride,
        # Recorded apart from camera_names because they are allowed to differ:
        # what the model reads and what a window must have are two decisions.
        "filter": {
            "require_cameras": sorted(require_cameras),
            "max_window_gap_frames": args.max_window_gap_frames,
            "require_pdm_progress": (
                str(args.require_pdm_progress) if args.require_pdm_progress else None
            ),
            "drivable_area_buffer_m": args.drivable_area_buffer_m,
            **{key: value for key, value in dropped.items() if value},
            "scenes_with_errors": scene_errors,
        },
    }
    return DataList(root=root, rows=rows, manifest=manifest)


def _scene_rows(
    builder: T4WindowBuilder,
    relative: str,
    require_cameras: Sequence[str],
    args: argparse.Namespace,
    dropped: Dict[str, int],
) -> List[tuple]:
    """Enumerate the accepted centres of one scene."""
    presence = builder.reader.scalars.get("cam_presence")
    valid_mask = builder.reader.scalars.get("valid_mask")
    camera_slots = [
        builder.reader.camera_indices[builder.reader.camera_names.index(name)]
        for name in require_cameras
        if name in builder.reader.camera_names
    ]

    rows: List[tuple] = []
    for center in builder.valid_centers():
        if args.limit_per_scene is not None and len(rows) >= args.limit_per_scene:
            break

        first = center - args.history_frames + 1
        last = center + args.future_frames

        if args.max_window_gap_frames is not None and valid_mask is not None:
            gaps = int((~valid_mask[first : last + 1]).sum())
            if gaps > args.max_window_gap_frames:
                dropped["rows_dropped_by_window_gap"] += 1
                continue

        if camera_slots and presence is not None:
            if not all(bool(presence[center, slot]) for slot in camera_slots):
                dropped["rows_dropped_by_camera"] += 1
                continue

        if args.require_pdm_progress is not None:
            if builder.read_pdm_progress(center) is None:
                dropped["rows_dropped_by_pdm_progress"] += 1
                continue

        rows.append((relative, int(center)))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    :param argv: argument vector.
    :return: process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    data_list = build(args)
    if not data_list.rows:
        logger.error("no windows survived the filters; nothing written")
        return 1
    path = data_list.write(args.out)
    print(
        json.dumps(
            {
                "out": str(path),
                "n_scenes": len(data_list.scene_dirs),
                "n_rows": len(data_list.rows),
                "dropped": {
                    key: value
                    for key, value in data_list.manifest["filter"].items()
                    if isinstance(value, int) and value
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
