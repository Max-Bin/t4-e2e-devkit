"""Score shared prediction manifests and write reproducible reports."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

LOG = logging.getLogger(__name__)

# Compact names are stable across model repositories and intentionally
# independent of any logging backend.
METRIC_TO_REPORT = {
    "score": "score",
    "no_at_fault_collisions": "nc",
    "drivable_area_compliance": "dac",
    "driving_direction_compliance": "ddc",
    "traffic_light_compliance": "tlc",
    "time_to_collision_within_bound": "ttc",
    "ego_progress": "ep",
    "lane_keeping": "lk",
    "history_comfort": "comfort",
    "extended_comfort": "ec",
}
REPORT_METRIC_KEYS = tuple(dict.fromkeys(METRIC_TO_REPORT.values()))


def _load_devkit(root: str | Path | None = None):
    if root is not None:
        import sys

        root = Path(root).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from t4_e2e_devkit.common.dataclasses import SceneFilter, SensorConfig, Trajectory
    from t4_e2e_devkit.dataset import T4Dataset
    from t4_e2e_devkit.dataset.datalist import load_data_list
    from t4_e2e_devkit.evaluation.navsim_score import (
        T4NavSimScorer,
        T4NavSimScorerConfig,
        aggregate_navsim_results,
    )
    from t4_e2e_devkit.evaluation.prediction_manifest import (
        data_list_sha256,
        file_sha256,
        load_prediction_manifest,
        validate_prediction_keys,
    )
    from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    return {
        "SceneFilter": SceneFilter,
        "SensorConfig": SensorConfig,
        "Trajectory": Trajectory,
        "TrajectorySampling": TrajectorySampling,
        "T4Dataset": T4Dataset,
        "T4NavSimScorer": T4NavSimScorer,
        "T4NavSimScorerConfig": T4NavSimScorerConfig,
        "aggregate_navsim_results": aggregate_navsim_results,
        "data_list_sha256": data_list_sha256,
        "file_sha256": file_sha256,
        "load_data_list": load_data_list,
        "load_prediction_manifest": load_prediction_manifest,
        "validate_prediction_keys": validate_prediction_keys,
    }


def _select_data_list(data_list: Any, max_rows: int | None, max_scenes: int | None) -> Any:
    if max_scenes is not None:
        if max_scenes < 1:
            raise ValueError("max_scenes must be positive")
        data_list = data_list.filtered(scene_dirs=data_list.scene_dirs[:max_scenes])
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        data_list = data_list.filtered(max_rows=max_rows)
    return data_list


def _shard_data_list(data_list: Any, shard_index: int | None, num_shards: int | None) -> Any:
    if shard_index is None and num_shards is None:
        return data_list
    if shard_index is None or num_shards is None:
        raise ValueError("shard_index and num_shards must be provided together")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    total = len(data_list)
    start = total * shard_index // num_shards
    end = total * (shard_index + 1) // num_shards
    return type(data_list)(
        root=data_list.root,
        rows=list(data_list.rows[start:end]),
        manifest={
            **data_list.manifest,
            "runtime_shard": {
                "shard_index": shard_index,
                "num_shards": num_shards,
                "rows_before": total,
                "rows_after": end - start,
            },
        },
        path=data_list.path,
    )


def _report_values(values: Mapping[str, float]) -> dict[str, float]:
    return {
        METRIC_TO_REPORT[name]: float(value)
        for name, value in values.items()
        if name in METRIC_TO_REPORT
    }


def _write_per_window(path: Path, results: Sequence[Any]) -> None:
    keys = [
        key
        for key in REPORT_METRIC_KEYS
        if any(key in _report_values(result.values) for result in results)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", *keys])
        for result in results:
            values = _report_values(result.values)
            writer.writerow(
                [result.token or ""]
                + ["" if key not in values else f"{values[key]:.8f}" for key in keys]
            )


def score_prediction_manifest(
    *,
    data_list_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    devkit_root: str | Path | None = None,
    version: str = "v2",
    metric_names: Sequence[str] | None = None,
    backend: str = "gpu",
    device: str | None = None,
    batch_size: int = 32,
    max_rows: int | None = None,
    max_scenes: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    scene_cache_size: int | None = 0,
    write_per_window: bool = True,
) -> dict[str, Any]:
    """Score every selected manifest row and write ``aggregate.json``."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if scene_cache_size is not None and int(scene_cache_size) < 0:
        raise ValueError("scene_cache_size must be non-negative or None")

    devkit = _load_devkit(devkit_root)
    data_list_path = Path(data_list_path).expanduser().resolve()
    predictions_path = Path(predictions_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    data_list = _select_data_list(
        devkit["load_data_list"](data_list_path), max_rows, max_scenes
    )
    manifest = devkit["load_prediction_manifest"](predictions_path)
    expected_rows = list(data_list)
    devkit["validate_prediction_keys"](manifest.records, expected_rows)

    actual_hash = devkit["data_list_sha256"](data_list_path)
    expected_hash = manifest.header.get("data_list_sha256")
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError(
            "prediction manifest is pinned to a different data list: "
            f"manifest={expected_hash}, requested={actual_hash}"
        )

    trajectory = manifest.header.get("trajectory", {})
    num_poses = int(trajectory["num_poses"])
    interval_seconds = float(trajectory["interval_seconds"])
    if trajectory.get("pose_format", "x_y_heading") != "x_y_heading":
        raise ValueError("prediction manifest pose_format must be 'x_y_heading'")
    if num_poses < 1 or not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
        raise ValueError("prediction manifest contains invalid trajectory sampling")
    sampling = devkit["TrajectorySampling"](
        num_poses=num_poses, interval_length=interval_seconds
    )

    data_list = _shard_data_list(data_list, shard_index, num_shards)
    shard_rows = list(data_list)
    dataset = devkit["T4Dataset"](
        data_list=data_list,
        sensor_config=devkit["SensorConfig"].build_no_sensors(),
        scene_filter=devkit["SceneFilter"](
            num_history_frames=31,
            num_future_frames=80,
            frame_interval=5,
            has_route=True,
        ),
        scene_cache_size=scene_cache_size,
    )
    scorer = devkit["T4NavSimScorer"](
        devkit["T4NavSimScorerConfig"](
            version=version,
            metric_names=metric_names,
            backend=backend,
            device=device,
        )
    )

    results: list[Any] = []
    try:
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            scenes = [dataset[index] for index in range(start, end)]
            trajectories = [
                devkit["Trajectory"](
                    poses=manifest.records[key].poses,
                    trajectory_sampling=sampling,
                )
                for key in shard_rows[start:end]
            ]
            results.extend(scorer.score_batch(trajectories, scenes))
            LOG.info("scored %d/%d windows", len(results), len(dataset))
    finally:
        dataset.close()

    aggregate = devkit["aggregate_navsim_results"]([result.values for result in results])
    report: dict[str, Any] = {
        key: float(value)
        for key, value in aggregate.items()
        if key == "num_scenes" or key in METRIC_TO_REPORT
    }
    for key in tuple(report):
        if key in METRIC_TO_REPORT:
            report[METRIC_TO_REPORT[key]] = report.pop(key)

    counts: dict[str, int] = {}
    for result in results:
        for key in _report_values(result.values):
            counts[key] = counts.get(key, 0) + 1
    report["_metric_counts"] = counts
    report.update(
        {
            "scorer": "pdm",
            "version": str(version).lower().removeprefix("navsim-"),
            "metric_names": list(results[0].metric_names) if results else [],
            "backend": scorer.config.backend,
            "data_list_sha256": actual_hash,
            "prediction_manifest_sha256": devkit["file_sha256"](predictions_path),
            "prediction_manifest_format": manifest.header["format"],
            "prediction_manifest_version": manifest.header["version"],
            "trajectory_num_poses": num_poses,
            "trajectory_interval_seconds": interval_seconds,
            "sensor_config": "no_sensors",
            "future_lidar_read": False,
        }
    )
    if shard_index is not None:
        report["shard_index"] = int(shard_index)
        report["num_shards"] = int(num_shards)
    if write_per_window:
        _write_per_window(output_dir / "per_window.csv", results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["METRIC_TO_REPORT", "REPORT_METRIC_KEYS", "score_prediction_manifest"]
