"""Generic distributed validation scoring for model training repositories."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from t4_e2e_devkit.evaluation.prediction_manifest import PredictionManifestWriter
from t4_e2e_devkit.evaluation.prediction_scoring import (
    REPORT_METRIC_KEYS,
    score_prediction_manifest,
)

LOG = logging.getLogger(__name__)

REPORT_TO_LOG = {
    "score": "pdms",
    "nc": "nc",
    "dac": "dac",
    "ddc": "ddc",
    "tlc": "tlc",
    "ttc": "ttc",
    "ep": "ep",
    "lk": "lk",
    "comfort": "comfort",
    "ec": "ec",
}


class OfficialDevkitScoreCallback(pl.Callback):
    """Score model predictions through the shared devkit after validation.

    The model only needs to populate these module attributes during validation:
    ``_official_score_predictions`` (``[N, T, 3|4]`` arrays) and
    ``_official_score_keys`` (``(scene_dir, center_frame)`` pairs).  All
    manifest, distributed scoring, report merging and logger writes stay here.
    """

    def __init__(
        self,
        data_list: str | Path,
        output_dir: str | Path,
        *,
        version: str = "v2",
        metric_names: tuple[str, ...] | None = None,
        interval_seconds: float = 0.1,
        batch_size: int = 128,
        scene_cache_size: int | None = 0,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("official devkit batch_size must be positive")
        if scene_cache_size is not None and int(scene_cache_size) < 0:
            raise ValueError("official devkit scene_cache_size must be non-negative or None")
        if not math.isfinite(float(interval_seconds)) or interval_seconds <= 0.0:
            raise ValueError("official devkit trajectory interval must be positive")
        self.data_list = Path(data_list).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.version = str(version)
        self.metric_names = metric_names
        self.interval_seconds = float(interval_seconds)
        self.batch_size = int(batch_size)
        self.scene_cache_size = 0 if scene_cache_size is None else int(scene_cache_size)
        self._active = False

    @staticmethod
    def _barrier() -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _world_size() -> int:
        import torch.distributed as dist

        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def _status_path(self, epoch: int, name: str) -> Path:
        return self.output_dir / f"epoch-{epoch:04d}" / name

    @staticmethod
    def _write_status(path: Path, message: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(message, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _wait_for_status(
        path: Path, *, failure_path: Path | None = None, timeout_s: float = 3600.0
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            if path.is_file():
                return True
            if failure_path is not None and failure_path.is_file():
                return False
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {path}")
            time.sleep(0.25)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        active = not bool(getattr(trainer, "sanity_checking", False))
        self._active = active
        epoch = int(getattr(trainer, "current_epoch", 0))
        rank = int(getattr(trainer, "global_rank", 0))
        if active and rank == 0:
            epoch_dir = self.output_dir / f"epoch-{epoch:04d}"
            for name in ("score-ready", "score-complete", "score-failed"):
                self._status_path(epoch, name).unlink(missing_ok=True)
            for index in range(self._world_size()):
                for suffix in ("complete", "failed"):
                    self._status_path(
                        epoch, f"score-rank-{index:05d}.{suffix}"
                    ).unlink(missing_ok=True)
            (epoch_dir / "predictions.jsonl").unlink(missing_ok=True)
        if active:
            self._barrier()

        pl_module._official_score_active = active
        pl_module._official_score_predictions = []
        pl_module._official_score_keys = []
        pl_module._official_score_error = None
        pl_module._official_score_num_poses = None
        pl_module._official_score_pose_dim = None
        pl_module._official_score_interval_seconds = self.interval_seconds

    def _rank_path(self, epoch: int, rank: int) -> Path:
        return self.output_dir / f"epoch-{epoch:04d}" / f"predictions-rank-{rank:05d}.npz"

    @staticmethod
    def _write_rank_predictions(path: Path, pl_module, interval_seconds: float) -> None:
        import numpy as np

        predictions = list(getattr(pl_module, "_official_score_predictions", []))
        keys = list(getattr(pl_module, "_official_score_keys", []))
        if len(predictions) != len(keys):
            raise RuntimeError(
                "official devkit prediction/key counts disagree: "
                f"predictions={len(predictions)} keys={len(keys)}"
            )
        poses = np.asarray(predictions, dtype=np.float32)
        if not len(predictions):
            poses = np.empty(
                (
                    0,
                    int(getattr(pl_module, "_official_score_num_poses", 0) or 0),
                    int(getattr(pl_module, "_official_score_pose_dim", 0) or 0),
                ),
                dtype=np.float32,
            )
        if poses.ndim != 3 or poses.shape[0] != len(keys) or (
            len(predictions) and (poses.shape[1] < 1 or poses.shape[2] not in (3, 4))
        ):
            raise RuntimeError(f"official devkit predictions must be [N, T, 3|4], got {poses.shape}")
        if not np.isfinite(poses).all():
            raise RuntimeError("official devkit predictions contain NaN or Inf")
        num_poses = int(poses.shape[1]) if len(predictions) else int(getattr(pl_module, "_official_score_num_poses", 0) or 0)
        pose_dim = int(poses.shape[2]) if len(predictions) else int(getattr(pl_module, "_official_score_pose_dim", 0) or 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene=np.asarray([key[0] for key in keys], dtype=str),
            center=np.asarray([key[1] for key in keys], dtype=np.int64),
            poses=poses,
            num_poses=np.asarray([num_poses], dtype=np.int64),
            pose_dim=np.asarray([pose_dim], dtype=np.int64),
            interval_seconds=np.asarray([interval_seconds], dtype=np.float64),
            error=np.asarray([getattr(pl_module, "_official_score_error", "") or ""], dtype=str),
        )

    def _merge_predictions(self, paths: list[Path], manifest_path: Path) -> int:
        import numpy as np

        records: dict[tuple[str, int], np.ndarray] = {}
        errors: list[str] = []
        num_poses: int | None = None
        pose_dim: int | None = None
        interval: float | None = None
        for path in paths:
            if not path.is_file():
                errors.append(f"missing rank prediction file: {path}")
                continue
            with np.load(path, allow_pickle=False) as payload:
                error = str(payload["error"][0]) if len(payload["error"]) else ""
                if error:
                    errors.append(error)
                scenes = payload["scene"].tolist()
                centers = payload["center"].tolist()
                poses = payload["poses"]
                rank_num_poses = int(payload["num_poses"][0])
                rank_pose_dim = int(payload["pose_dim"][0])
                rank_interval = float(payload["interval_seconds"][0])
                if len(scenes) and (
                    rank_num_poses < 1
                    or rank_pose_dim not in (3, 4)
                    or not math.isfinite(rank_interval)
                    or rank_interval <= 0.0
                ):
                    errors.append(f"invalid trajectory metadata in {path}")
                    continue
                if len(scenes) != len(centers) or poses.ndim != 3 or poses.shape[0] != len(scenes):
                    errors.append(f"malformed rank prediction file: {path}")
                    continue
                if len(scenes) and poses.shape[1:] != (rank_num_poses, rank_pose_dim):
                    errors.append(f"rank trajectory shape/metadata mismatch: {path}")
                    continue
                if len(scenes):
                    if num_poses is None:
                        num_poses, pose_dim, interval = rank_num_poses, rank_pose_dim, rank_interval
                    elif num_poses != rank_num_poses or pose_dim != rank_pose_dim or not math.isclose(interval or 0.0, rank_interval, rel_tol=0.0, abs_tol=1e-9):
                        errors.append(f"inconsistent trajectory metadata across ranks: {path}")
                        continue
                for scene, center, trajectory in zip(scenes, centers, poses, strict=True):
                    key = (str(scene), int(center))
                    if key in records:
                        errors.append(f"duplicate prediction key across ranks: {key!r}")
                        continue
                    records[key] = np.asarray(trajectory)
        if errors:
            raise RuntimeError("; ".join(errors))
        if not records or num_poses is None or interval is None:
            raise RuntimeError("official devkit validation produced no predictions")

        with PredictionManifestWriter(
            manifest_path,
            data_list=self.data_list,
            num_poses=num_poses,
            interval_seconds=interval,
        ) as writer:
            for (scene, center), trajectory in records.items():
                writer.write(scene, center, trajectory)
        return len(records)

    def _score_on_rank(self, epoch: int, rank: int, world_size: int, manifest_path: Path) -> dict:
        import torch

        return score_prediction_manifest(
            data_list_path=self.data_list,
            predictions_path=manifest_path,
            output_dir=self.output_dir / f"epoch-{epoch:04d}" / f"score-rank-{rank:05d}",
            version=self.version,
            metric_names=self.metric_names,
            backend="gpu",
            device=f"cuda:{torch.cuda.current_device()}",
            batch_size=self.batch_size,
            shard_index=rank,
            num_shards=world_size,
            scene_cache_size=self.scene_cache_size,
            write_per_window=False,
        )

    def _merge_reports(self, epoch: int, world_size: int) -> dict[str, Any]:
        epoch_dir = self.output_dir / f"epoch-{epoch:04d}"
        reports = []
        for rank in range(world_size):
            path = epoch_dir / f"score-rank-{rank:05d}" / "aggregate.json"
            if not path.is_file():
                raise RuntimeError(f"missing rank score report: {path}")
            reports.append(json.loads(path.read_text(encoding="utf-8")))

        merged: dict[str, Any] = {
            "num_scenes": sum(float(report.get("num_scenes", 0.0)) for report in reports),
            "distributed_shards": float(world_size),
        }
        counts: dict[str, int] = {}
        for key in REPORT_METRIC_KEYS:
            total = 0.0
            count = 0
            for report in reports:
                value = report.get(key)
                metric_count = int(report.get("_metric_counts", {}).get(key, 0))
                if isinstance(value, (int, float)) and not isinstance(value, bool) and metric_count > 0:
                    total += float(value) * metric_count
                    count += metric_count
            if count:
                merged[key] = total / count
                counts[key] = count
        for key in (
            "scorer",
            "version",
            "metric_names",
            "backend",
            "data_list_sha256",
            "prediction_manifest_sha256",
            "prediction_manifest_format",
            "prediction_manifest_version",
            "trajectory_num_poses",
            "trajectory_interval_seconds",
            "sensor_config",
            "future_lidar_read",
        ):
            if key in reports[0]:
                merged[key] = reports[0][key]
        merged["_metric_counts"] = counts
        (epoch_dir / "aggregate.json").write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return merged

    @staticmethod
    def _log_report(trainer, epoch: int, report: dict[str, Any]) -> None:
        payload = {
            f"devkit/{REPORT_TO_LOG[key]}": float(report[key])
            for key in REPORT_TO_LOG
            if key in report and isinstance(report[key], (int, float)) and not isinstance(report[key], bool)
        }
        if not payload:
            return
        payload["epoch"] = int(epoch)
        step = int(getattr(trainer, "global_step", epoch))
        for logger in getattr(trainer, "loggers", []) or []:
            log_metrics = getattr(logger, "log_metrics", None)
            if not callable(log_metrics):
                continue
            try:
                log_metrics(payload, step=step)
            except Exception:  # noqa: BLE001 - telemetry must not kill scoring
                LOG.exception("official devkit report logger failed")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._active:
            return
        epoch = int(trainer.current_epoch)
        rank = int(getattr(trainer, "global_rank", 0))
        world_size = self._world_size()
        paths = [self._rank_path(epoch, index) for index in range(world_size)]
        manifest_path = self.output_dir / f"epoch-{epoch:04d}" / "predictions.jsonl"
        ready = self._status_path(epoch, "score-ready")
        failed = self._status_path(epoch, "score-failed")
        complete = self._status_path(epoch, "score-complete")
        try:
            try:
                self._write_rank_predictions(paths[rank], pl_module, self.interval_seconds)
            except Exception as error:  # noqa: BLE001
                LOG.exception("official prediction export failed on rank %d", rank)
                pl_module._official_score_error = repr(error)
                self._write_rank_predictions(paths[rank], pl_module, self.interval_seconds)
            self._barrier()

            if rank == 0:
                try:
                    count = self._merge_predictions(paths, manifest_path)
                    self._write_status(ready, str(count))
                except Exception as error:  # noqa: BLE001
                    LOG.exception("official prediction merge failed at epoch %d", epoch)
                    self._write_status(failed, repr(error))

            if self._wait_for_status(ready, failure_path=failed):
                rank_complete = self._status_path(epoch, f"score-rank-{rank:05d}.complete")
                rank_failed = self._status_path(epoch, f"score-rank-{rank:05d}.failed")
                try:
                    self._score_on_rank(epoch, rank, world_size, manifest_path)
                    self._write_status(rank_complete, "ok")
                except Exception as error:  # noqa: BLE001
                    LOG.exception("official GPU scoring failed on rank %d", rank)
                    self._write_status(rank_failed, repr(error))

                if rank == 0:
                    failures = []
                    for index in range(world_size):
                        rank_complete = self._status_path(epoch, f"score-rank-{index:05d}.complete")
                        rank_failed = self._status_path(epoch, f"score-rank-{index:05d}.failed")
                        if not self._wait_for_status(rank_complete, failure_path=rank_failed):
                            failures.append(rank_failed.read_text(encoding="utf-8"))
                    if failures:
                        self._write_status(failed, "; ".join(failures))
                    else:
                        try:
                            report = self._merge_reports(epoch, world_size)
                            self._log_report(trainer, epoch, report)
                            self._write_status(complete, "ok")
                        except Exception as error:  # noqa: BLE001
                            LOG.exception("official report merge failed at epoch %d", epoch)
                            self._write_status(failed, repr(error))
        except Exception as error:  # noqa: BLE001
            LOG.exception("official scoring coordination failed on rank %d", rank)
            if rank == 0:
                self._write_status(failed, repr(error))
            else:
                self._write_status(
                    self._status_path(epoch, f"score-rank-{rank:05d}.failed"), repr(error)
                )

        if rank != 0:
            self._wait_for_status(complete, failure_path=failed)
        self._barrier()
        self._active = False
        pl_module._official_score_active = False
        pl_module._official_score_predictions = []
        pl_module._official_score_keys = []
        pl_module._official_score_error = None
        pl_module._official_score_num_poses = None
        pl_module._official_score_pose_dim = None
        pl_module._official_score_interval_seconds = None


__all__ = ["OfficialDevkitScoreCallback"]
