"""Rank-partitioned batch orchestration and portable worker manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from t4_e2e_devkit.evaluation.worker_pool import (
    WorkerPool,
    WorkerResult,
    WorkerTask,
    merge_worker_results,
)


@dataclass(frozen=True)
class DistributedRunConfig:
    """Explicit rank configuration for a reproducible batch run."""

    run_id: str
    rank: int = 0
    world_size: int = 1
    workers: int = 1
    backend: str = "process"

    def __post_init__(self) -> None:
        if not str(self.run_id):
            raise ValueError("run_id must not be empty")
        if self.world_size < 1 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        if self.workers < 1:
            raise ValueError("workers must be positive")


@dataclass(frozen=True)
class WorkerManifest:
    """JSON manifest written by one rank and consumed by a merger."""

    run_id: str
    rank: int
    world_size: int
    task_ids: tuple[str, ...]
    results: tuple[WorkerResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "t4.worker-manifest.v1",
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "task_ids": list(self.task_ids),
            "results": [
                {
                    "task_id": result.task_id,
                    "value": _portable(result.value),
                    "rank": result.rank,
                    "worker_index": result.worker_index,
                    "duration_s": result.duration_s,
                    "error": result.error,
                }
                for result in self.results
            ],
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    @classmethod
    def read(cls, path: str | Path) -> "WorkerManifest":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if value.get("format") != "t4.worker-manifest.v1":
            raise ValueError(f"unsupported worker manifest format: {source}")
        results = tuple(
            WorkerResult(
                task_id=str(item["task_id"]),
                value=item.get("value"),
                rank=int(item["rank"]),
                worker_index=int(item.get("worker_index", 0)),
                duration_s=float(item.get("duration_s", 0.0)),
                error=item.get("error"),
            )
            for item in value.get("results", [])
        )
        manifest = cls(
            run_id=str(value["run_id"]),
            rank=int(value["rank"]),
            world_size=int(value["world_size"]),
            task_ids=tuple(str(item) for item in value.get("task_ids", [])),
            results=results,
        )
        if manifest.world_size < 1 or manifest.rank < 0 or manifest.rank >= manifest.world_size:
            raise ValueError(f"invalid rank/world size in worker manifest: {source}")
        result_ids = tuple(result.task_id for result in manifest.results)
        if set(result_ids) != set(manifest.task_ids) or len(result_ids) != len(manifest.task_ids):
            raise ValueError(f"worker manifest task_ids do not match results: {source}")
        if any(result.rank != manifest.rank for result in manifest.results):
            raise ValueError(f"worker result rank does not match manifest rank: {source}")
        return manifest


class DistributedExecutor:
    """Execute this rank's partition and optionally persist a manifest."""

    def __init__(self, config: DistributedRunConfig) -> None:
        self.config = config

    def run(
        self,
        tasks: Iterable[WorkerTask],
        *,
        manifest_path: Optional[str | Path] = None,
    ) -> list[WorkerResult]:
        values = list(tasks)
        with WorkerPool(
            workers=self.config.workers,
            rank=self.config.rank,
            world_size=self.config.world_size,
            backend=self.config.backend,
        ) as pool:
            results = pool.run_tasks(values)
        manifest = WorkerManifest(
            run_id=self.config.run_id,
            rank=self.config.rank,
            world_size=self.config.world_size,
            task_ids=tuple(result.task_id for result in results),
            results=tuple(results),
        )
        if manifest_path is not None:
            manifest.write(manifest_path)
        return results

    @staticmethod
    def merge_manifests(
        paths: Sequence[str | Path],
        *,
        run_id: Optional[str] = None,
        world_size: Optional[int] = None,
        expected_task_ids: Optional[Sequence[str]] = None,
        require_complete_world: bool = True,
        require_success: bool = True,
    ) -> list[WorkerResult]:
        manifests = [WorkerManifest.read(path) for path in paths]
        if not manifests:
            raise ValueError("at least one worker manifest is required")
        actual_run_id = manifests[0].run_id
        actual_world_size = manifests[0].world_size
        if run_id is not None and actual_run_id != run_id:
            raise ValueError(f"worker run ID mismatch: {actual_run_id!r} != {run_id!r}")
        if world_size is not None and actual_world_size != world_size:
            raise ValueError("worker world size does not match the requested run")
        if any(item.run_id != actual_run_id for item in manifests):
            raise ValueError("worker manifests belong to different runs")
        if any(item.world_size != actual_world_size for item in manifests):
            raise ValueError("worker manifests have inconsistent world sizes")
        ranks = {item.rank for item in manifests}
        if len(ranks) != len(manifests):
            raise ValueError("worker manifests contain duplicate ranks")
        if require_complete_world and ranks != set(range(actual_world_size)):
            raise ValueError(f"worker manifests are incomplete: got ranks {sorted(ranks)}")
        return merge_worker_results(
            (result for manifest in manifests for result in manifest.results),
            expected_task_ids=expected_task_ids,
            require_success=require_success,
        )


def _portable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_dict"):
        return _portable(value.as_dict())
    return str(value)


def merge_worker_manifests(
    paths: Sequence[str | Path],
    *,
    output_path: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    world_size: Optional[int] = None,
    require_complete_world: bool = True,
    require_success: bool = False,
) -> list[WorkerResult]:
    """Merge portable rank manifests and optionally write one combined file."""

    manifests = [WorkerManifest.read(path) for path in paths]
    expected = [task_id for manifest in manifests for task_id in manifest.task_ids]
    results = DistributedExecutor.merge_manifests(
        paths,
        run_id=run_id,
        world_size=world_size,
        expected_task_ids=expected,
        require_complete_world=require_complete_world,
        require_success=require_success,
    )
    if output_path is not None:
        if not manifests:
            raise ValueError("at least one worker manifest is required")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "t4.worker-manifest-merged.v1",
            "run_id": manifests[0].run_id,
            "world_size": manifests[0].world_size,
            "task_ids": [result.task_id for result in results],
            "results": [
                {
                    "task_id": result.task_id,
                    "value": _portable(result.value),
                    "rank": result.rank,
                    "worker_index": result.worker_index,
                    "duration_s": result.duration_s,
                    "error": result.error,
                }
                for result in results
            ],
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


__all__ = [
    "DistributedExecutor",
    "DistributedRunConfig",
    "WorkerManifest",
    "merge_worker_manifests",
]
