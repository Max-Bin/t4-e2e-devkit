"""Optional Ray worker backend.

Ray is intentionally not a core dependency.  Importing this module is lazy,
so CPU-only installs retain the same small startup surface while internal
clusters can use the same ``WorkerTask`` and ``WorkerResult`` contract.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from t4_e2e_devkit.evaluation.worker_pool import WorkerResult, WorkerTask, _execute_task


class RayWorkerPool:
    """Execute rank-local tasks through a Ray runtime."""

    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        address: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("invalid rank/world_size for RayWorkerPool")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.address = address
        self.namespace = namespace
        self._ray = None
        self._owns_runtime = False

    def __enter__(self) -> "RayWorkerPool":
        try:
            import ray
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "worker backend 'ray' requires the optional distributed dependency; "
                "install it in the internal runtime"
            ) from error
        self._ray = ray
        if not ray.is_initialized():
            kwargs = {"ignore_reinit_error": True}
            if self.address is not None:
                kwargs["address"] = self.address
            if self.namespace is not None:
                kwargs["namespace"] = self.namespace
            ray.init(**kwargs)
            self._owns_runtime = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        if self._owns_runtime and self._ray is not None:
            self._ray.shutdown()
            self._owns_runtime = False

    def run_tasks(
        self,
        tasks: Iterable[WorkerTask],
        *,
        skip_task_ids: Sequence[str] = (),
    ) -> list[WorkerResult]:
        if self._ray is None:
            raise RuntimeError("RayWorkerPool must be entered before run_tasks")
        values = list(tasks)
        task_ids = [task.task_id for task in values]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("worker tasks contain duplicate task IDs")
        from t4_e2e_devkit.evaluation.executor import rank_indices

        selected = [
            values[index] for index in rank_indices(len(values), self.rank, self.world_size)
        ]
        skipped = {str(value) for value in skip_task_ids}
        selected = [task for task in selected if task.task_id not in skipped]
        if not selected:
            return []

        remote_execute = self._ray.remote(_execute_task)
        refs = []
        for index, task in enumerate(selected):
            options = {
                "num_cpus": int(task.resources.cpu),
                "num_gpus": float(task.resources.gpu),
            }
            if task.resources.memory_gb is not None:
                options["memory"] = int(task.resources.memory_gb * 1024**3)
            refs.append(remote_execute.options(**options).remote(task, self.rank, index))
        return list(self._ray.get(refs))


__all__ = ["RayWorkerPool"]
