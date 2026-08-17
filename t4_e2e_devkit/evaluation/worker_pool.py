"""Dependency-light worker pools for local and rank-partitioned evaluation.

The public API mirrors the useful part of NuPlan's worker-pool boundary while
keeping execution local.  A rank is an explicit input, so the same task list
can be split across processes, machines or an external launcher without making
the devkit depend on a scheduler or tracking service.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar

from t4_e2e_devkit.evaluation.executor import rank_indices

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class WorkerResources:
    """Resource metadata carried with a task for an outer launcher."""

    cpu: int = 1
    gpu: int = 0
    memory_gb: Optional[float] = None

    def __post_init__(self) -> None:
        if self.cpu < 1:
            raise ValueError("worker resource cpu must be positive")
        if self.gpu < 0:
            raise ValueError("worker resource gpu must be non-negative")
        if self.memory_gb is not None and self.memory_gb <= 0.0:
            raise ValueError("worker resource memory_gb must be positive when provided")


@dataclass(frozen=True)
class WorkerTask:
    """A serializable unit of work for :class:`WorkerPool.run_tasks`."""

    task_id: str
    function: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    resources: WorkerResources = field(default_factory=WorkerResources)

    def __post_init__(self) -> None:
        if not str(self.task_id):
            raise ValueError("worker task_id must not be empty")
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class WorkerResult:
    """One deterministic task result, including rank and elapsed time."""

    task_id: str
    value: Any = None
    rank: int = 0
    worker_index: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """:return: whether the task completed without an exception."""

        return self.error is None


def _execute_task(task: WorkerTask, rank: int, worker_index: int) -> WorkerResult:
    started = time.perf_counter()
    try:
        value = task.function(*task.args, **task.kwargs)
    except BaseException as error:  # preserve the task boundary for batch runs
        return WorkerResult(
            task_id=task.task_id,
            rank=rank,
            worker_index=worker_index,
            duration_s=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
    return WorkerResult(
        task_id=task.task_id,
        value=value,
        rank=rank,
        worker_index=worker_index,
        duration_s=time.perf_counter() - started,
    )


class WorkerPool:
    """Ordered worker execution with explicit rank/world-size partitioning."""

    def __init__(
        self,
        workers: int = 1,
        *,
        rank: int = 0,
        world_size: int = 1,
        backend: str = "process",
        start_method: Optional[str] = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if backend not in {"serial", "thread", "process"}:
            raise ValueError("backend must be one of serial, thread or process")
        if start_method is not None and start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"unsupported multiprocessing start method {start_method!r}; "
                f"available: {mp.get_all_start_methods()}"
            )
        self.workers = int(workers)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.backend = backend
        self.start_method = start_method
        self._executor: Optional[ThreadPoolExecutor | ProcessPoolExecutor] = None

    def __enter__(self) -> "WorkerPool":
        self._ensure_executor()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.shutdown()

    def _ensure_executor(self):
        if self._executor is not None or self.backend == "serial":
            return self._executor
        if self.backend == "thread":
            self._executor = ThreadPoolExecutor(max_workers=self.workers)
        else:
            context = mp.get_context(self.start_method) if self.start_method else None
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=context,
            )
        return self._executor

    def submit(self, task: WorkerTask) -> Future[WorkerResult]:
        """Submit one task, preserving the rank in its result."""

        executor = self._ensure_executor()
        if executor is None:
            future: Future[WorkerResult] = Future()
            future.set_result(_execute_task(task, self.rank, 0))
            return future
        return executor.submit(_execute_task, task, self.rank, 0)

    def run_tasks(self, tasks: Iterable[WorkerTask]) -> List[WorkerResult]:
        """Run tasks in input order and return failures as data."""

        values = list(tasks)
        if self.world_size > 1:
            indices = rank_indices(len(values), self.rank, self.world_size)
            values = [values[index] for index in indices]
        if not values:
            return []
        return self._run_local_tasks(values)

    def _run_local_tasks(self, values: Sequence[WorkerTask]) -> List[WorkerResult]:
        if self.backend == "serial" or self.workers == 1 or len(values) == 1:
            return [_execute_task(task, self.rank, index) for index, task in enumerate(values)]
        executor = self._ensure_executor()
        assert executor is not None
        futures = [executor.submit(_execute_task, task, self.rank, index) for index, task in enumerate(values)]
        return [future.result() for future in futures]

    def map(self, function: Callable[[T], R], items: Iterable[T]) -> List[R]:
        """Map a function over this rank's partition, raising task exceptions."""

        values = list(items)
        selected = [values[index] for index in rank_indices(len(values), self.rank, self.world_size)]
        tasks = [WorkerTask(str(index), function, args=(value,)) for index, value in enumerate(selected)]
        results = self._run_local_tasks(tasks)
        failures = [result for result in results if not result.succeeded]
        if failures:
            raise RuntimeError(failures[0].error or "worker task failed")
        return [result.value for result in results]

    def map_indexed(self, function: Callable[[T], R], items: Sequence[T]) -> List[tuple[int, R]]:
        """Map a function and retain original indices for deterministic merging."""

        indices = rank_indices(len(items), self.rank, self.world_size)
        values = [items[index] for index in indices]
        results = self.map(function, values) if self.world_size == 1 else self._map_local(function, values)
        return list(zip(indices, results, strict=True))

    def _map_local(self, function: Callable[[T], R], values: Sequence[T]) -> List[R]:
        tasks = [WorkerTask(str(index), function, args=(value,)) for index, value in enumerate(values)]
        results = self._run_local_tasks(tasks)
        failures = [result for result in results if not result.succeeded]
        if failures:
            raise RuntimeError(failures[0].error or "worker task failed")
        return [result.value for result in results]

    def shutdown(self, wait: bool = True) -> None:
        """Release local workers; safe to call more than once."""

        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None


def merge_worker_results(
    results: Iterable[WorkerResult],
    *,
    expected_task_ids: Optional[Sequence[str]] = None,
    require_success: bool = True,
) -> List[WorkerResult]:
    """Validate and deterministically merge results from multiple ranks."""

    ordered = sorted(results, key=lambda result: result.task_id)
    task_ids = [result.task_id for result in ordered]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("worker results contain duplicate task IDs")
    if expected_task_ids is not None:
        expected = sorted(str(task_id) for task_id in expected_task_ids)
        if task_ids != expected:
            missing = sorted(set(expected) - set(task_ids))
            extra = sorted(set(task_ids) - set(expected))
            raise ValueError(f"worker result set mismatch: missing={missing}, extra={extra}")
    if require_success:
        failures = [result for result in ordered if not result.succeeded]
        if failures:
            raise RuntimeError(
                "worker task failed: "
                + "; ".join(f"{item.task_id}: {item.error}" for item in failures)
            )
    return ordered


__all__ = [
    "WorkerPool",
    "WorkerResources",
    "WorkerResult",
    "WorkerTask",
    "merge_worker_results",
]
