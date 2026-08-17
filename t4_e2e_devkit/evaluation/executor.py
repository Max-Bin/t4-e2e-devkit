"""Deterministic local execution for metric and feature jobs."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def shard_indices(length: int, rank: int = 0, world_size: int = 1) -> tuple[int, ...]:
    """Return deterministic ``rank``-strided indices for a finite collection."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return tuple(range(int(rank), int(length), int(world_size)))


@dataclass(frozen=True)
class LocalExecutor:
    """Run an ordered map serially or with local worker processes.

    The executor intentionally has no scheduler integration.  ``workers=1``
    is the default and is useful for debugging; larger values use Python's
    standard ``ProcessPoolExecutor`` and preserve input order.
    """

    workers: int = 1
    start_method: Optional[str] = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be positive, got {self.workers}")
        if self.start_method is not None:
            available = mp.get_all_start_methods()
            if self.start_method not in available:
                raise ValueError(
                    f"unsupported multiprocessing start method {self.start_method!r}; "
                    f"available: {available}"
                )

    def shard(self, items: Sequence[T], rank: int = 0, world_size: int = 1) -> list[T]:
        """Return this rank's stable subset without reordering the input."""

        return [items[index] for index in shard_indices(len(items), rank, world_size)]

    def map(
        self,
        function: Callable[[T], R],
        items: Iterable[T],
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> list[R]:
        """Apply ``function`` in deterministic input order."""

        values = list(items)
        selected = self.shard(values, rank=rank, world_size=world_size)
        if self.workers == 1 or len(selected) <= 1:
            return [function(value) for value in selected]
        context = mp.get_context(self.start_method) if self.start_method else None
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=context) as pool:
            return list(pool.map(function, selected))

    def map_indexed(
        self,
        function: Callable[[T], R],
        items: Iterable[T],
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> list[tuple[int, R]]:
        """Return ``(original_index, result)`` pairs for deterministic merging."""

        values = list(items)
        indices = shard_indices(len(values), rank, world_size)
        selected_results = self.map(
            function,
            [values[index] for index in indices],
            rank=0,
            world_size=1,
        )
        return list(zip(indices, selected_results, strict=True))

    def run(
        self,
        function: Callable[[T], R],
        items: Iterable[T],
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> list[R]:
        """Alias for :meth:`map` for executor-style call sites."""

        return self.map(function, items, rank=rank, world_size=world_size)


__all__ = ["LocalExecutor", "shard_indices"]
