"""A filesystem-only barrier for externally launched local workers."""

from __future__ import annotations

import time
from pathlib import Path


class FileBackedBarrier:
    """Wait until ``world_size`` rank marker files exist."""

    def __init__(
        self, path: str | Path, rank: int, world_size: int, *, timeout_s: float = 600.0
    ) -> None:
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.path = Path(path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.timeout_s = float(timeout_s)

    def wait(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / f"rank-{self.rank}.done").touch(exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        expected = [self.path / f"rank-{rank}.done" for rank in range(self.world_size)]
        while not all(path.is_file() for path in expected):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                missing = [str(path.name) for path in expected if not path.is_file()]
                raise TimeoutError(f"barrier timed out; missing ranks: {missing}")
            time.sleep(min(0.2, remaining))


__all__ = ["FileBackedBarrier"]
