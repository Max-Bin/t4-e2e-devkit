"""Reproducible local orchestration for rank-partitioned T4 runs.

The evaluator already knows how to execute one rank and how to merge rank
artifacts.  This module owns the missing outer lifecycle: start every rank,
persist status and logs, retry failed ranks, resume completed ranks, and only
invoke the merger after the declared world is complete.  It deliberately
uses subprocesses rather than a shell so the same launcher is safe from a
terminal, a notebook, or a CI job.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

ORCHESTRATION_FORMAT = "t4.orchestration.run"
ORCHESTRATION_VERSION = 1


@dataclass(frozen=True)
class RankLaunch:
    """Portable lifecycle record for one launched rank."""

    rank: int
    status: str
    attempts: int
    returncode: Optional[int]
    pid: Optional[int]
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_s: float
    log_path: str
    error: Optional[str] = None
    resumed: bool = False
    run_id: Optional[str] = None
    command_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.status not in {"pending", "running", "completed", "failed", "timeout"}:
            raise ValueError(f"unsupported rank status: {self.status!r}")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        if self.duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "status": self.status,
            "attempts": self.attempts,
            "returncode": self.returncode,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "log_path": self.log_path,
            "error": self.error,
            "resumed": self.resumed,
            "run_id": self.run_id,
            "command_digest": self.command_digest,
        }


@dataclass(frozen=True)
class DistributedLaunchResult:
    """Outcome of one complete local orchestration run."""

    run_id: str
    status: str
    world_size: int
    ranks: tuple[RankLaunch, ...]
    output_dir: Path
    merged_dir: Optional[Path] = None
    merge_error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise ValueError(f"unsupported launch status: {self.status!r}")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if len(self.ranks) != self.world_size:
            raise ValueError("one rank record is required for every world rank")

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": ORCHESTRATION_FORMAT,
            "version": ORCHESTRATION_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "world_size": self.world_size,
            "output_dir": ".",
            "merged_dir": None
            if self.merged_dir is None
            else str(self.merged_dir.relative_to(self.output_dir)),
            "merge_error": self.merge_error,
            "ranks": [rank.as_dict() for rank in self.ranks],
        }


class LocalDistributedLauncher:
    """Launch rank-aware CLI commands with durable status and retry semantics.

    ``command`` is the command for one rank without rank-specific options,
    for example ``["t4e2e", "evaluate", list_path, "--agent", "agent"]``.
    The launcher replaces or appends ``--rank``, ``--world-size`` and
    ``--output-dir``.  A caller may provide a merger callback that receives
    all rank directories and ``output_dir / "merged"`` after every rank has
    completed successfully.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        world_size: int,
        run_id: Optional[str] = None,
        backend: str = "process",
        max_retries: int = 0,
        timeout_s: Optional[float] = None,
        resume: bool = True,
        fail_fast: bool = False,
        cwd: str | Path | None = None,
        environment: Optional[Mapping[str, str]] = None,
        environment_factory: Optional[Callable[[int], Mapping[str, str]]] = None,
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if backend not in {"sequential", "process"}:
            raise ValueError("backend must be sequential or process")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if timeout_s is not None and timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive when provided")
        self.output_dir = Path(output_dir).resolve()
        self.world_size = int(world_size)
        self.run_id = str(run_id or self._default_run_id())
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        self.backend = backend
        self.max_retries = int(max_retries)
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.resume = bool(resume)
        self.fail_fast = bool(fail_fast)
        self.cwd = None if cwd is None else str(Path(cwd).resolve())
        self.environment = {str(key): str(value) for key, value in (environment or {}).items()}
        self.environment_factory = environment_factory

    @property
    def rank_dirs(self) -> tuple[Path, ...]:
        return tuple(self.output_dir / f"rank-{rank}" for rank in range(self.world_size))

    @property
    def merged_dir(self) -> Path:
        return self.output_dir / "merged"

    def run(
        self,
        command: Sequence[str],
        *,
        merger: Optional[Callable[[Sequence[Path], Path], Any]] = None,
        rank_option: str = "--rank",
        world_size_option: str = "--world-size",
        output_option: str = "--output-dir",
    ) -> DistributedLaunchResult:
        """Run all ranks and optionally merge their reports.

        The run descriptor is written before launching and after every rank
        transition.  If the parent process is interrupted, completed rank
        descriptors remain usable by a subsequent ``resume=True`` call.
        """

        base_command = tuple(str(value) for value in command)
        if not base_command:
            raise ValueError("command must not be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        status_dir = self.output_dir / "orchestration"
        status_dir.mkdir(parents=True, exist_ok=True)
        run_path = status_dir / "run.json"
        command_digest = hashlib.sha256(
            json.dumps(list(base_command), separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        self._write_run(
            run_path,
            {
                "format": ORCHESTRATION_FORMAT,
                "version": ORCHESTRATION_VERSION,
                "run_id": self.run_id,
                "status": "running",
                "world_size": self.world_size,
                "command_digest": command_digest,
                "backend": self.backend,
                "max_retries": self.max_retries,
                "timeout_s": self.timeout_s,
                "ranks": [],
            },
        )

        def launch(rank: int) -> RankLaunch:
            return self._run_rank(
                rank,
                base_command,
                status_dir,
                rank_option=rank_option,
                world_size_option=world_size_option,
                output_option=output_option,
                command_digest=command_digest,
            )

        if self.backend == "sequential" or self.world_size == 1:
            rank_results = [launch(rank) for rank in range(self.world_size)]
        else:
            with ThreadPoolExecutor(max_workers=self.world_size) as executor:
                futures = [executor.submit(launch, rank) for rank in range(self.world_size)]
                rank_results = [future.result() for future in futures]
        rank_results.sort(key=lambda item: item.rank)
        self._write_run(
            run_path,
            {
                "format": ORCHESTRATION_FORMAT,
                "version": ORCHESTRATION_VERSION,
                "run_id": self.run_id,
                "status": "ranks_completed"
                if all(item.status == "completed" for item in rank_results)
                else "failed",
                "world_size": self.world_size,
                "command_digest": command_digest,
                "backend": self.backend,
                "max_retries": self.max_retries,
                "timeout_s": self.timeout_s,
                "ranks": [item.as_dict() for item in rank_results],
            },
        )

        merge_error = None
        merged_dir: Optional[Path] = None
        all_succeeded = all(item.status == "completed" for item in rank_results)
        if all_succeeded and merger is not None:
            merged_dir = self.merged_dir
            try:
                merger(self.rank_dirs, merged_dir)
            except Exception as error:  # noqa: BLE001 - durable launch failure
                merge_error = f"{type(error).__name__}: {error}"
                all_succeeded = False

        result = DistributedLaunchResult(
            run_id=self.run_id,
            status="completed" if all_succeeded else "failed",
            world_size=self.world_size,
            ranks=tuple(rank_results),
            output_dir=self.output_dir,
            merged_dir=merged_dir,
            merge_error=merge_error,
        )
        self._write_run(
            run_path,
            {
                **result.as_dict(),
                "command_digest": command_digest,
                "backend": self.backend,
                "max_retries": self.max_retries,
                "timeout_s": self.timeout_s,
            },
        )
        return result

    def _run_rank(
        self,
        rank: int,
        base_command: Sequence[str],
        status_dir: Path,
        *,
        rank_option: str,
        world_size_option: str,
        output_option: str,
        command_digest: str,
    ) -> RankLaunch:
        rank_dir = self.rank_dirs[rank]
        rank_dir.mkdir(parents=True, exist_ok=True)
        log_path = status_dir / f"rank-{rank}.log"
        relative_log = str(log_path.relative_to(self.output_dir))
        state_path = status_dir / f"rank-{rank}.json"
        previous = self._read_rank(state_path)
        if (
            self.resume
            and previous is not None
            and previous.get("status") == "completed"
            and previous.get("run_id") == self.run_id
            and previous.get("command_digest") == command_digest
            and self._has_completed_marker(rank_dir)
        ):
            return RankLaunch(
                rank=rank,
                status="completed",
                attempts=int(previous.get("attempts", 0)),
                returncode=0,
                pid=previous.get("pid"),
                started_at=previous.get("started_at"),
                finished_at=previous.get("finished_at"),
                duration_s=float(previous.get("duration_s", 0.0)),
                log_path=relative_log,
                resumed=True,
                run_id=self.run_id,
                command_digest=command_digest,
            )

        command = _set_option(base_command, rank_option, str(rank))
        command = _set_option(command, world_size_option, str(self.world_size))
        command = _set_option(command, output_option, str(rank_dir))
        env = os.environ.copy()
        env.update(self.environment)
        if self.environment_factory is not None:
            env.update(
                {str(key): str(value) for key, value in self.environment_factory(rank).items()}
            )
        env.update(
            {
                "T4E2E_RUN_ID": self.run_id,
                "T4E2E_RANK": str(rank),
                "T4E2E_WORLD_SIZE": str(self.world_size),
                "T4E2E_OUTPUT_DIR": str(rank_dir),
            }
        )

        last: Optional[RankLaunch] = None
        for attempt in range(1, self.max_retries + 2):
            started = _now()
            started_clock = time.monotonic()
            self._write_rank(
                state_path,
                RankLaunch(
                    rank=rank,
                    status="running",
                    attempts=attempt,
                    returncode=None,
                    pid=None,
                    started_at=started,
                    finished_at=None,
                    duration_s=0.0,
                    log_path=relative_log,
                    run_id=self.run_id,
                    command_digest=command_digest,
                ),
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== attempt {attempt} ===\n")
                log.flush()
                try:
                    process = subprocess.Popen(
                        list(command),
                        cwd=self.cwd,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=os.name == "posix",
                    )
                except OSError as error:
                    finished = _now()
                    last = RankLaunch(
                        rank=rank,
                        status="failed",
                        attempts=attempt,
                        returncode=None,
                        pid=None,
                        started_at=started,
                        finished_at=finished,
                        duration_s=time.monotonic() - started_clock,
                        log_path=relative_log,
                        error=f"rank could not start: {error}",
                        run_id=self.run_id,
                        command_digest=command_digest,
                    )
                    self._write_rank(state_path, last)
                    if self.fail_fast:
                        break
                    continue
                timed_out = False
                while True:
                    try:
                        returncode = process.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if (
                            self.timeout_s is not None
                            and time.monotonic() - started_clock >= self.timeout_s
                        ):
                            timed_out = True
                            _stop_process(process, signal.SIGTERM)
                            try:
                                returncode = process.wait(timeout=5.0)
                            except subprocess.TimeoutExpired:
                                _stop_process(process, signal.SIGKILL)
                                returncode = process.wait()
                            break
                finished = _now()
            duration = time.monotonic() - started_clock
            if timed_out:
                status = "timeout"
                error = f"rank exceeded timeout of {self.timeout_s:g}s"
            elif returncode == 0:
                status = "completed"
                error = None
            else:
                status = "failed"
                error = f"rank exited with code {returncode}"
            last = RankLaunch(
                rank=rank,
                status=status,
                attempts=attempt,
                returncode=int(returncode),
                pid=int(process.pid),
                started_at=started,
                finished_at=finished,
                duration_s=duration,
                log_path=relative_log,
                error=error,
                run_id=self.run_id,
                command_digest=command_digest,
            )
            self._write_rank(state_path, last)
            if status == "completed":
                return last
            if self.fail_fast:
                break
        assert last is not None
        return last

    def _default_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run-{stamp}-{os.getpid()}"

    @staticmethod
    def _read_rank(path: Path) -> Optional[dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _has_completed_marker(directory: Path) -> bool:
        """Return whether a rank wrote a successful output manifest."""

        for name in ("run.json", "status.json"):
            try:
                value = json.loads((directory / name).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("status") == "completed":
                return True
        return False

    @staticmethod
    def _write_rank(path: Path, rank: RankLaunch) -> None:
        LocalDistributedLauncher._write_json(path, rank.as_dict())

    @staticmethod
    def _write_run(path: Path, value: Mapping[str, Any]) -> None:
        LocalDistributedLauncher._write_json(path, value)

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def _stop_process(process: subprocess.Popen[str], signum: signal.Signals) -> None:
    """Stop a rank and its descendants when the platform supports process groups."""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signum)
            return
        except (ProcessLookupError, PermissionError):
            pass
    if signum == signal.SIGKILL:
        process.kill()
    else:
        process.terminate()


def _set_option(command: Sequence[str], option: str, value: str) -> tuple[str, ...]:
    """Replace one argv option without invoking a shell."""

    values = list(command)
    for index, item in enumerate(values):
        if item == option:
            if index + 1 >= len(values):
                raise ValueError(f"command option {option!r} has no value")
            values[index + 1] = value
            return tuple(values)
        if item.startswith(option + "="):
            values[index] = f"{option}={value}"
            return tuple(values)
    values.extend((option, value))
    return tuple(values)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DistributedLaunchResult",
    "LocalDistributedLauncher",
    "ORCHESTRATION_FORMAT",
    "ORCHESTRATION_VERSION",
    "RankLaunch",
]
