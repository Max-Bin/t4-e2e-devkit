"""Local and rank-partitioned execution for simulation runners."""

from __future__ import annotations

from typing import Iterable

from t4_e2e_devkit.evaluation.worker_pool import WorkerPool, WorkerTask
from t4_e2e_devkit.planning.simulation.runner.abstract_runner import AbstractRunner
from t4_e2e_devkit.planning.simulation.runner.runner_report import RunnerReport
from t4_e2e_devkit.planning.simulation.runner.simulations_runner import run_simulation


class RunnerExecutor:
    """Run independent runners with the same worker boundary as evaluation."""

    def __init__(
        self,
        *,
        workers: int = 1,
        backend: str = "serial",
        rank: int = 0,
        world_size: int = 1,
        raise_on_error: bool = False,
    ) -> None:
        self.workers = workers
        self.backend = backend
        self.rank = rank
        self.world_size = world_size
        self.raise_on_error = raise_on_error

    def run(self, runners: Iterable[AbstractRunner]) -> list[RunnerReport]:
        values = list(runners)
        if not values:
            return []
        tasks = [
            WorkerTask(_runner_task_id(runner, index), _run_one, args=(runner, self.raise_on_error))
            for index, runner in enumerate(values)
        ]
        with WorkerPool(
            workers=self.workers,
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size,
        ) as pool:
            results = pool.run_tasks(tasks)
        failures = [result for result in results if not result.succeeded]
        if failures:
            raise RuntimeError(
                "runner worker failed: " + "; ".join(item.error or "unknown" for item in failures)
            )
        reports = [
            result.value
            if isinstance(result.value, RunnerReport)
            else RunnerReport.from_dict(result.value)
            for result in results
        ]
        return sorted(
            reports, key=lambda report: (report.scenario_name, report.planner_name, report.log_name)
        )


def _run_one(runner: AbstractRunner, raise_on_error: bool) -> RunnerReport:
    return run_simulation(runner, raise_on_error=raise_on_error)


def _runner_task_id(runner: AbstractRunner, index: int) -> str:
    scenario = getattr(runner, "scenario", None)
    token = getattr(scenario, "token", getattr(scenario, "scenario_name", index))
    return f"{index}:{token}"


__all__ = ["RunnerExecutor"]
