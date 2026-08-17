"""Evaluate sensor-replay closed-loop rollouts from a T4 data list."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from t4_e2e_devkit.agents.registry import build_agent
from t4_e2e_devkit.common.constants import PAST_FRAMES
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.evaluation.closed_loop import (
    ClosedLoopMetrics,
    compute_closed_loop_metrics,
)
from t4_e2e_devkit.evaluation.closed_loop_artifact import (
    CLOSED_LOOP_ARTIFACT_FORMAT,
    CLOSED_LOOP_ARTIFACT_VERSION,
    load_rollout_metrics,
    rollout_artifact_path,
    write_failed_artifact,
    write_rollout_artifact,
)
from t4_e2e_devkit.evaluation.closed_loop_report import (
    write_closed_loop_csv,
    write_closed_loop_ticks,
    write_static_html_report,
)
from t4_e2e_devkit.evaluation.distributed import (
    DistributedRunConfig,
    WorkerManifest,
)
from t4_e2e_devkit.evaluation.report import aggregate_evaluation
from t4_e2e_devkit.evaluation.worker_pool import WorkerPool, WorkerResult, WorkerTask
from t4_e2e_devkit.planning.simulation.closed_loop import (
    T4ClosedLoopConfig,
    run_t4_closed_loop,
)

logger = logging.getLogger(__name__)

RUN_FORMAT = "t4.closed_loop.run"
RUN_VERSION = 1


def _run_rollout_task(
    agent_name: str,
    scene_dir: str,
    root: str,
    start_frame: int,
    num_steps: int,
    loop_config: T4ClosedLoopConfig,
    token: str,
    max_retries: int,
) -> dict[str, Any]:
    """Run one rollout in an isolated worker and return an inspectable result."""

    last_error = "unknown error"
    for attempt in range(1, max_retries + 2):
        try:
            agent = build_agent(agent_name)
            agent.initialize()
            result = run_t4_closed_loop(
                agent,
                scene_dir=scene_dir,
                root=root,
                start_frame=start_frame,
                num_steps=num_steps,
                config=loop_config,
            )
            return {"status": "ok", "result": result, "attempts": attempt}
        except Exception as error:  # noqa: BLE001 - task failure is reported to the caller
            last_error = repr(error)
            logger.warning(
                "closed-loop worker failed for %s (attempt %d/%d): %s",
                token,
                attempt,
                max_retries + 1,
                error,
            )
    return {
        "status": "failed",
        "error": last_error,
        "attempts": max_retries + 1,
    }


def evaluate_closed_loop(
    data_list: DataList,
    *,
    agent_name: str,
    output_dir: str | Path,
    history_frames: int = PAST_FRAMES,
    num_steps: int = 200,
    replan_interval: int = 1,
    max_speed_mps: float = 20.0,
    goal_radius_m: float = 2.0,
    ttc_horizon_s: Optional[float] = 1.0,
    max_rows: Optional[int] = None,
    resume: bool = False,
    max_retries: int = 0,
    rank: int = 0,
    world_size: int = 1,
    workers: int = 1,
    worker_backend: str = "serial",
    run_id: Optional[str] = None,
    manifest_path: Optional[str | Path] = None,
) -> dict[str, dict[str, float]]:
    """Run closed loop for every row and write independent result files.

    A data-list row supplies the scene and the initial source frame. The
    rollout then reads consecutive source frames from that scene. Failures are
    recorded separately and never silently removed from the aggregate.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    effective_rank = int(rank)
    effective_world_size = int(world_size)
    if effective_world_size < 1 or effective_rank < 0 or effective_rank >= effective_world_size:
        raise ValueError(
            f"rank must be in [0, {effective_world_size}); got {effective_rank}"
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    if worker_backend not in {"serial", "thread", "process"}:
        raise ValueError("worker_backend must be serial, thread or process")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    selected = data_list.filtered(max_rows=max_rows) if max_rows is not None else data_list
    selected_rows = list(selected)
    rows = [
        (row_index, row)
        for row_index, row in enumerate(selected_rows)
        if row_index % effective_world_size == effective_rank
    ]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loop_config = T4ClosedLoopConfig(
        history_frames=history_frames,
        replan_interval=replan_interval,
        max_speed_mps=max_speed_mps,
        goal_radius_m=goal_radius_m,
        ttc_horizon_s=ttc_horizon_s,
    )
    run_config: dict[str, Any] = {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "artifact_format": CLOSED_LOOP_ARTIFACT_FORMAT,
        "artifact_version": CLOSED_LOOP_ARTIFACT_VERSION,
        "agent": agent_name,
        "history_frames": int(history_frames),
        "num_steps": int(num_steps),
        "replan_interval": int(replan_interval),
        "max_speed_mps": float(max_speed_mps),
        "goal_radius_m": float(goal_radius_m),
        "ttc_horizon_s": None if ttc_horizon_s is None else float(ttc_horizon_s),
        "max_rows": None if max_rows is None else int(max_rows),
        "rank": int(effective_rank),
        "world_size": int(effective_world_size),
        "workers": int(workers),
        "worker_backend": worker_backend,
        "selected_rows": int(len(selected_rows)),
        "rank_rows": int(len(rows)),
    }
    if run_id is None:
        identity = {
            key: value
            for key, value in run_config.items()
            if key not in {"rank", "world_size", "workers", "worker_backend"}
        }
        run_id = f"closed-loop-{_fingerprint(identity)[:16]}"
    run_config["run_id"] = str(run_id)
    distributed_config = DistributedRunConfig(
        run_id=str(run_id),
        rank=effective_rank,
        world_size=effective_world_size,
        workers=workers,
        backend=worker_backend,
    )
    config_fingerprint = _fingerprint(run_config)
    artifact_dir = output_path / "rollouts"
    _write_json(
        output_path / "run.json",
        {
            **run_config,
            "status": "running",
            "config_fingerprint": config_fingerprint,
        },
    )

    metrics: list[ClosedLoopMetrics] = []
    failures: list[tuple[str, str]] = []
    manifest_results: list[WorkerResult] = []
    resumed = 0
    attempts_total = 0
    pending: list[tuple[int, tuple[str, int], str, Path]] = []
    for row_index, (scene_relative, start_frame) in rows:
        token = f"{scene_relative}@{start_frame}"
        artifact_path = _artifact_path(artifact_dir, row_index, token)
        if resume:
            cached = load_rollout_metrics(
                artifact_path,
                token=token,
                config_fingerprint=config_fingerprint,
            )
            if cached is not None:
                metrics.append(cached)
                resumed += 1
                manifest_results.append(
                    WorkerResult(
                        task_id=token,
                        value={"status": "ok", "metrics": dict(cached.values)},
                        rank=effective_rank,
                    )
                )
                continue
        pending.append(
            (
                row_index,
                (scene_relative, int(start_frame)),
                token,
                artifact_path,
            )
        )

    if worker_backend == "serial" and workers == 1:
        agent = None
        worker_outputs: list[WorkerResult] = []
        for _row_index, (scene_relative, start_frame), token, _artifact_file in pending:
            output = {"status": "failed", "error": "unknown error", "attempts": 0}
            for attempt in range(1, max_retries + 2):
                attempts_total += 1
                try:
                    if agent is None:
                        agent = build_agent(agent_name)
                        agent.initialize()
                    result = run_t4_closed_loop(
                        agent,
                        scene_dir=selected.absolute_scene_dir(scene_relative),
                        root=selected.root,
                        start_frame=start_frame,
                        num_steps=num_steps,
                        config=loop_config,
                    )
                    output = {"status": "ok", "result": result, "attempts": attempt}
                    break
                except Exception as error:  # noqa: BLE001 - one bad row is reported
                    output = {"status": "failed", "error": repr(error), "attempts": attempt}
                    logger.warning(
                        "closed-loop failed for %s (attempt %d/%d): %s",
                        token,
                        attempt,
                        max_retries + 1,
                        error,
                    )
            worker_outputs.append(WorkerResult(task_id=token, value=output, rank=effective_rank))
    else:
        tasks = [
            WorkerTask(
                task_id=token,
                function=_run_rollout_task,
                args=(
                    agent_name,
                    str(selected.absolute_scene_dir(scene_relative)),
                    str(selected.root),
                    start_frame,
                    num_steps,
                    loop_config,
                    token,
                    max_retries,
                ),
            )
            for _, (scene_relative, start_frame), token, _ in pending
        ]
        with WorkerPool(workers=workers, backend=worker_backend) as pool:
            raw_outputs = pool.run_tasks(tasks)
        worker_outputs = [
            WorkerResult(
                task_id=result.task_id,
                value=result.value,
                rank=effective_rank,
                worker_index=result.worker_index,
                duration_s=result.duration_s,
                error=result.error,
            )
            for result in raw_outputs
        ]

    pending_by_token = {token: (row_index, artifact_path) for row_index, _, token, artifact_path in pending}
    for worker_result in worker_outputs:
        row_index, artifact_path = pending_by_token[worker_result.task_id]
        token = worker_result.task_id
        output = worker_result.value if isinstance(worker_result.value, dict) else {}
        attempts = int(output.get("attempts", 0))
        attempts_total += attempts if not (worker_backend == "serial" and workers == 1) else 0
        if worker_result.error is not None:
            output = {"status": "failed", "error": worker_result.error, "attempts": attempts}
        if output.get("status") == "ok":
            result = output["result"]
            window_metrics = compute_closed_loop_metrics(result, token=token)
            write_rollout_artifact(
                artifact_path,
                token=token,
                result=result,
                metrics=window_metrics,
                config=run_config,
                config_fingerprint=config_fingerprint,
                attempts=attempts,
            )
            metrics.append(window_metrics)
            manifest_results.append(
                WorkerResult(
                    task_id=token,
                    value={"status": "ok", "metrics": dict(window_metrics.values)},
                    rank=effective_rank,
                    worker_index=worker_result.worker_index,
                    duration_s=worker_result.duration_s,
                )
            )
        else:
            error = str(output.get("error", "unknown error"))
            failures.append((token, error))
            write_failed_artifact(
                artifact_path,
                token=token,
                config=run_config,
                config_fingerprint=config_fingerprint,
                error=error,
                attempts=attempts or max_retries + 1,
            )
            manifest_results.append(
                WorkerResult(
                    task_id=token,
                    value={"status": "failed", "error": error},
                    rank=effective_rank,
                    worker_index=worker_result.worker_index,
                    duration_s=worker_result.duration_s,
                    error=error,
                )
            )

    manifest_file = Path(manifest_path) if manifest_path is not None else output_path / f"worker-manifest-rank-{effective_rank}.json"
    WorkerManifest(
        run_id=distributed_config.run_id,
        rank=effective_rank,
        world_size=effective_world_size,
        task_ids=tuple(
            f"{scene_relative}@{start_frame}"
            for _, (scene_relative, start_frame) in rows
        ),
        results=tuple(sorted(manifest_results, key=lambda item: item.task_id)),
    ).write(manifest_file)

    report = aggregate_evaluation(closed_loop=metrics, num_failed=len(failures))
    report["run"].update(
        {
            "num_rows": float(len(rows)),
            "num_resumed": float(resumed),
            "num_attempts": float(attempts_total),
            "world_size": float(effective_world_size),
            "rank": float(effective_rank),
        }
    )
    write_closed_loop_csv(output_path / "closed_loop.csv", metrics)
    write_closed_loop_ticks(output_path / "closed_loop_ticks.csv", metrics)
    _write_json(output_path / "aggregate.json", report)
    OmegaConf.save(OmegaConf.create(report), output_path / "aggregate.yaml")
    with (output_path / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "error"])
        writer.writerows(failures)
    _write_json(
        output_path / "run.json",
        {
            **run_config,
            "status": "completed",
            "config_fingerprint": config_fingerprint,
            "num_completed": len(metrics),
            "num_failed": len(failures),
            "num_resumed": resumed,
            "num_attempts": attempts_total,
            "manifest": manifest_file.name,
        },
    )
    write_static_html_report(output_path)
    return report


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_path(directory: Path, row_index: int, token: str) -> Path:
    return rollout_artifact_path(directory, row_index, token)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="t4e2e evaluate-closed-loop",
        description="Evaluate ego-only sensor-replay closed loop on a T4 data list.",
    )
    parser.add_argument("data_list", help="T4 data-list JSON")
    parser.add_argument("--agent", required=True, help="registered deployable agent")
    parser.add_argument("--output-dir", default="closed_loop", help="report directory")
    parser.add_argument("--history-frames", type=int, default=PAST_FRAMES)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--max-speed-mps", type=float, default=20.0)
    parser.add_argument("--goal-radius-m", type=float, default=2.0)
    parser.add_argument(
        "--ttc-horizon-s",
        type=float,
        default=1.0,
        help="constant-velocity replay TTC horizon; omit with --disable-ttc",
    )
    parser.add_argument("--disable-ttc", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0, help="current distributed rank")
    parser.add_argument("--world-size", type=int, default=1, help="total distributed ranks")
    parser.add_argument("--workers", type=int, default=1, help="local workers within this rank")
    parser.add_argument(
        "--worker-backend",
        choices=("serial", "thread", "process"),
        default="serial",
        help="local worker implementation",
    )
    parser.add_argument("--run-id", default=None, help="stable ID shared by all ranks")
    parser.add_argument("--manifest", default=None, help="worker manifest output path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse successful rollout artifacts matching this run configuration",
    )
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    data_list = load_data_list(args.data_list)
    report = evaluate_closed_loop(
        data_list,
        agent_name=args.agent,
        output_dir=args.output_dir,
        history_frames=args.history_frames,
        num_steps=args.num_steps,
        replan_interval=args.replan_interval,
        max_speed_mps=args.max_speed_mps,
        goal_radius_m=args.goal_radius_m,
        ttc_horizon_s=None if args.disable_ttc else args.ttc_horizon_s,
        max_rows=args.max_rows,
        resume=args.resume,
        max_retries=args.max_retries,
        rank=args.rank,
        world_size=args.world_size,
        workers=args.workers,
        worker_backend=args.worker_backend,
        run_id=args.run_id,
        manifest_path=args.manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
