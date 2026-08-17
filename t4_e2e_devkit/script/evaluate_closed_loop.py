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
from t4_e2e_devkit.evaluation.report import aggregate_evaluation
from t4_e2e_devkit.planning.simulation.closed_loop import (
    T4ClosedLoopConfig,
    run_t4_closed_loop,
)

logger = logging.getLogger(__name__)

RUN_FORMAT = "t4.closed_loop.run"
RUN_VERSION = 1


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
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    max_retries: int = 0,
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
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}"
        )
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    selected = data_list.filtered(max_rows=max_rows) if max_rows is not None else data_list
    selected_rows = list(selected)
    rows = [
        (row_index, row)
        for row_index, row in enumerate(selected_rows)
        if row_index % num_shards == shard_index
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
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "data_list": None if data_list.path is None else str(data_list.path),
        "root": str(data_list.root),
        "selected_rows": int(len(selected_rows)),
        "shard_rows": int(len(rows)),
    }
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
    resumed = 0
    attempts_total = 0
    agent = None
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
                continue

        scene_dir = selected.absolute_scene_dir(scene_relative)
        last_error = "unknown error"
        completed = False
        for attempt in range(1, max_retries + 2):
            attempts_total += 1
            try:
                if agent is None:
                    agent = build_agent(agent_name)
                    agent.initialize()
                result = run_t4_closed_loop(
                    agent,
                    scene_dir=scene_dir,
                    root=selected.root,
                    start_frame=start_frame,
                    num_steps=num_steps,
                    config=loop_config,
                )
                window_metrics = compute_closed_loop_metrics(result, token=token)
                write_rollout_artifact(
                    artifact_path,
                    token=token,
                    result=result,
                    metrics=window_metrics,
                    config=run_config,
                    config_fingerprint=config_fingerprint,
                    attempts=attempt,
                )
                metrics.append(window_metrics)
                completed = True
                break
            except Exception as error:  # noqa: BLE001 - one bad row is reported
                last_error = repr(error)
                logger.warning(
                    "closed-loop failed for %s (attempt %d/%d): %s",
                    token,
                    attempt,
                    max_retries + 1,
                    error,
                )
        if not completed:
            failures.append((token, last_error))
            write_failed_artifact(
                artifact_path,
                token=token,
                config=run_config,
                config_fingerprint=config_fingerprint,
                error=last_error,
                attempts=max_retries + 1,
            )

    report = aggregate_evaluation(closed_loop=metrics, num_failed=len(failures))
    report["run"].update(
        {
            "num_rows": float(len(rows)),
            "num_resumed": float(resumed),
            "num_attempts": float(attempts_total),
            "num_shards": float(num_shards),
            "shard_index": float(shard_index),
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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
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
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        resume=args.resume,
        max_retries=args.max_retries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
