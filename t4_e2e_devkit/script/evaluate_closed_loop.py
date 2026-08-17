"""Evaluate sensor-replay closed-loop rollouts from a T4 data list."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from omegaconf import OmegaConf

from t4_e2e_devkit.agents.registry import build_agent
from t4_e2e_devkit.common.constants import PAST_FRAMES
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.evaluation.closed_loop import (
    ClosedLoopMetrics,
    compute_closed_loop_metrics,
)
from t4_e2e_devkit.evaluation.report import aggregate_evaluation
from t4_e2e_devkit.planning.simulation.closed_loop import (
    T4ClosedLoopConfig,
    run_t4_closed_loop,
)

logger = logging.getLogger(__name__)


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
    max_rows: Optional[int] = None,
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

    selected = data_list.filtered(max_rows=max_rows) if max_rows is not None else data_list
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    agent = build_agent(agent_name)
    agent.initialize()
    config = T4ClosedLoopConfig(
        history_frames=history_frames,
        replan_interval=replan_interval,
        max_speed_mps=max_speed_mps,
        goal_radius_m=goal_radius_m,
    )

    metrics: list[ClosedLoopMetrics] = []
    failures: list[tuple[str, str]] = []
    for scene_relative, start_frame in selected:
        token = f"{scene_relative}@{start_frame}"
        scene_dir = selected.absolute_scene_dir(scene_relative)
        try:
            result = run_t4_closed_loop(
                agent,
                scene_dir=scene_dir,
                root=selected.root,
                start_frame=start_frame,
                num_steps=num_steps,
                config=config,
            )
            metrics.append(compute_closed_loop_metrics(result, token=token))
        except Exception as error:  # noqa: BLE001 - one bad row must be reported
            failures.append((token, repr(error)))
            logger.warning("closed-loop failed for %s: %s", token, error)

    report = aggregate_evaluation(closed_loop=metrics, num_failed=len(failures))
    _write_closed_loop_csv(output_path / "closed_loop.csv", metrics)
    _write_json(output_path / "aggregate.json", report)
    OmegaConf.save(OmegaConf.create(report), output_path / "aggregate.yaml")
    if failures:
        with (output_path / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["token", "error"])
            writer.writerows(failures)
    return report


def _write_closed_loop_csv(path: Path, results: Iterable[ClosedLoopMetrics]) -> None:
    rows = list(results)
    names = sorted({name for result in rows for name in result.values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "termination_reason", *names])
        for result in rows:
            values = result.values
            writer.writerow(
                [
                    result.token or "",
                    result.termination_reason or "",
                    *[
                        f"{values[name]:.6f}" if name in values else ""
                        for name in names
                    ],
                ]
            )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument("--max-rows", type=int, default=None)
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
        max_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
