"""Run a resolved T4 experiment configuration through the public CLI paths."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from t4_e2e_devkit.config import ExperimentConfig, load_experiment_config


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e run-config")
    parser.add_argument("config")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config, overrides=args.override)
    if args.dry_run:
        print(json.dumps({**config.as_dict(), "fingerprint": config.fingerprint}, indent=2, sort_keys=True))
        return 0
    if config.mode in {"evaluate", "closed_loop", "submit"}:
        from t4_e2e_devkit.script.distributed import main as distributed_main

        return distributed_main(_distributed_args(config))
    if config.evaluation.submission_dir is None:
        parser.error("evaluation.submission_dir is required for mode=score_submission")
    from t4_e2e_devkit.script.distributed import main as distributed_main

    return distributed_main(_score_distributed_args(config))


def _distributed_args(config: ExperimentConfig) -> list[str]:
    kind = "closed-loop" if config.mode == "closed_loop" else config.mode
    args = [
        kind,
        config.dataset.data_list,
        "--agent",
        config.agent,
        "--output-dir",
        config.output.directory,
        "--world-size",
        str(config.workers.world_size),
        "--workers",
        str(config.workers.workers),
        "--worker-backend",
        config.workers.backend,
        "--launcher-backend",
        config.workers.launcher_backend,
        "--max-rank-retries",
        str(config.workers.max_retries),
        "--max-retries",
        str(config.evaluation.max_retries),
        "--run-id",
        config.fingerprint[:24],
    ]
    if config.workers.timeout_s is not None:
        args.extend(("--timeout-s", str(config.workers.timeout_s)))
    if not config.workers.resume:
        args.append("--no-resume")
    if config.evaluation.checkpoint is not None:
        args.extend(("--checkpoint", config.evaluation.checkpoint))
    if config.agent_params:
        args.extend(
            (
                "--agent-params-json",
                json.dumps(config.agent_params, sort_keys=True, separators=(",", ":")),
            )
        )
    if config.dataset.reader:
        args.extend(
            (
                "--reader-config-json",
                json.dumps(config.dataset.reader, sort_keys=True, separators=(",", ":")),
            )
        )
    if config.mode == "evaluate":
        args.extend(("--backend", config.evaluation.backend))
        if config.evaluation.device is not None:
            args.extend(("--device", config.evaluation.device))
        args.extend(("--families", *config.evaluation.families))
        args.extend(
            (
                "--history-frames",
                str(config.dataset.history_frames),
                "--future-frames",
                str(config.dataset.future_frames),
                "--frame-interval",
                str(config.dataset.frame_interval),
            )
        )
    elif config.mode == "closed_loop":
        args.extend(
            (
                "--history-frames",
                str(config.dataset.history_frames),
                "--num-steps",
                str(config.simulation.num_steps),
                "--replan-interval",
                str(config.simulation.replan_interval),
                "--max-speed-mps",
                str(config.simulation.max_speed_mps),
                "--goal-radius-m",
                str(config.simulation.goal_radius_m),
                "--traffic-policy",
                config.simulation.traffic_policy,
            )
        )
        if config.evaluation.device is not None:
            args.extend(("--device", config.evaluation.device))
        if config.simulation.ttc_horizon_s is not None:
            args.extend(("--ttc-horizon-s", str(config.simulation.ttc_horizon_s)))
        if config.simulation.stop_on_collision:
            args.append("--stop-on-collision")
        if config.simulation.stop_on_goal:
            args.append("--stop-on-goal")
    else:
        args.extend(
            (
                "--history-frames",
                str(config.dataset.history_frames),
                "--frame-interval",
                str(config.dataset.frame_interval),
            )
        )
        if config.evaluation.device is not None:
            args.extend(("--device", config.evaluation.device))
    return args


def _score_distributed_args(config: ExperimentConfig) -> list[str]:
    assert config.evaluation.submission_dir is not None
    args = [
        "score-submission",
        config.dataset.data_list,
        "--submission-dir",
        config.evaluation.submission_dir,
        "--output-dir",
        config.output.directory,
        "--world-size",
        str(config.workers.world_size),
        "--workers",
        str(config.workers.workers),
        "--worker-backend",
        config.workers.backend,
        "--launcher-backend",
        config.workers.launcher_backend,
        "--max-rank-retries",
        str(config.workers.max_retries),
        "--max-retries",
        str(config.evaluation.max_retries),
        "--run-id",
        config.fingerprint[:24],
        "--backend",
        config.evaluation.backend,
        "--families",
        *config.evaluation.families,
        "--history-frames",
        str(config.dataset.history_frames),
        "--future-frames",
        str(config.dataset.future_frames),
        "--frame-interval",
        str(config.dataset.frame_interval),
    ]
    if config.workers.timeout_s is not None:
        args.extend(("--timeout-s", str(config.workers.timeout_s)))
    if not config.workers.resume:
        args.append("--no-resume")
    if config.evaluation.device is not None:
        args.extend(("--device", config.evaluation.device))
    if config.dataset.reader:
        args.extend(
            (
                "--reader-config-json",
                json.dumps(config.dataset.reader, sort_keys=True, separators=(",", ":")),
            )
        )
    return args


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
