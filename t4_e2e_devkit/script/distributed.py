"""Orchestrate a complete rank-partitioned evaluation on one machine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from t4_e2e_devkit.evaluation.orchestration import LocalDistributedLauncher


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="t4e2e distribute",
        description="launch and merge all ranks for an evaluation run",
    )
    parser.add_argument(
        "kind", choices=("evaluate", "closed-loop", "submit", "score-submission")
    )
    parser.add_argument("data_list")
    parser.add_argument("--agent", default=None)
    parser.add_argument(
        "--agent-params-json",
        default=None,
        help="JSON object forwarded to the registered agent constructor",
    )
    parser.add_argument(
        "--reader-config-json",
        default=None,
        help="JSON object forwarded to the T4 reader",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submission-dir", default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--worker-backend",
        choices=("serial", "thread", "process", "ray"),
        default="serial",
    )
    parser.add_argument("--launcher-backend", choices=("sequential", "process"), default="process")
    parser.add_argument("--max-rank-retries", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=0, help="retries for one data-list row")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--families", nargs="+", choices=("open_loop", "pdm"), default=None)
    parser.add_argument("--history-frames", type=int, default=None)
    parser.add_argument("--future-frames", type=int, default=None)
    parser.add_argument("--frame-interval", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--replan-interval", type=int, default=None)
    parser.add_argument("--max-speed-mps", type=float, default=None)
    parser.add_argument("--goal-radius-m", type=float, default=None)
    parser.add_argument("--ttc-horizon-s", type=float, default=None)
    parser.add_argument("--traffic-policy", choices=("replay", "constant_velocity", "idm"), default="replay")
    parser.add_argument("--stop-on-collision", action="store_true")
    parser.add_argument("--stop-on-goal", action="store_true")
    args = parser.parse_args(argv)
    if args.kind != "score-submission" and not args.agent:
        parser.error("--agent is required for evaluate, closed-loop and submit")
    if args.kind == "score-submission" and not args.submission_dir:
        parser.error("--submission-dir is required for score-submission")
    if args.agent_params_json is not None:
        try:
            agent_params = json.loads(args.agent_params_json)
        except json.JSONDecodeError as error:
            parser.error(f"--agent-params-json must be valid JSON: {error}")
        if not isinstance(agent_params, dict):
            parser.error("--agent-params-json must contain a JSON object")
        args.agent_params_json = json.dumps(agent_params, sort_keys=True, separators=(",", ":"))
    if args.reader_config_json is not None:
        try:
            reader_config = json.loads(args.reader_config_json)
        except json.JSONDecodeError as error:
            parser.error(f"--reader-config-json must be valid JSON: {error}")
        if not isinstance(reader_config, dict):
            parser.error("--reader-config-json must contain a JSON object")
        args.reader_config_json = json.dumps(reader_config, sort_keys=True, separators=(",", ":"))
    if args.world_size < 1 or args.workers < 1:
        parser.error("world-size and workers must be positive")
    if args.world_size > 1 and args.device is not None:
        device_name = str(args.device).lower()
        if device_name.startswith("cuda:"):
            parser.error("omit an indexed --device for multi-rank runs; one GPU is assigned per rank")

    launcher = LocalDistributedLauncher(
        args.output_dir,
        world_size=args.world_size,
        run_id=args.run_id,
        backend=args.launcher_backend,
        max_retries=args.max_rank_retries,
        timeout_s=args.timeout_s,
        resume=not args.no_resume,
        environment_factory=_gpu_environment_factory(args, parser),
    )
    command = _rank_command(args, launcher.run_id)
    if args.kind == "evaluate":
        from t4_e2e_devkit.script.merge_evaluation import merge_evaluation_reports

        def merger(rank_dirs, merged):
            return merge_evaluation_reports(rank_dirs, merged)
    elif args.kind == "closed-loop":
        from t4_e2e_devkit.script.merge_closed_loop import merge_closed_loop_reports

        def merger(rank_dirs, merged):
            return merge_closed_loop_reports(rank_dirs, merged)
    elif args.kind == "submit":
        from t4_e2e_devkit.evaluation.submission import SubmissionPackage

        def merger(rank_dirs, merged):
            return SubmissionPackage.merge(rank_dirs, merged)
    else:
        from t4_e2e_devkit.script.merge_submission_score import merge_submission_scores

        def merger(rank_dirs, merged):
            return merge_submission_scores(rank_dirs, merged)
    result = launcher.run(command, merger=merger)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.succeeded else 1


def _rank_command(args: argparse.Namespace, run_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "t4_e2e_devkit.cli",
        (
            "evaluate"
            if args.kind == "evaluate"
            else "evaluate-closed-loop"
            if args.kind == "closed-loop"
            else "score-submission"
            if args.kind == "score-submission"
            else "submit"
        ),
        str(Path(args.data_list)),
        "--output-dir",
        str(Path(args.output_dir)),
        "--workers",
        str(args.workers),
        "--worker-backend",
        args.worker_backend,
    ]
    if args.kind == "score-submission":
        command.extend(("--submission-dir", str(Path(args.submission_dir))))
    else:
        command.extend(("--agent", args.agent))
    if args.agent_params_json is not None and args.kind != "score-submission":
        command.extend(("--agent-params-json", args.agent_params_json))
    if args.reader_config_json is not None:
        command.extend(("--reader-config-json", args.reader_config_json))
    if args.kind in {"evaluate", "closed-loop"}:
        command.extend(("--run-id", run_id))
    if args.kind == "evaluate":
        if args.no_resume:
            command.append("--no-resume")
    elif args.kind == "closed-loop" and not args.no_resume:
        command.append("--resume")
    elif args.kind == "score-submission" and args.no_resume:
        command.append("--no-resume")
    if args.kind in {"evaluate", "closed-loop", "submit", "score-submission"}:
        command.extend(("--max-retries", str(args.max_retries)))
    if args.max_rows is not None:
        command.extend(("--max-rows", str(args.max_rows)))
    if args.checkpoint is not None and args.kind in {"evaluate", "closed-loop", "submit"}:
        command.extend(("--checkpoint", str(args.checkpoint)))
    if args.kind == "evaluate":
        command.extend(("--backend", args.backend))
        if args.device is not None:
            command.extend(("--device", args.device))
        if args.families:
            command.extend(("--families", *args.families))
        for name in ("history_frames", "future_frames", "frame_interval"):
            value = getattr(args, name)
            if value is not None:
                command.extend((f"--{name.replace('_', '-')}", str(value)))
    elif args.kind == "closed-loop":
        if args.device is not None:
            command.extend(("--device", str(args.device)))
        for name, flag in (
            ("history_frames", "--history-frames"),
            ("num_steps", "--num-steps"),
            ("replan_interval", "--replan-interval"),
            ("max_speed_mps", "--max-speed-mps"),
            ("goal_radius_m", "--goal-radius-m"),
            ("ttc_horizon_s", "--ttc-horizon-s"),
        ):
            value = getattr(args, name)
            if value is not None:
                command.extend((flag, str(value)))
        command.extend(("--traffic-policy", args.traffic_policy))
        if args.stop_on_collision:
            command.append("--stop-on-collision")
        if args.stop_on_goal:
            command.append("--stop-on-goal")
    elif args.kind == "submit":
        for name, flag in (
            ("history_frames", "--history-frames"),
            ("frame_interval", "--frame-interval"),
        ):
            value = getattr(args, name)
            if value is not None:
                command.extend((flag, str(value)))
        if args.device is not None:
            command.extend(("--device", args.device))
    else:
        command.extend(("--backend", args.backend))
        if args.device is not None:
            command.extend(("--device", args.device))
        if args.families:
            command.extend(("--families", *args.families))
        for name in ("history_frames", "future_frames", "frame_interval"):
            value = getattr(args, name)
            if value is not None:
                command.extend((f"--{name.replace('_', '-')}", str(value)))
    return command


def _gpu_environment_factory(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    """Assign one visible CUDA device to each rank when GPU execution is active."""

    if args.world_size <= 1:
        return None
    if args.device is not None and str(args.device).lower().startswith("cpu"):
        return None
    if args.kind == "evaluate" and args.backend == "cpu" and args.device is None:
        return None
    if args.kind == "score-submission" and args.backend == "cpu":
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = [item.strip() for item in visible.split(",") if item.strip()] if visible else []
    count = int(torch.cuda.device_count())
    if len(device_ids) < count:
        device_ids = [str(index) for index in range(count)]
    if count < args.world_size or len(device_ids) < args.world_size:
        parser.error(
            f"world-size={args.world_size} needs one CUDA device per rank, "
            f"but only {count} visible device(s) are available"
        )
    return lambda rank: {"CUDA_VISIBLE_DEVICES": device_ids[rank]}


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
