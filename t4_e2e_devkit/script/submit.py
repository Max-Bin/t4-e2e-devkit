"""Generate a portable trajectory submission from a T4 data list."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from t4_e2e_devkit.agents.registry import build_agent
from t4_e2e_devkit.common.dataclasses import SceneFilter, Trajectory
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.evaluation.executor import rank_indices
from t4_e2e_devkit.evaluation.submission import SubmissionPackage, TrajectorySubmission
from t4_e2e_devkit.evaluation.worker_pool import WorkerPool, WorkerTask
from t4_e2e_devkit.script.utils import file_digest, load_agent_checkpoint


def generate_submission(
    data_list: DataList | str | Path,
    *,
    agent_name: str,
    output_dir: str | Path,
    agent_params: Optional[Mapping[str, Any]] = None,
    checkpoint_path: str | Path | None = None,
    history_frames: int = 31,
    frame_interval: int = 5,
    max_rows: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    workers: int = 1,
    worker_backend: str = "serial",
    device: str | None = None,
    reader_config: Optional[Mapping[str, Any]] = None,
    max_retries: int = 0,
) -> dict[str, Any]:
    """Generate one rank's package and return its validation summary."""

    selected = data_list if isinstance(data_list, DataList) else load_data_list(data_list)
    if max_rows is not None:
        selected = selected.filtered(max_rows=max_rows)
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError("invalid rank/world_size")
    if workers < 1:
        raise ValueError("workers must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if worker_backend not in {"serial", "thread", "process", "ray"}:
        raise ValueError("worker_backend must be serial, thread, process or ray")
    if worker_backend == "process" and workers > 1:
        import torch

        active_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(active_device).lower().startswith("cuda"):
            raise ValueError(
                "GPU submission uses one process; use rank/world_size for parallel GPUs"
            )
    if checkpoint_path is not None and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    assigned = [selected[index] for index in rank_indices(len(selected), rank, world_size)]
    tasks = [
        WorkerTask(
            task_id=f"{scene}@{int(center)}",
            function=_predict_with_retries,
            args=(
                agent_name,
                dict(agent_params or {}),
                None if checkpoint_path is None else str(checkpoint_path),
                selected.root,
                scene,
                int(center),
                history_frames,
                frame_interval,
                device,
                dict(reader_config or {}),
                int(max_retries),
            ),
        )
        for scene, center in assigned
    ]
    with WorkerPool(
        workers=workers,
        rank=0,
        world_size=1,
        backend=worker_backend,
    ) as pool:
        results = pool.run_tasks(tasks)
    entries: list[TrajectorySubmission] = []
    failures: list[dict[str, str]] = []
    for result in results:
        if result.error is not None:
            failures.append({"token": result.task_id, "error": result.error})
            continue
        if not isinstance(result.value, Trajectory):
            failures.append({"token": result.task_id, "error": "agent did not return Trajectory"})
            continue
        entries.append(TrajectorySubmission(result.task_id, result.value))
    package = SubmissionPackage(
        tuple(entries),
        metadata={
            "format": "t4.internal.submission",
            "agent": str(agent_name),
            "agent_params": dict(agent_params or {}),
            "checkpoint_digest": file_digest(checkpoint_path),
            "reader_config_digest": _mapping_digest(dict(reader_config or {})),
            "data_digest": _digest_rows(selected),
            "rank": int(rank),
            "world_size": int(world_size),
            "history_frames": int(history_frames),
            "frame_interval": int(frame_interval),
            "max_retries": int(max_retries),
        },
    )
    output = package.write(output_dir)
    (output / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected = [f"{scene}@{int(center)}" for scene, center in assigned]
    validation = package.validate(expected_tokens=expected)
    summary = {
        "status": "completed" if validation.valid and not failures else "failed",
        "num_expected": len(expected),
        "num_predictions": len(entries),
        "num_failed": len(failures),
        "validation": validation.as_dict(),
        "output_dir": str(output),
    }
    (output / "status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _predict_with_retries(
    *args: Any,
) -> Trajectory:
    *configuration, max_retries = args
    last_error: Optional[BaseException] = None
    for _attempt in range(int(max_retries) + 1):
        try:
            return _predict_one(*configuration)
        except Exception as error:  # preserve the worker task boundary
            last_error = error
    assert last_error is not None
    raise last_error


def _predict_one(
    agent_name: str,
    agent_params: Mapping[str, Any],
    checkpoint_path: str | None,
    root: str | Path,
    scene: str,
    center: int,
    history_frames: int,
    frame_interval: int,
    device: str | None,
    reader_config: Mapping[str, Any],
) -> Trajectory:
    import torch

    from t4_e2e_devkit.dataset.window import T4WindowBuilder

    agent = build_agent(agent_name, **dict(agent_params))
    agent.initialize()
    if checkpoint_path is not None:
        load_agent_checkpoint(agent, checkpoint_path)
    active_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(torch.device(active_device))
    builder = T4WindowBuilder(
        Path(root) / scene,
        root,
        sensor_config=agent.get_sensor_config(),
        scene_filter=SceneFilter(
            num_history_frames=history_frames,
            num_future_frames=0,
            frame_interval=frame_interval,
            has_route=True,
        ),
        reader_config=dict(reader_config),
    )
    try:
        window = builder.build(int(center))
        if getattr(agent, "requires_scene", False):
            raise ValueError(
                "submission agents must plan from T4AgentInput, not privileged scene data"
            )
        with torch.inference_mode():
            trajectory = agent.compute_trajectory(window.get_agent_input())
        if not isinstance(trajectory, Trajectory):
            raise TypeError(f"agent returned {type(trajectory).__name__}, expected Trajectory")
        return trajectory
    finally:
        builder.close()


def _digest_rows(data_list: DataList) -> str:
    payload = json.dumps(list(data_list.rows), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e submit")
    parser.add_argument("data_list")
    parser.add_argument("--agent", required=True)
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--history-frames", type=int, default=31)
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--worker-backend", choices=("serial", "thread", "process", "ray"), default="serial"
    )
    parser.add_argument("--max-retries", type=int, default=0)
    args = parser.parse_args(argv)
    agent_params: Mapping[str, Any] = {}
    if args.agent_params_json is not None:
        try:
            value = json.loads(args.agent_params_json)
        except json.JSONDecodeError as error:
            parser.error(f"--agent-params-json must be valid JSON: {error}")
        if not isinstance(value, Mapping):
            parser.error("--agent-params-json must contain a JSON object")
        agent_params = dict(value)
    reader_config: Mapping[str, Any] = {}
    if args.reader_config_json is not None:
        try:
            value = json.loads(args.reader_config_json)
        except json.JSONDecodeError as error:
            parser.error(f"--reader-config-json must be valid JSON: {error}")
        if not isinstance(value, Mapping):
            parser.error("--reader-config-json must contain a JSON object")
        reader_config = dict(value)
    summary = generate_submission(
        args.data_list,
        agent_name=args.agent,
        agent_params=agent_params,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        history_frames=args.history_frames,
        frame_interval=args.frame_interval,
        max_rows=args.max_rows,
        rank=args.rank,
        world_size=args.world_size,
        workers=args.workers,
        worker_backend=args.worker_backend,
        device=args.device,
        reader_config=reader_config,
        max_retries=args.max_retries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_submission", "main"]
