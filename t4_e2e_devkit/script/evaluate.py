"""Run the independent T4 evaluation families on a data list."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import OmegaConf

from t4_e2e_devkit.agents.registry import build_agent
from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.evaluation.batch import (
    RUN_FORMAT,
    RUN_VERSION,
    aggregate_records,
    fingerprint,
    read_record,
    record_path,
    write_family_csv,
    write_json,
)
from t4_e2e_devkit.evaluation.batch import config_fingerprint as make_config_fingerprint
from t4_e2e_devkit.evaluation.distributed import WorkerManifest
from t4_e2e_devkit.evaluation.executor import rank_indices
from t4_e2e_devkit.evaluation.navsim_score import resolve_navsim_metric_names
from t4_e2e_devkit.evaluation.worker_pool import WorkerPool, WorkerResult, WorkerTask
from t4_e2e_devkit.script.utils import file_digest, load_agent_checkpoint

logger = logging.getLogger(__name__)
FAMILIES = ("open_loop", "pdm")
PDM_VERSIONS = ("navsim-v1", "navsim-v2")


def evaluate_data_list(
    data_list: DataList | str | Path,
    *,
    agent_name: str,
    output_dir: str | Path,
    families: Sequence[str] = FAMILIES,
    agent_params: Optional[Mapping[str, Any]] = None,
    checkpoint_path: str | Path | None = None,
    backend: str = "auto",
    device: str | None = None,
    history_frames: int = 31,
    future_frames: int = 80,
    frame_interval: int = 5,
    max_rows: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    workers: int = 1,
    worker_backend: str = "serial",
    max_retries: int = 0,
    resume: bool = True,
    run_id: str | None = None,
    reader_config: Optional[Mapping[str, Any]] = None,
    pdm_version: str = "navsim-v2",
    pdm_metric_names: Optional[Sequence[str]] = None,
    pdm_previous_interval_frames: int | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate one deterministic rank of a T4 data list.

    Every worker receives plain configuration values and opens its own scene
    reader. This keeps process execution safe and avoids serializing CUDA or
    open file handles. ``output_dir`` is a rank directory when ``world_size``
    is greater than one; rank reports are merged separately.
    """

    selected = data_list if isinstance(data_list, DataList) else load_data_list(data_list)
    if max_rows is not None:
        selected = selected.filtered(max_rows=max_rows)
    family_names = tuple(dict.fromkeys(str(name) for name in families))
    unknown = sorted(set(family_names) - set(FAMILIES))
    if not family_names or unknown:
        raise ValueError(
            f"families must be a non-empty subset of {FAMILIES}; got {unknown or family_names}"
        )
    pdm_version = str(pdm_version).lower()
    if pdm_version not in PDM_VERSIONS:
        raise ValueError(f"pdm_version must be one of {PDM_VERSIONS}")
    pdm_metric_names = (
        None
        if pdm_metric_names is None
        else resolve_navsim_metric_names(pdm_version.removeprefix("navsim-"), pdm_metric_names)
    )
    if pdm_previous_interval_frames is not None and pdm_previous_interval_frames < 1:
        raise ValueError("pdm_previous_interval_frames must be positive")
    backend = _resolve_backend(backend)
    if workers < 1 or max_retries < 0:
        raise ValueError("workers must be positive and max_retries must be non-negative")
    if worker_backend not in {"serial", "thread", "process", "ray"}:
        raise ValueError("worker_backend must be serial, thread, process or ray")
    if backend == "gpu" and worker_backend == "process" and workers > 1:
        raise ValueError("GPU evaluation uses one process; use rank/world_size for multiple GPUs")
    if checkpoint_path is not None and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    rank_indices(len(selected), rank, world_size)

    resolved_reader_config = dict(reader_config or {})
    checkpoint_digest = file_digest(checkpoint_path)
    data_digest = fingerprint(list(selected.rows))

    output = Path(output_dir)
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "agent": str(agent_name),
        "families": list(family_names),
        "backend": backend,
        "device": device,
        "history_frames": int(history_frames),
        "future_frames": int(future_frames),
        "frame_interval": int(frame_interval),
        "agent_params": dict(agent_params or {}),
        "checkpoint_digest": checkpoint_digest,
        "data_digest": data_digest,
        "reader_config_digest": fingerprint(resolved_reader_config),
        "selected_rows": len(selected),
        "rank": int(rank),
        "world_size": int(world_size),
        "workers": int(workers),
        "worker_backend": worker_backend,
        "max_retries": int(max_retries),
        "pdm_version": pdm_version,
        "pdm_metric_names": None if pdm_metric_names is None else list(pdm_metric_names),
        "pdm_previous_interval_frames": pdm_previous_interval_frames,
    }
    base_config = dict(config)
    if run_id is None:
        run_id = f"evaluation-{make_config_fingerprint(base_config)[:16]}"
    config["run_id"] = str(run_id)
    resolved_fingerprint = make_config_fingerprint(base_config)
    write_json(
        output / "run.json",
        {**config, "status": "running", "config_fingerprint": resolved_fingerprint},
    )

    assigned = [selected[index] for index in rank_indices(len(selected), rank, world_size)]
    previous: dict[str, dict[str, Any]] = {}
    if resume:
        for scene, center in assigned:
            token = f"{scene}@{int(center)}"
            cached = read_record(
                record_path(records_dir, token), config_fingerprint=resolved_fingerprint
            )
            if cached is not None:
                previous[token] = cached

    scene_filter = {
        "num_history_frames": int(history_frames),
        "num_future_frames": int(future_frames),
        "frame_interval": int(frame_interval),
        "has_route": True,
    }
    task_args = (
        str(agent_name),
        dict(agent_params or {}),
        None if checkpoint_path is None else str(checkpoint_path),
        backend,
        device,
        scene_filter,
        str(selected.root),
        family_names,
        resolved_reader_config,
        pdm_version,
        pdm_metric_names,
        pdm_previous_interval_frames,
    )
    tasks = [
        WorkerTask(
            task_id=f"{scene}@{int(center)}",
            function=_evaluate_with_retries,
            args=(*task_args, scene, int(center), int(max_retries)),
        )
        for scene, center in assigned
    ]
    previous_ids = tuple(previous)
    with WorkerPool(
        workers=workers,
        rank=0,
        world_size=1,
        backend=worker_backend,
    ) as pool:
        results = pool.run_tasks(tasks, skip_task_ids=previous_ids)

    records = list(previous.values())
    failures: list[tuple[str, str]] = []
    attempts_total = 0
    manifest_results: list[WorkerResult] = [
        WorkerResult(task_id=token, value={"status": "ok", "resumed": True}, rank=rank)
        for token in previous
    ]
    for result in results:
        token = result.task_id
        value = result.value if isinstance(result.value, Mapping) else {}
        if result.error is not None:
            value = {"status": "failed", "error": result.error, "attempts": 0}
        attempts_total += int(value.get("attempts", 0))
        if value.get("status") == "ok":
            record = {
                "format": "t4.evaluation.record",
                "version": 1,
                "status": "ok",
                "token": token,
                "families": value.get("families", {}),
                "attempts": int(value.get("attempts", 1)),
                "config_fingerprint": resolved_fingerprint,
            }
            write_json(record_path(records_dir, token), record)
            records.append(record)
            manifest_results.append(WorkerResult(task_id=token, value={"status": "ok"}, rank=rank))
        else:
            error = str(value.get("error", "evaluation failed"))
            failures.append((token, error))
            write_json(
                record_path(records_dir, token),
                {
                    "format": "t4.evaluation.record",
                    "version": 1,
                    "status": "failed",
                    "token": token,
                    "error": error,
                    "attempts": int(value.get("attempts", max_retries + 1)),
                    "config_fingerprint": resolved_fingerprint,
                },
            )
            manifest_results.append(
                WorkerResult(task_id=token, value={"status": "failed"}, rank=rank, error=error)
            )

    records.sort(key=lambda item: str(item["token"]))
    report = aggregate_records(records, num_failed=len(failures))
    write_family_csv(output, records)
    write_json(output / "aggregate.json", report)
    OmegaConf.save(OmegaConf.create(report), output / "aggregate.yaml")
    with (output / "failures.csv").open("w", newline="", encoding="utf-8") as stream:
        import csv

        writer = csv.writer(stream)
        writer.writerow(["token", "error"])
        writer.writerows(failures)

    manifest = WorkerManifest(
        run_id=str(run_id),
        rank=int(rank),
        world_size=int(world_size),
        task_ids=tuple(token for scene, center in assigned for token in [f"{scene}@{int(center)}"]),
        results=tuple(sorted(manifest_results, key=lambda item: item.task_id)),
    )
    manifest_path = output / f"worker-manifest-rank-{rank}.json"
    manifest.write(manifest_path)
    write_json(
        output / "run.json",
        {
            **config,
            "status": "failed" if failures else "completed",
            "config_fingerprint": resolved_fingerprint,
            "rank_rows": len(assigned),
            "num_completed": len(records),
            "num_failed": len(failures),
            "num_resumed": len(previous),
            "num_attempts": attempts_total,
            "manifest": manifest_path.name,
        },
    )
    return report


def _evaluate_with_retries(*args: Any) -> dict[str, Any]:
    *configuration, scene, center, max_retries = args
    last_error = "evaluation failed"
    for attempt in range(1, int(max_retries) + 2):
        try:
            value = _evaluate_one(*configuration, scene, center)
            return {"status": "ok", "families": value, "attempts": attempt}
        except Exception as error:  # noqa: BLE001 - row failures are report data
            last_error = f"{type(error).__name__}: {error}"
            logger.warning("evaluation failed for %s@%s (%d): %s", scene, center, attempt, error)
    return {"status": "failed", "error": last_error, "attempts": int(max_retries) + 1}


def _evaluate_one(
    agent_name: str,
    agent_params: Mapping[str, Any],
    checkpoint_path: str | None,
    backend: str,
    device: str | None,
    scene_filter_values: Mapping[str, Any],
    root: str,
    families: Sequence[str],
    reader_config: Mapping[str, Any],
    pdm_version: str,
    pdm_metric_names: Optional[Sequence[str]],
    pdm_previous_interval_frames: int | None,
    scene: str,
    center: int,
) -> dict[str, Mapping[str, float]]:
    import torch

    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.evaluation.navsim_score import (
        T4NavSimScorer,
        T4NavSimScorerConfig,
    )
    from t4_e2e_devkit.evaluation.open_loop import compute_open_loop_metrics

    agent = build_agent(agent_name, **dict(agent_params))
    agent.initialize()
    if checkpoint_path:
        load_agent_checkpoint(agent, checkpoint_path)
    active_device = device or ("cuda" if backend == "gpu" else "cpu")
    agent.to(torch.device(active_device))
    builder = T4WindowBuilder(
        Path(root) / scene,
        root,
        sensor_config=agent.get_sensor_config(),
        scene_filter=SceneFilter(**dict(scene_filter_values)),
        reader_config=dict(reader_config),
    )
    try:
        window = builder.build(int(center))

        def predict(value):
            with torch.inference_mode():
                return (
                    agent.compute_trajectory_from_scene(value)
                    if getattr(agent, "requires_scene", False)
                    else agent.compute_trajectory(value.get_agent_input())
                )

        trajectory = predict(window)
        values: dict[str, Mapping[str, float]] = {}
        if "open_loop" in families:
            values["open_loop"] = compute_open_loop_metrics(
                trajectory, window, token=window.scene_metadata.token
            ).values
        if "pdm" in families:
            previous_trajectory = None
            previous_window = None
            previous_builder = None
            previous_interval = pdm_previous_interval_frames or int(
                scene_filter_values["frame_interval"]
            )
            previous_center = int(center) - previous_interval
            first_valid_center = int(scene_filter_values["num_history_frames"]) - 1
            if pdm_version == "navsim-v2" and previous_center >= first_valid_center:
                previous_builder = T4WindowBuilder(
                    Path(root) / scene,
                    root,
                    sensor_config=agent.get_sensor_config(),
                    scene_filter=SceneFilter(**dict(scene_filter_values)),
                    reader_config=dict(reader_config),
                )
                previous_window = previous_builder.build(previous_center)
                previous_trajectory = predict(previous_window)
            try:
                scorer = T4NavSimScorer(
                    T4NavSimScorerConfig(
                        version=pdm_version.removeprefix("navsim-"),
                        backend=backend,
                        device=active_device,
                        metric_names=pdm_metric_names,
                    )
                )
                result = scorer.score(
                    trajectory,
                    window,
                    previous_trajectory=previous_trajectory,
                    previous_scene=previous_window,
                )
                values["pdm"] = {"pdm_version": pdm_version, **result.values}
            finally:
                if previous_builder is not None:
                    previous_builder.close()
        return values
    finally:
        builder.close()


def _resolve_backend(backend: str) -> str:
    """Resolve the portable default without hiding an explicit GPU error."""

    requested = str(backend).lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be auto, cpu or gpu")
    if requested == "auto":
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    if requested == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise ValueError(
                "backend='gpu' needs CUDA; pass backend='cpu' to request the audit path"
            )
    return requested


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e evaluate")
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
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pdm-version", choices=PDM_VERSIONS, default="navsim-v2")
    parser.add_argument(
        "--pdm-metrics",
        nargs="+",
        default=None,
        help="PDM metrics to compute and report; omit to use all metrics for the version",
    )
    parser.add_argument("--pdm-previous-interval-frames", type=int, default=None)
    parser.add_argument("--maps-root", default=None)
    parser.add_argument("--scene-tags-root", default=None)
    parser.add_argument("--attach-map-ids", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--history-frames", type=int, default=31)
    parser.add_argument("--future-frames", type=int, default=80)
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--worker-backend",
        choices=("serial", "thread", "process", "ray"),
        default="serial",
    )
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--run-id", default=None)
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
    reader_config: dict[str, Any] = {}
    if args.reader_config_json is not None:
        try:
            value = json.loads(args.reader_config_json)
        except json.JSONDecodeError as error:
            parser.error(f"--reader-config-json must be valid JSON: {error}")
        if not isinstance(value, Mapping):
            parser.error("--reader-config-json must contain a JSON object")
        reader_config.update(value)
    reader_config.update(
        {
            key: value
            for key, value in {
                "t4_maps_root": args.maps_root,
                "t4_scene_tags_root": args.scene_tags_root,
                "t4_attach_map_ids": args.attach_map_ids,
            }.items()
            if value not in (None, False)
        }
    )
    report = evaluate_data_list(
        args.data_list,
        agent_name=args.agent,
        agent_params=agent_params,
        output_dir=args.output_dir,
        families=args.families,
        checkpoint_path=args.checkpoint,
        reader_config=reader_config,
        backend=args.backend,
        device=args.device,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        frame_interval=args.frame_interval,
        max_rows=args.max_rows,
        rank=args.rank,
        world_size=args.world_size,
        workers=args.workers,
        worker_backend=args.worker_backend,
        max_retries=args.max_retries,
        resume=not args.no_resume,
        run_id=args.run_id,
        pdm_version=args.pdm_version,
        pdm_metric_names=args.pdm_metrics,
        pdm_previous_interval_frames=args.pdm_previous_interval_frames,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("run", {}).get("num_failed", 0.0) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
