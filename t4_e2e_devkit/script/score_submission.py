"""Score a portable trajectory submission against T4 ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import OmegaConf

from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.evaluation.batch import (
    aggregate_records,
    config_fingerprint,
    fingerprint,
    read_record,
    record_path,
    write_family_csv,
    write_json,
)
from t4_e2e_devkit.evaluation.distributed import WorkerManifest
from t4_e2e_devkit.evaluation.executor import rank_indices
from t4_e2e_devkit.evaluation.navsim_score import resolve_navsim_metric_names
from t4_e2e_devkit.evaluation.submission import SubmissionPackage
from t4_e2e_devkit.evaluation.worker_pool import WorkerPool, WorkerResult, WorkerTask
from t4_e2e_devkit.script.evaluate import FAMILIES, PDM_VERSIONS, _resolve_backend


def score_submission(
    data_list: DataList | str | Path,
    *,
    submission_dir: str | Path,
    output_dir: str | Path,
    families: Sequence[str] = FAMILIES,
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
    reader_config: Optional[Mapping[str, Any]] = None,
    resume: bool = True,
    pdm_version: str = "navsim-v2",
    pdm_metric_names: Optional[Sequence[str]] = None,
    pdm_previous_interval_frames: int | None = None,
) -> dict[str, dict[str, float]]:
    selected = data_list if isinstance(data_list, DataList) else load_data_list(data_list)
    if max_rows is not None:
        selected = selected.filtered(max_rows=max_rows)
    package = SubmissionPackage.read(submission_dir)
    expected_tokens = [f"{scene}@{int(center)}" for scene, center in selected.rows]
    package.validate(expected_tokens=expected_tokens).raise_for_errors()
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
        raise ValueError(
            "GPU submission scoring uses one process; use rank/world_size for parallel GPUs"
        )
    assigned = [selected[index] for index in rank_indices(len(selected), rank, world_size)]
    output = Path(output_dir)
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "t4.submission-score.run",
        "version": 1,
        "submission": str(Path(submission_dir).name),
        "families": list(family_names),
        "backend": backend,
        "device": device,
        "history_frames": history_frames,
        "future_frames": future_frames,
        "frame_interval": frame_interval,
        "submission_digest": fingerprint([entry.as_dict() for entry in package.entries]),
        "data_digest": fingerprint(list(selected.rows)),
        "reader_config_digest": fingerprint(dict(reader_config or {})),
        "rank": rank,
        "world_size": world_size,
        "workers": workers,
        "worker_backend": worker_backend,
        "max_retries": max_retries,
        "pdm_version": pdm_version,
        "pdm_metric_names": None if pdm_metric_names is None else list(pdm_metric_names),
        "pdm_previous_interval_frames": pdm_previous_interval_frames,
    }
    resolved_fingerprint = config_fingerprint(config)
    run_id = f"submission-score-{resolved_fingerprint[:16]}"
    write_json(
        output / "run.json",
        {
            **config,
            "run_id": run_id,
            "status": "running",
            "config_fingerprint": resolved_fingerprint,
        },
    )
    tasks = [
        WorkerTask(
            task_id=f"{scene}@{int(center)}",
            function=_score_one_with_retries,
            args=(
                package.entry(f"{scene}@{int(center)}").trajectory,
                (
                    package.entry(
                        f"{scene}@{int(center) - (pdm_previous_interval_frames or frame_interval)}"
                    ).trajectory
                    if (
                        pdm_version == "navsim-v2"
                        and f"{scene}@{int(center) - (pdm_previous_interval_frames or frame_interval)}"
                        in package.tokens
                    )
                    else None
                ),
                selected.root,
                scene,
                int(center),
                family_names,
                backend,
                device,
                SceneFilter(
                    num_history_frames=history_frames,
                    num_future_frames=future_frames,
                    frame_interval=frame_interval,
                    has_route=True,
                ),
                dict(reader_config or {}),
                pdm_version,
                pdm_metric_names,
                pdm_previous_interval_frames,
                max_retries,
            ),
        )
        for scene, center in assigned
    ]
    previous: dict[str, dict[str, Any]] = {}
    if resume:
        for scene, center in assigned:
            token = f"{scene}@{int(center)}"
            cached = read_record(
                record_path(records_dir, token),
                config_fingerprint=resolved_fingerprint,
            )
            if cached is not None and str(cached.get("token")) == token:
                previous[token] = cached
    previous_ids = tuple(previous)
    with WorkerPool(workers=workers, rank=0, world_size=1, backend=worker_backend) as pool:
        results = pool.run_tasks(tasks, skip_task_ids=previous_ids)
    records: list[dict[str, Any]] = list(previous.values())
    failures: list[tuple[str, str]] = []
    manifest_results: list[WorkerResult] = [
        WorkerResult(task_id=token, value={"status": "ok", "resumed": True}, rank=rank)
        for token in previous
    ]
    for result in results:
        value = result.value if isinstance(result.value, Mapping) else {}
        if result.error is not None:
            value = {"status": "failed", "error": result.error, "attempts": 0}
        token = result.task_id
        attempts = int(value.get("attempts", 0))
        if value.get("status") == "ok":
            record = {
                "format": "t4.submission-score.record",
                "version": 1,
                "status": "ok",
                "token": token,
                "families": value.get("families", {}),
                "attempts": attempts or 1,
                "config_fingerprint": resolved_fingerprint,
            }
            records.append(record)
            write_json(record_path(records_dir, token), record)
            manifest_results.append(WorkerResult(task_id=token, value={"status": "ok"}, rank=rank))
        else:
            error = str(value.get("error", "scoring failed"))
            failures.append((token, error))
            write_json(
                record_path(records_dir, token),
                {
                    "format": "t4.submission-score.record",
                    "version": 1,
                    "status": "failed",
                    "token": token,
                    "error": error,
                    "attempts": attempts or max_retries + 1,
                    "config_fingerprint": resolved_fingerprint,
                },
            )
            manifest_results.append(
                WorkerResult(task_id=token, value={"status": "failed"}, rank=rank, error=error)
            )
    records.sort(key=lambda value: str(value["token"]))
    report = aggregate_records(records, num_failed=len(failures))
    write_family_csv(output, records)
    write_json(output / "aggregate.json", report)
    OmegaConf.save(OmegaConf.create(report), output / "aggregate.yaml")
    with (output / "failures.csv").open("w", encoding="utf-8") as stream:
        stream.write("token,error\n")
        for token, error in failures:
            stream.write(
                f'"{token.replace(chr(34), chr(34) * 2)}","{error.replace(chr(34), chr(34) * 2)}"\n'
            )
    manifest = WorkerManifest(
        run_id=run_id,
        rank=rank,
        world_size=world_size,
        task_ids=tuple(f"{scene}@{int(center)}" for scene, center in assigned),
        results=tuple(sorted(manifest_results, key=lambda value: value.task_id)),
    )
    manifest_path = output / f"worker-manifest-rank-{rank}.json"
    manifest.write(manifest_path)
    write_json(
        output / "run.json",
        {
            **config,
            "run_id": run_id,
            "status": "failed" if failures else "completed",
            "config_fingerprint": resolved_fingerprint,
            "rank_rows": len(assigned),
            "num_completed": len(records),
            "num_failed": len(failures),
            "num_resumed": len(previous),
            "manifest": manifest_path.name,
        },
    )
    return report


def _score_one_with_retries(*args: Any) -> dict[str, Any]:
    *configuration, max_retries = args
    last_error = "scoring failed"
    for attempt in range(1, int(max_retries) + 2):
        try:
            return {"status": "ok", "families": _score_one(*configuration), "attempts": attempt}
        except Exception as error:  # noqa: BLE001 - row failure is report data
            last_error = f"{type(error).__name__}: {error}"
    return {"status": "failed", "error": last_error, "attempts": int(max_retries) + 1}


def _score_one(
    trajectory,
    previous_trajectory,
    root: str | Path,
    scene: str,
    center: int,
    families: Sequence[str],
    backend: str,
    device: str | None,
    scene_filter: SceneFilter,
    reader_config: Mapping[str, Any],
    pdm_version: str,
    pdm_metric_names: Optional[Sequence[str]],
    pdm_previous_interval_frames: int | None,
) -> dict[str, Mapping[str, float]]:
    from t4_e2e_devkit.common.dataclasses import SensorConfig
    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.evaluation.navsim_score import (
        T4NavSimScorer,
        T4NavSimScorerConfig,
    )
    from t4_e2e_devkit.evaluation.open_loop import compute_open_loop_metrics

    builder = T4WindowBuilder(
        Path(root) / scene,
        root,
        sensor_config=SensorConfig.build_no_sensors(),
        scene_filter=scene_filter,
        reader_config=dict(reader_config),
    )
    try:
        window = builder.build(int(center))
        values: dict[str, Mapping[str, float]] = {}
        if "open_loop" in families:
            values["open_loop"] = compute_open_loop_metrics(trajectory, window).values
        if "pdm" in families:
            previous_window = None
            previous_center = int(center) - (
                pdm_previous_interval_frames or int(scene_filter.frame_interval)
            )
            if previous_trajectory is not None and pdm_version == "navsim-v2":
                if previous_center >= scene_filter.num_history_frames - 1:
                    previous_window = builder.build(previous_center)
                else:
                    previous_trajectory = None
            result = T4NavSimScorer(
                T4NavSimScorerConfig(
                    version=pdm_version.removeprefix("navsim-"),
                    backend=backend,
                    device=device,
                    metric_names=pdm_metric_names,
                )
            ).score(
                trajectory,
                window,
                previous_trajectory=previous_trajectory,
                previous_scene=previous_window,
            )
            values["pdm"] = {"pdm_version": pdm_version, **result.values}
        return values
    finally:
        builder.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e score-submission")
    parser.add_argument("data_list")
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--history-frames", type=int, default=31)
    parser.add_argument("--future-frames", type=int, default=80)
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--pdm-version", choices=PDM_VERSIONS, default="navsim-v2")
    parser.add_argument(
        "--pdm-metrics",
        nargs="+",
        default=None,
        help="PDM metrics to compute and report; omit to use all metrics for the version",
    )
    parser.add_argument("--pdm-previous-interval-frames", type=int, default=None)
    parser.add_argument(
        "--reader-config-json",
        default=None,
        help="JSON object forwarded to the T4 reader",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--worker-backend", choices=("serial", "thread", "process", "ray"), default="serial"
    )
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    reader_config: Mapping[str, Any] = {}
    if args.reader_config_json is not None:
        try:
            value = json.loads(args.reader_config_json)
        except json.JSONDecodeError as error:
            parser.error(f"--reader-config-json must be valid JSON: {error}")
        if not isinstance(value, Mapping):
            parser.error("--reader-config-json must contain a JSON object")
        reader_config = dict(value)
    report = score_submission(
        args.data_list,
        submission_dir=args.submission_dir,
        output_dir=args.output_dir,
        families=args.families,
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
        reader_config=reader_config,
        resume=not args.no_resume,
        pdm_version=args.pdm_version,
        pdm_metric_names=args.pdm_metrics,
        pdm_previous_interval_frames=args.pdm_previous_interval_frames,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("run", {}).get("num_failed", 0.0) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["score_submission", "main"]
