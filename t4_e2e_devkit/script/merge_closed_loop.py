"""Merge deterministic closed-loop evaluation ranks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import OmegaConf

from t4_e2e_devkit.evaluation.closed_loop import ClosedLoopMetrics
from t4_e2e_devkit.evaluation.closed_loop_artifact import (
    load_rollout_artifact,
    load_rollout_metrics,
    rollout_artifact_path,
    write_rollout_payload,
)
from t4_e2e_devkit.evaluation.closed_loop_report import (
    write_closed_loop_csv,
    write_closed_loop_ticks,
    write_static_html_report,
)
from t4_e2e_devkit.evaluation.distributed import WorkerManifest
from t4_e2e_devkit.evaluation.report import aggregate_evaluation

RUN_FORMAT = "t4.closed_loop.run"
RUN_VERSION = 1
_VOLATILE_RUN_KEYS = {
    "status",
    "config_fingerprint",
    "num_completed",
    "num_failed",
    "num_resumed",
    "num_attempts",
    "num_rows",
    "rank",
    "world_size",
    "rank_rows",
    "merged",
    "input_dirs",
    "source_world_size",
    "workers",
    "worker_backend",
    "manifest",
}


def merge_closed_loop_reports(
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, dict[str, float]]:
    """Merge completed rank directories and recompute their aggregate."""

    sources = [Path(value).resolve() for value in input_dirs]
    if not sources:
        raise ValueError("at least one input directory is required")
    destination = Path(output_dir).resolve()
    if destination in sources:
        raise ValueError("output directory must be different from every input directory")

    runs = [_load_run(source) for source in sources]
    signatures = [_run_signature(run) for run in runs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("closed-loop rank configurations do not match")

    source_world_size = int(runs[0].get("world_size", 1))
    ranks = [int(run.get("rank", 0)) for run in runs]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks: {ranks}")
    if not allow_incomplete:
        expected = set(range(source_world_size))
        actual = set(ranks)
        if source_world_size != len(sources) or actual != expected:
            raise ValueError(
                "incomplete rank set; expected ranks "
                f"{sorted(expected)}, got {sorted(actual)}. "
                "Pass --allow-incomplete to merge a subset explicitly."
            )

    entries: list[tuple[str, dict[str, Any], Optional[ClosedLoopMetrics], Path]] = []
    seen_tokens: set[str] = set()
    for source, run in zip(sources, runs, strict=True):
        artifact_dir = source / "rollouts"
        paths = sorted(artifact_dir.glob("*.json"))
        expected_rows = int(run.get("rank_rows", -1))
        if expected_rows >= 0 and len(paths) != expected_rows:
            raise ValueError(
                f"{source} contains {len(paths)} artifacts, but run.json declares "
                f"{expected_rows} rank rows"
            )
        expected_tokens = _manifest_tokens(source, run)
        actual_tokens: set[str] = set()
        for path in paths:
            payload = load_rollout_artifact(path)
            token = str(payload["token"])
            actual_tokens.add(token)
            if token in seen_tokens:
                raise ValueError(f"duplicate rollout token across ranks: {token}")
            seen_tokens.add(token)
            metrics = None
            if payload["status"] == "ok":
                metrics = load_rollout_metrics(path)
                if metrics is None:
                    raise ValueError(f"successful artifact has invalid metrics: {path}")
            entries.append((token, payload, metrics, path))
        if expected_tokens is not None and actual_tokens != expected_tokens:
            raise ValueError(f"rollout artifacts do not match the worker manifest in {source}")

    entries.sort(key=lambda entry: entry[0])
    metrics = [entry[2] for entry in entries if entry[2] is not None]
    failures = [
        (token, str(payload.get("error", "unknown error")))
        for token, payload, metric, _ in entries
        if metric is None
    ]

    base_config = dict(signatures[0])
    run_config: dict[str, Any] = {
        **base_config,
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "world_size": len(sources),
        "rank": None,
        "rank_rows": len(entries),
        "merged": True,
        "source_world_size": source_world_size,
    }
    config_fingerprint = _fingerprint(run_config)
    output = Path(output_dir)
    (output / "rollouts").mkdir(parents=True, exist_ok=True)
    for index, (token, payload, _, _) in enumerate(entries):
        merged_payload = dict(payload)
        merged_payload["config"] = run_config
        merged_payload["config_fingerprint"] = config_fingerprint
        write_rollout_payload(rollout_artifact_path(output / "rollouts", index, token), merged_payload)

    report = aggregate_evaluation(closed_loop=metrics, num_failed=len(failures))
    report["run"].update(
        {
            "num_rows": float(len(entries)),
            "num_completed": float(len(metrics)),
            "num_failed": float(len(failures)),
            "num_inputs": float(len(sources)),
            "source_world_size": float(source_world_size),
            "merged": 1.0,
        }
    )
    write_closed_loop_csv(output / "closed_loop.csv", metrics)
    write_closed_loop_ticks(output / "closed_loop_ticks.csv", metrics)
    _write_json(output / "aggregate.json", report)
    OmegaConf.save(OmegaConf.create(report), output / "aggregate.yaml")
    with (output / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "error"])
        writer.writerows(failures)
    _write_json(
        output / "run.json",
        {
            **run_config,
            "status": "completed",
            "config_fingerprint": config_fingerprint,
            "num_completed": len(metrics),
            "num_failed": len(failures),
            "num_rows": len(entries),
            "num_attempts": sum(int(run.get("num_attempts", 0)) for run in runs),
            "num_resumed": sum(int(run.get("num_resumed", 0)) for run in runs),
        },
    )
    write_static_html_report(output)
    return report


def _load_run(directory: Path) -> dict[str, Any]:
    try:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {directory / 'run.json'}") from error
    if not isinstance(run, dict) or run.get("format") != RUN_FORMAT:
        raise ValueError(f"not a closed-loop run directory: {directory}")
    if run.get("version") != RUN_VERSION:
        raise ValueError(f"unsupported closed-loop run version in {directory}")
    if run.get("status") != "completed":
        raise ValueError(f"run is not completed: {directory}")
    return run


def _manifest_tokens(directory: Path, run: Mapping[str, Any]) -> set[str] | None:
    manifest_name = run.get("manifest")
    if not manifest_name:
        return None
    path = directory / str(manifest_name)
    if not path.is_file():
        raise ValueError(f"closed-loop run is missing its worker manifest: {path}")
    manifest = WorkerManifest.read(path)
    if manifest.run_id != str(run.get("run_id")):
        raise ValueError(f"worker manifest belongs to a different run: {path}")
    if manifest.rank != int(run.get("rank", 0)):
        raise ValueError(f"worker manifest rank does not match run.json: {path}")
    return set(manifest.task_ids)


def _run_signature(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in _VOLATILE_RUN_KEYS}


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="t4e2e merge-closed-loop",
        description="Merge completed sensor-replay closed-loop rank reports.",
    )
    parser.add_argument("--input-dir", nargs="+", required=True, help="completed rank reports")
    parser.add_argument("--output-dir", required=True, help="merged report directory")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="merge a subset of the declared ranks",
    )
    args = parser.parse_args(argv)
    report = merge_closed_loop_reports(
        args.input_dir,
        args.output_dir,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
