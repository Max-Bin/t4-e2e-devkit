"""Merge rank-local trajectory-submission score reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import OmegaConf

from t4_e2e_devkit.evaluation.batch import (
    aggregate_records,
    config_fingerprint,
    record_path,
    write_family_csv,
    write_json,
)
from t4_e2e_devkit.evaluation.distributed import WorkerManifest

RUN_FORMAT = "t4.submission-score.run"
RUN_VERSION = 1
_VOLATILE = {
    "status",
    "config_fingerprint",
    "rank",
    "rank_rows",
    "world_size",
    "workers",
    "worker_backend",
    "manifest",
    "num_completed",
    "num_failed",
    "merged",
    "input_dirs",
}


def merge_submission_scores(
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, dict[str, float]]:
    """Merge completed rank score directories and recompute aggregates."""

    sources = [Path(value).resolve() for value in input_dirs]
    if not sources:
        raise ValueError("at least one input directory is required")
    destination = Path(output_dir).resolve()
    if destination in sources:
        raise ValueError("output directory must differ from every input directory")

    runs = [_read_run(source) for source in sources]
    signatures = [
        {key: value for key, value in run.items() if key not in _VOLATILE}
        for run in runs
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("submission-score rank configurations do not match")

    declared_world_size = int(runs[0].get("world_size", 1))
    ranks = [int(run.get("rank", 0)) for run in runs]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks: {ranks}")
    if not allow_incomplete and (
        len(sources) != declared_world_size or set(ranks) != set(range(declared_world_size))
    ):
        raise ValueError(
            f"incomplete rank set; expected {list(range(declared_world_size))}, "
            f"got {sorted(ranks)}"
        )

    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, run in zip(sources, runs, strict=True):
        expected_tokens = _manifest_tokens(source, run)
        actual_tokens: set[str] = set()
        for path in sorted((source / "records").glob("record-*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read submission score record {path}") from error
            if not isinstance(record, dict) or not record.get("token"):
                raise ValueError(f"invalid submission score record: {path}")
            token = str(record["token"])
            actual_tokens.add(token)
            if token in seen:
                raise ValueError(f"duplicate submission score token across ranks: {token}")
            seen.add(token)
            if record.get("config_fingerprint") != run.get("config_fingerprint"):
                raise ValueError(f"submission score record has a stale configuration: {path}")
            if record.get("status") == "ok":
                records.append(record)
            else:
                failures.append((token, str(record.get("error", "scoring failed"))))
        if expected_tokens is not None and actual_tokens != expected_tokens:
            raise ValueError(f"submission score records do not match the worker manifest in {source}")

    records.sort(key=lambda item: str(item["token"]))
    output = Path(output_dir)
    (output / "records").mkdir(parents=True, exist_ok=True)
    merged_config = {
        **signatures[0],
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "rank": None,
        "world_size": declared_world_size,
        "merged": True,
    }
    resolved_fingerprint = config_fingerprint(merged_config)
    for record in records:
        merged_record = {**record, "config_fingerprint": resolved_fingerprint}
        write_json(record_path(output / "records", str(record["token"])), merged_record)

    report = aggregate_records(records, num_failed=len(failures))
    write_family_csv(output, records)
    write_json(output / "aggregate.json", report)
    OmegaConf.save(OmegaConf.create(report), output / "aggregate.yaml")
    with (output / "failures.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["token", "error"])
        writer.writerows(failures)
    write_json(
        output / "run.json",
        {
            **merged_config,
            "status": "failed" if failures else "completed",
            "config_fingerprint": resolved_fingerprint,
            "rank_rows": len(records) + len(failures),
            "num_completed": len(records),
            "num_failed": len(failures),
        },
    )
    return report


def _read_run(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {directory / 'run.json'}") from error
    if (
        not isinstance(value, dict)
        or value.get("format") != RUN_FORMAT
        or value.get("version") != RUN_VERSION
        or value.get("status") not in {"completed", "failed"}
    ):
        raise ValueError(f"not a completed submission-score run directory: {directory}")
    return value


def _manifest_tokens(directory: Path, run: Mapping[str, Any]) -> set[str] | None:
    manifest_name = run.get("manifest")
    if not manifest_name:
        return None
    path = directory / str(manifest_name)
    if not path.is_file():
        raise ValueError(f"submission-score run is missing its worker manifest: {path}")
    manifest = WorkerManifest.read(path)
    if manifest.run_id != str(run.get("run_id")):
        raise ValueError(f"worker manifest belongs to a different run: {path}")
    if manifest.rank != int(run.get("rank", 0)):
        raise ValueError(f"worker manifest rank does not match run.json: {path}")
    return set(manifest.task_ids)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e merge-score-submission")
    parser.add_argument("--input-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    report = merge_submission_scores(
        args.input_dir,
        args.output_dir,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("run", {}).get("num_failed", 0.0) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "merge_submission_scores"]
