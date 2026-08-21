"""Merge rank directories produced by :mod:`t4_e2e_devkit.script.evaluate`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from omegaconf import OmegaConf

from t4_e2e_devkit.evaluation.batch import (
    RUN_FORMAT,
    RUN_VERSION,
    aggregate_records,
    config_fingerprint,
    record_path,
    write_family_csv,
    write_json,
)
from t4_e2e_devkit.script.utils import manifest_tokens, read_run

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
    "num_resumed",
    "merged",
    "input_dirs",
}


def merge_evaluation_reports(
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, dict[str, float]]:
    """Merge complete rank directories and recompute all family aggregates."""

    sources = [Path(value).resolve() for value in input_dirs]
    if not sources:
        raise ValueError("at least one input directory is required")
    destination = Path(output_dir).resolve()
    if destination in sources:
        raise ValueError("output directory must differ from every input directory")
    runs = [
        read_run(source, kind="evaluation", run_format=RUN_FORMAT, run_version=RUN_VERSION)
        for source in sources
    ]
    signatures = [
        {key: value for key, value in run.items() if key not in _VOLATILE} for run in runs
    ]
    if any(value != signatures[0] for value in signatures[1:]):
        raise ValueError("evaluation rank configurations do not match")
    declared_world_size = int(runs[0].get("world_size", 1))
    ranks = [int(run.get("rank", 0)) for run in runs]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks: {ranks}")
    if not allow_incomplete and (
        set(ranks) != set(range(declared_world_size)) or len(sources) != declared_world_size
    ):
        raise ValueError(
            f"incomplete rank set; expected {list(range(declared_world_size))}, got {sorted(ranks)}"
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    failures: list[tuple[str, str]] = []
    for source, run in zip(sources, runs, strict=True):
        record_files = sorted((source / "records").glob("record-*.json"))
        expected_tokens = manifest_tokens(source, run, kind="evaluation")
        actual_tokens: set[str] = set()
        for path in record_files:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read evaluation record {path}") from error
            if not isinstance(record, dict) or not record.get("token"):
                raise ValueError(f"invalid evaluation record: {path}")
            token = str(record["token"])
            actual_tokens.add(token)
            if record.get("config_fingerprint") != run.get("config_fingerprint"):
                raise ValueError(f"evaluation record has a stale configuration: {path}")
            if token in seen:
                raise ValueError(f"duplicate evaluation token across ranks: {token}")
            seen.add(token)
            if record.get("status") == "ok":
                records.append(record)
            else:
                failures.append((token, str(record.get("error", "evaluation failed"))))
        if expected_tokens is not None and actual_tokens != expected_tokens:
            raise ValueError(f"evaluation records do not match the worker manifest in {source}")

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
        record = {**record, "config_fingerprint": resolved_fingerprint}
        write_json(record_path(output / "records", str(record["token"])), record)
    report = aggregate_records(records, num_failed=len(failures))
    write_family_csv(output, records)
    write_json(output / "aggregate.json", report)
    OmegaConf.save(OmegaConf.create(report), output / "aggregate.yaml")
    with (output / "failures.csv").open("w", encoding="utf-8") as stream:
        stream.write("token,error\n")
        for token, error in failures:
            stream.write(f"{_csv(token)},{_csv(error)}\n")
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


def _csv(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e merge-evaluation")
    parser.add_argument("--input-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    report = merge_evaluation_reports(
        args.input_dir,
        args.output_dir,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("run", {}).get("num_failed", 0.0) == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
