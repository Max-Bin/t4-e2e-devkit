"""Portable artifacts for rank-partitioned T4 evaluation runs.

The format is intentionally small: one JSON record per scenario, one run
manifest, family-specific CSV files and one aggregate.  It is enough for local
resume and rank merging without a database or an external service.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from t4_e2e_devkit.common.constants import PDM_COMPONENT_ORDER
from t4_e2e_devkit.common.dataclasses import PDMResults
from t4_e2e_devkit.evaluation.open_loop import OpenLoopMetrics
from t4_e2e_devkit.evaluation.report import aggregate_evaluation

RUN_FORMAT = "t4.evaluation.run"
RUN_VERSION = 1
EXECUTION_CONFIG_KEYS = frozenset(
    {"rank", "world_size", "workers", "worker_backend", "max_retries"}
)


def fingerprint(value: Any) -> str:
    """Return a stable configuration fingerprint."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Fingerprint resolved evaluation settings, excluding worker topology."""

    return fingerprint(
        {
            str(key): value
            for key, value in config.items()
            if key not in EXECUTION_CONFIG_KEYS
            and key
            not in {
                "status",
                "config_fingerprint",
                "manifest",
                "rank_rows",
                "merged",
                "input_dirs",
                "source_world_size",
            }
        }
    )


def record_path(directory: str | Path, token: str) -> Path:
    """Return a privacy-safe, collision-resistant path for one token."""

    digest = hashlib.sha256(str(token).encode("utf-8")).hexdigest()[:24]
    return Path(directory) / f"record-{digest}.json"


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Write JSON atomically so an interrupted worker cannot fake completion."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
    return output


def read_record(path: str | Path, *, config_fingerprint: str | None = None) -> dict[str, Any] | None:
    """Read a successful record, returning ``None`` for stale/corrupt data."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("status") != "ok":
        return None
    if config_fingerprint is not None and value.get("config_fingerprint") != config_fingerprint:
        return None
    if not isinstance(value.get("families"), Mapping) or not value.get("token"):
        return None
    return value


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    num_failed: int = 0,
) -> dict[str, dict[str, float]]:
    """Aggregate records while keeping metric families independent."""

    pdm: list[PDMResults] = []
    open_loop: list[OpenLoopMetrics] = []
    tier4: list[Mapping[str, float]] = []
    for record in records:
        families = record.get("families", {})
        if not isinstance(families, Mapping):
            continue
        values = families.get("pdm")
        if isinstance(values, Mapping) and all(name in values for name in PDM_COMPONENT_ORDER):
            score = None
            if values.get("score") is not None:
                candidate = float(values["score"])
                if math.isfinite(candidate):
                    score = candidate
            pdm.append(
                PDMResults.from_components(
                    [float(values[name]) for name in PDM_COMPONENT_ORDER],
                    score=score,
                    token=str(record["token"]),
                )
            )
        values = families.get("open_loop")
        if isinstance(values, Mapping):
            required = {
                "ade_m",
                "fde_m",
                "heading_mae_rad",
                "final_heading_error_rad",
                "miss_rate",
                "horizon_s",
                "num_poses",
            }
            if required.issubset(values):
                open_loop.append(
                    OpenLoopMetrics(
                        **{name: float(values[name]) for name in required},
                        token=str(record["token"]),
                    )
                )
        values = families.get("tier4")
        if isinstance(values, Mapping):
            tier4.append({str(key): float(value) for key, value in values.items()})
    return aggregate_evaluation(
        pdm=pdm,
        open_loop=open_loop,
        tier4=tier4,
        num_failed=int(num_failed),
    )


def write_family_csv(directory: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write one CSV per requested family from portable records."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for record in records:
        token = str(record.get("token", ""))
        families = record.get("families", {})
        if not isinstance(families, Mapping):
            continue
        for family, values in families.items():
            if isinstance(values, Mapping):
                grouped.setdefault(str(family), []).append((token, values))
    for family, rows in grouped.items():
        names = sorted({str(name) for _, values in rows for name in values})
        with (output / f"{family}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["token", *names])
            for token, values in rows:
                writer.writerow([token, *[values.get(name, "") for name in names]])
    for family in {"open_loop", "pdm", "tier4", "closed_loop"} - set(grouped):
        stale = output / f"{family}.csv"
        if stale.is_file():
            stale.unlink()


__all__ = [
    "RUN_FORMAT",
    "RUN_VERSION",
    "EXECUTION_CONFIG_KEYS",
    "aggregate_records",
    "config_fingerprint",
    "fingerprint",
    "read_record",
    "record_path",
    "write_family_csv",
    "write_json",
]
