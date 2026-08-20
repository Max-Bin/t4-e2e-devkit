"""Dependency-light metric summary callback."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

from .abstract_main_callback import AbstractMainCallback


class MetricSummaryCallback(AbstractMainCallback):
    """Summarize portable metric JSON files into one local report.

    The callback intentionally reads JSON/JSONL only. Parquet and database
    storage are optional concerns for an outer application, not runtime
    requirements of the T4 devkit.
    """

    def __init__(self, metric_dir: str | Path, output_path: str | Path) -> None:
        self.metric_dir = Path(metric_dir)
        self.output_path = Path(output_path)
        self.summary: dict[str, Any] = {}

    def on_run_simulation_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        files = (
            sorted(
                path
                for path in self.metric_dir.rglob("*")
                if path.is_file() and path.suffix == ".json"
            )
            if self.metric_dir.is_dir()
            else []
        )
        records = [_read(path) for path in files]
        records = [record for record in records if record is not None]
        self.summary = {
            "format": "t4.metric-summary.v1",
            "num_files": len(files),
            "num_records": len(records),
            "metric_names": dict(Counter(_metric_name(record) for record in records)),
            "files": [str(path.relative_to(self.metric_dir)) for path in files],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _read(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _metric_name(record: Mapping[str, Any]) -> str:
    for key in ("metric_name", "name", "metric_category"):
        if key in record:
            return str(record[key])
    return "unknown"


__all__ = ["MetricSummaryCallback"]
