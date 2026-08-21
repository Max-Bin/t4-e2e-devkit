"""Portable completion marker for resumable local runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from t4_e2e_devkit.common.artifact_io import portable_value

from .abstract_main_callback import AbstractMainCallback


class CompletionCallback(AbstractMainCallback):
    """Write a small completion record after a batch finishes."""

    def __init__(self, output_dir: str | Path, *, run_id: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = None if run_id is None else str(run_id)
        self.path = self.output_dir / "completed.json"

    def on_run_simulation_end(self, report: Any = None, **kwargs: Any) -> None:
        payload: dict[str, Any] = {
            "format": "t4.simulation-completion.v1",
            "run_id": self.run_id,
            "succeeded": _succeeded(report),
        }
        if isinstance(report, Mapping):
            payload["summary"] = portable_value(report)
        payload.update({str(key): portable_value(value) for key, value in kwargs.items()})
        _atomic_json(self.path, payload)


def _succeeded(report: Any) -> bool:
    if report is None:
        return True
    if isinstance(report, Mapping):
        return bool(report.get("succeeded", True))
    return bool(getattr(report, "succeeded", True))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


__all__ = ["CompletionCallback"]
