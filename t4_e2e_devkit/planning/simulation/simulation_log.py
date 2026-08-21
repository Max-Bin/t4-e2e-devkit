"""Portable simulation log storage without a database dependency."""

from __future__ import annotations

import json
import lzma
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from t4_e2e_devkit.common.artifact_io import portable_value


@dataclass
class SimulationLog:
    """Scenario, planner and completed history for one simulation."""

    file_path: Path
    scenario: Any
    planner: Any
    simulation_history: Any

    @property
    def history(self) -> Any:
        return self.simulation_history

    def save_to_file(self) -> Path:
        """Save as ``.json.xz`` or ``.pkl.xz`` based on the path suffix."""

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        log_type = self.simulation_log_type(self.file_path)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.file_path.name}.", dir=str(self.file_path.parent)
        )
        os.close(descriptor)
        try:
            if log_type == "json":
                payload = {
                    "format": "t4.simulation-log.v1",
                    "scenario": portable_value(self.scenario),
                    "planner": portable_value(self.planner),
                    "history": portable_value(self.simulation_history),
                }
                with lzma.open(temporary, "wt", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
            else:
                with lzma.open(temporary, "wb", preset=0) as stream:
                    pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, self.file_path)
        finally:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        return self.file_path

    @staticmethod
    def simulation_log_type(file_path: str | Path) -> str:
        suffixes = Path(file_path).suffixes
        if len(suffixes) < 2 or suffixes[-1] != ".xz":
            raise ValueError(f"simulation logs must use a compressed suffix: {file_path}")
        if suffixes[-2] in {".json", ".jsonl"}:
            return "json"
        if suffixes[-2] in {".pkl", ".pickle", ".msgpack"}:
            return "pickle"
        raise ValueError(f"cannot infer simulation log type from {file_path}")

    @classmethod
    def load_data(cls, file_path: str | Path) -> Any:
        """Load a pickled log or the portable JSON representation."""

        path = Path(file_path)
        if cls.simulation_log_type(path) == "json":
            with lzma.open(path, "rt", encoding="utf-8") as stream:
                return json.load(stream)
        with lzma.open(path, "rb") as stream:
            return pickle.load(stream)

    @classmethod
    def load(cls, file_path: str | Path) -> "SimulationLog":
        value = cls.load_data(file_path)
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or value.get("format") != "t4.simulation-log.v1":
            raise ValueError(f"unsupported simulation log payload: {file_path}")
        from t4_e2e_devkit.planning.simulation.runtime import SimulationHistory

        history = SimulationHistory.from_dict(value.get("history", {}))
        return cls(Path(file_path), value.get("scenario"), value.get("planner"), history)


__all__ = ["SimulationLog"]
