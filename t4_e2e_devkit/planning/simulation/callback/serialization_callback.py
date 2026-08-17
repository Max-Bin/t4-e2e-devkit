"""Portable simulation-log callback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from t4_e2e_devkit.planning.simulation.simulation_log import SimulationLog

from .abstract_callback import AbstractCallback


class SimulationLogCallback(AbstractCallback):
    """Write one compressed, dependency-free log after a successful run."""

    def __init__(self, output_dir: str | Path, *, suffix: str = ".json.xz") -> None:
        self.output_dir = Path(output_dir)
        self.suffix = str(suffix)
        self.paths: list[Path] = []

    def on_simulation_end(self, setup: Any, planner: Any, history: Any) -> None:
        token = str(getattr(getattr(setup, "scenario", None), "token", "simulation"))
        safe_token = "".join(character if character.isalnum() or character in "-_" else "_" for character in token)
        path = self.output_dir / f"{safe_token}{self.suffix}"
        self.paths.append(SimulationLog(path, setup.scenario, planner, history).save_to_file())


__all__ = ["SimulationLogCallback"]
