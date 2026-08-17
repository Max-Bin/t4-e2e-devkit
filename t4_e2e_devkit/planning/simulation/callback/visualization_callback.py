"""Simulation-frame serialization hook for local visualizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from t4_e2e_devkit.planning.simulation.runtime import _portable_value

from .abstract_callback import AbstractCallback


class VisualizationCallback(AbstractCallback):
    """Serialize selected simulation frames under an ignored results folder."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        renderer: Optional[Callable[[Any, Any, Any], Any]] = None,
        every_n_steps: int = 1,
    ) -> None:
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        self.output_dir = Path(output_dir)
        self.renderer = renderer
        self.every_n_steps = int(every_n_steps)
        self.paths: list[Path] = []

    def on_simulation_start(self, setup: Any) -> None:
        del setup
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, setup: Any, planner: Any, sample: Any) -> None:
        if int(sample.iteration.index) % self.every_n_steps:
            return
        payload = (
            self.renderer(setup, planner, sample)
            if self.renderer is not None
            else _portable_value(sample)
        )
        token = str(getattr(getattr(setup, "scenario", None), "token", "simulation"))
        safe_token = "".join(character if character.isalnum() or character in "-_" else "_" for character in token)
        path = self.output_dir / safe_token / f"{int(sample.iteration.index):06d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_portable_value(payload), sort_keys=True) + "\n", encoding="utf-8")
        self.paths.append(path)


__all__ = ["VisualizationCallback"]
