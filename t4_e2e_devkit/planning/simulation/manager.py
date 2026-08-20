"""Lifecycle orchestration for local T4 closed-loop simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from t4_e2e_devkit.planning.simulation.closed_loop import (
    T4ClosedLoopConfig,
    T4ClosedLoopResult,
    T4ClosedLoopRunner,
)
from t4_e2e_devkit.planning.simulation.interfaces import (
    EgoController,
    ObservationProvider,
    SimulationCallback,
    TrafficPolicy,
)


@dataclass(frozen=True)
class SimulationRequest:
    """One deterministic source-frame rollout request."""

    start_frame: int
    num_steps: int
    token: Optional[str] = None

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")


class T4SimulationManager:
    """Run one or more rollouts with a shared lifecycle and callback set.

    The manager is deliberately local and deterministic. It provides the
    simulation-manager boundary needed by evaluation code without introducing
    a scheduler, database, tracking service or sensor renderer.
    """

    def __init__(
        self,
        runner: T4ClosedLoopRunner,
        *,
        callbacks: Optional[Iterable[SimulationCallback]] = None,
    ) -> None:
        self.runner = runner
        self.callbacks = tuple(callbacks or ())

    @classmethod
    def from_scene_dir(
        cls,
        agent,
        scene_dir: str | Path,
        root: str | Path,
        *,
        config: Optional[T4ClosedLoopConfig] = None,
        reader_config: Optional[dict] = None,
        controller: Optional[EgoController] = None,
        observation_provider: Optional[ObservationProvider] = None,
        traffic_policy: Optional[TrafficPolicy] = None,
        callbacks: Optional[Iterable[SimulationCallback]] = None,
    ) -> "T4SimulationManager":
        """Create a manager backed by one T4 scene reader."""

        runner = T4ClosedLoopRunner.from_scene_dir(
            agent,
            scene_dir,
            root,
            config=config,
            reader_config=reader_config,
            controller=controller,
            observation_provider=observation_provider,
            traffic_policy=traffic_policy,
        )
        return cls(runner, callbacks=callbacks)

    def run(self, request: SimulationRequest) -> T4ClosedLoopResult:
        """Execute one request and return its realized rollout."""

        if not isinstance(request, SimulationRequest):
            raise TypeError(
                f"T4SimulationManager.run expects SimulationRequest, got {type(request).__name__}"
            )
        return self.runner.run(
            request.start_frame,
            request.num_steps,
            callbacks=list(self.callbacks),
        )

    def run_many(self, requests: Iterable[SimulationRequest]) -> List[T4ClosedLoopResult]:
        """Run requests in input order, resetting the runner per request."""

        return [self.run(request) for request in requests]

    def close(self) -> None:
        """Release the reader owned by :meth:`from_scene_dir`."""

        self.runner.close()

    def __enter__(self) -> "T4SimulationManager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = ["SimulationRequest", "T4SimulationManager"]
