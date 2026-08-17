"""Generic tracked-object predictor boundary."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Any, Optional, Type

from t4_e2e_devkit.planning.simulation.observation.observation_type import Observation
from t4_e2e_devkit.planning.simulation.predictor.predictor_report import PredictorReport
from t4_e2e_devkit.planning.simulation.runtime import SimulationHistoryBuffer
from t4_e2e_devkit.planning.simulation.simulation_iteration import SimulationIteration


@dataclass(frozen=True)
class PredictorInitialization:
    map_api: Any = None


@dataclass(frozen=True)
class PredictorInput:
    iteration: SimulationIteration
    history: SimulationHistoryBuffer
    traffic_light_data: Optional[list[Any]] = None


class AbstractPredictor(abc.ABC):
    requires_scenario: bool = False

    def __new__(cls, *args: Any, **kwargs: Any):
        instance = super().__new__(cls)
        instance._compute_predictions_runtimes = []
        return instance

    @abc.abstractmethod
    def name(self) -> str:
        """Return a stable predictor name."""

    @abc.abstractmethod
    def initialize(self, initialization: PredictorInitialization) -> None:
        """Initialize for one scenario."""

    @abc.abstractmethod
    def observation_type(self) -> Type[Observation]:
        """Return the observation type consumed by the predictor."""

    @abc.abstractmethod
    def compute_predicted_trajectories(self, current_input: PredictorInput) -> Any:
        """Attach predictions to the current tracked-object observation."""

    def compute_predictions(self, current_input: PredictorInput) -> Any:
        started = time.perf_counter()
        try:
            return self.compute_predicted_trajectories(current_input)
        finally:
            self._compute_predictions_runtimes.append(time.perf_counter() - started)

    def generate_predictor_report(self, clear_stats: bool = True) -> PredictorReport:
        report = PredictorReport(self._compute_predictions_runtimes)
        if clear_stats:
            self._compute_predictions_runtimes.clear()
        return report


__all__ = ["AbstractPredictor", "PredictorInitialization", "PredictorInput"]
