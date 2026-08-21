"""Run registered metric builders after a simulation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from t4_e2e_devkit.evaluation.metric_api import (
    AbstractMetricBuilder,
    MetricBuilderRegistry,
    MetricResult,
)

from .abstract_callback import AbstractCallback


class MetricCallback(AbstractCallback):
    """Compute metrics at simulation end and retain their results."""

    def __init__(
        self,
        builders: Iterable[AbstractMetricBuilder],
        *,
        scenario_token: Optional[str] = None,
    ) -> None:
        self.registry = MetricBuilderRegistry(tuple(builders))
        self.scenario_token = scenario_token
        self.results: tuple[MetricResult, ...] = ()

    def on_simulation_end(self, setup: Any, planner: Any, history: Any) -> None:
        del planner
        token = (
            self.scenario_token
            or str(getattr(getattr(setup, "scenario", None), "token", ""))
            or None
        )
        self.results = self.registry.compute(history, scenario_token=token)

    def on_simulation_error(self, setup: Any, planner: Any, error: BaseException) -> None:
        del setup, planner, error


__all__ = ["MetricCallback"]
