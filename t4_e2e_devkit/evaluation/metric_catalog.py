"""Explicit metric catalog for configuration-driven simulation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from t4_e2e_devkit.evaluation.metric_api import AbstractMetricBuilder


@dataclass(frozen=True)
class MetricSpec:
    """Metadata and factory for one metric builder."""

    name: str
    family: str
    category: str
    factory: Callable[[], AbstractMetricBuilder]


class MetricCatalog:
    """Strict registry that builds metrics by name or independent family."""

    def __init__(self, specs: Sequence[MetricSpec] = ()) -> None:
        self._specs: dict[str, MetricSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: MetricSpec, *, overwrite: bool = False) -> None:
        if not spec.name or not spec.family or not spec.category:
            raise ValueError("metric spec name, family and category must not be empty")
        if spec.name in self._specs and not overwrite:
            raise ValueError(f"metric is already registered: {spec.name}")
        self._specs[spec.name] = spec

    @property
    def specs(self) -> tuple[MetricSpec, ...]:
        return tuple(self._specs.values())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({spec.family for spec in self._specs.values()}))

    def build(
        self,
        *,
        names: Optional[Sequence[str]] = None,
        families: Optional[Sequence[str]] = None,
    ) -> tuple[AbstractMetricBuilder, ...]:
        wanted_names = None if names is None else {str(value) for value in names}
        wanted_families = None if families is None else {str(value) for value in families}
        unknown_names = sorted((wanted_names or set()) - set(self._specs))
        unknown_families = sorted((wanted_families or set()) - set(self.families()))
        if unknown_names or unknown_families:
            details = []
            if unknown_names:
                details.append(f"unknown metric names: {unknown_names}")
            if unknown_families:
                details.append(f"unknown metric families: {unknown_families}")
            raise ValueError("; ".join(details))
        selected = tuple(
            spec
            for spec in self.specs
            if (wanted_names is None or spec.name in wanted_names)
            and (wanted_families is None or spec.family in wanted_families)
        )
        if not selected:
            raise ValueError(
                f"no metrics selected; names={sorted(wanted_names or ())}, "
                f"families={sorted(wanted_families or ())}"
            )
        return tuple(spec.factory() for spec in selected)

    @classmethod
    def t4_default(cls) -> "MetricCatalog":
        from t4_e2e_devkit.evaluation.metric_builders import (
            ClosedLoopMetricBuilder,
            CollisionMetricBuilder,
            ComfortMetricBuilder,
            DrivableAreaMetricBuilder,
            OpenLoopMetricBuilder,
            PDMMetricBuilder,
            ProgressMetricBuilder,
            StopLineViolationMetricBuilder,
            TrafficLightMetricBuilder,
            TTCMetricBuilder,
        )

        return cls(
            (
                MetricSpec("open_loop", "tracking", "trajectory", OpenLoopMetricBuilder),
                MetricSpec("closed_loop", "simulation", "rollout", ClosedLoopMetricBuilder),
                MetricSpec("pdm", "pdm", "planning", PDMMetricBuilder),
                MetricSpec("comfort", "comfort", "comfort", ComfortMetricBuilder),
                MetricSpec("progress", "progress", "progress", ProgressMetricBuilder),
                MetricSpec("collision", "safety", "safety", CollisionMetricBuilder),
                MetricSpec("drivable_area", "safety", "safety", DrivableAreaMetricBuilder),
                MetricSpec("ttc", "safety", "safety", TTCMetricBuilder),
                MetricSpec("traffic_light", "safety", "safety", TrafficLightMetricBuilder),
                MetricSpec("stop_line", "safety", "safety", StopLineViolationMetricBuilder),
            )
        )


__all__ = ["MetricCatalog", "MetricSpec"]
