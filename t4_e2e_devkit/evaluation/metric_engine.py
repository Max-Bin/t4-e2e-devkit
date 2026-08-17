"""Registry, execution and aggregation for independent metric families."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from t4_e2e_devkit.common.dataclasses import T4Scene, Trajectory
from t4_e2e_devkit.evaluation.metric_cache import MetricCache
from t4_e2e_devkit.planning.simulation.closed_loop import T4ClosedLoopResult


@dataclass(frozen=True)
class MetricContext:
    """Inputs available to one metric computation."""

    token: str
    prediction: Optional[Trajectory] = None
    ground_truth: Optional[Trajectory | T4Scene] = None
    scene: Optional[T4Scene] = None
    closed_loop: Optional[T4ClosedLoopResult] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        """Stable signature for caller-provided metric inputs/configuration."""
        payload = json.dumps(dict(self.metadata), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


MetricComputer = Callable[[MetricContext], Mapping[str, float] | float]
MetricGroup = Callable[["MetricRecord"], str]


@dataclass(frozen=True)
class MetricDefinition:
    """One registered metric family and its computation function."""

    name: str
    compute: MetricComputer
    family: str = "default"


@dataclass(frozen=True)
class MetricRecord:
    """One metric output for one scenario token."""

    name: str
    family: str
    token: str
    values: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "token": self.token,
            "values": {str(key): float(value) for key, value in self.values.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetricRecord":
        values = value.get("values")
        if not isinstance(values, Mapping):
            raise ValueError("metric record has no values mapping")
        return cls(
            name=str(value["name"]),
            family=str(value["family"]),
            token=str(value["token"]),
            values={str(key): float(item) for key, item in values.items()},
            metadata=(value.get("metadata", {}) if isinstance(value.get("metadata", {}), Mapping) else {}),
        )


@dataclass(frozen=True)
class MetricReport:
    """Metric records plus independent family/group aggregates."""

    records: tuple[MetricRecord, ...]

    def aggregate(self, group_by: Optional[MetricGroup] = None) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[MetricRecord]] = {}
        for record in self.records:
            key = group_by(record) if group_by is not None else record.family
            grouped.setdefault(str(key), []).append(record)
        result: dict[str, dict[str, float]] = {}
        for group, records in grouped.items():
            names = sorted({name for record in records for name in record.values})
            values: dict[str, float] = {"num_records": float(len(records))}
            for name in names:
                available = [record.values[name] for record in records if name in record.values]
                if available:
                    values[name] = sum(available) / len(available)
            result[group] = values
        return result

    def as_dict(self, group_by: Optional[MetricGroup] = None) -> dict[str, Any]:
        return {
            "records": [record.as_dict() for record in self.records],
            "aggregates": self.aggregate(group_by=group_by),
        }


class MetricEngine:
    """Execute registered metrics without coupling their families together."""

    def __init__(self, definitions: Optional[Sequence[MetricDefinition]] = None) -> None:
        self._definitions: dict[str, MetricDefinition] = {}
        for definition in definitions or ():
            self.register(definition)

    @property
    def definitions(self) -> tuple[MetricDefinition, ...]:
        return tuple(self._definitions.values())

    def register(
        self,
        definition: MetricDefinition | str,
        compute: Optional[MetricComputer] = None,
        *,
        family: str = "default",
    ) -> None:
        if isinstance(definition, str):
            if compute is None:
                raise ValueError("compute is required when registering a named metric")
            definition = MetricDefinition(definition, compute, family)
        if not definition.name:
            raise ValueError("metric name must not be empty")
        if definition.name in self._definitions:
            raise ValueError(f"metric is already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def evaluate(
        self,
        context: MetricContext,
        *,
        cache: Optional[MetricCache] = None,
        metric_names: Optional[Sequence[str]] = None,
        families: Optional[Sequence[str]] = None,
    ) -> MetricReport:
        records: list[MetricRecord] = []
        wanted_names = None if metric_names is None else {str(name) for name in metric_names}
        wanted_families = None if families is None else {str(name) for name in families}
        definitions = tuple(
            definition
            for definition in self.definitions
            if (wanted_names is None or definition.name in wanted_names)
            and (wanted_families is None or definition.family in wanted_families)
        )
        if not definitions:
            raise ValueError("no registered metrics match metric_names/families")
        for definition in definitions:
            cache_key = None
            if cache is not None:
                cache_key = cache.key(context.token, definition.name, context.signature)
                cached = cache.load(cache_key)
                if cached is not None:
                    try:
                        record = MetricRecord.from_dict(cached)
                        if (
                            record.name == definition.name
                            and record.family == definition.family
                            and record.token == context.token
                        ):
                            records.append(record)
                            continue
                    except (KeyError, TypeError, ValueError):
                        pass
            output = definition.compute(context)
            values = {definition.name: float(output)} if isinstance(output, Real) else {
                str(key): float(value) for key, value in output.items()
            }
            record = MetricRecord(
                name=definition.name,
                family=definition.family,
                token=context.token,
                values=values,
                metadata=dict(context.metadata),
            )
            records.append(record)
            if cache is not None and cache_key is not None:
                cache.save(cache_key, record.as_dict())
        return MetricReport(tuple(records))

    def evaluate_many(
        self,
        contexts: Iterable[MetricContext],
        *,
        cache: Optional[MetricCache] = None,
        metric_names: Optional[Sequence[str]] = None,
        families: Optional[Sequence[str]] = None,
    ) -> MetricReport:
        records: list[MetricRecord] = []
        for context in contexts:
            records.extend(
                self.evaluate(
                    context,
                    cache=cache,
                    metric_names=metric_names,
                    families=families,
                ).records
            )
        return MetricReport(tuple(records))

    @classmethod
    def t4_default(cls) -> "MetricEngine":
        """Create an engine with the dependency-light T4 metric families."""
        from t4_e2e_devkit.evaluation.closed_loop import compute_closed_loop_metrics
        from t4_e2e_devkit.evaluation.open_loop import compute_open_loop_metrics

        engine = cls()

        def open_loop(context: MetricContext) -> Mapping[str, float]:
            if context.prediction is None or context.ground_truth is None:
                raise ValueError("open_loop requires prediction and ground_truth")
            return compute_open_loop_metrics(
                context.prediction,
                context.ground_truth,
                token=context.token,
            ).values

        def closed_loop(context: MetricContext) -> Mapping[str, float]:
            if context.closed_loop is None:
                raise ValueError("closed_loop requires a closed_loop result")
            return compute_closed_loop_metrics(
                context.closed_loop,
                token=context.token,
            ).values

        engine.register("open_loop", open_loop, family="open_loop")
        engine.register("closed_loop", closed_loop, family="closed_loop")
        return engine


__all__ = [
    "MetricContext",
    "MetricDefinition",
    "MetricEngine",
    "MetricRecord",
    "MetricReport",
]
