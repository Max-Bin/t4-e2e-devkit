"""Registry, execution and aggregation for independent metric families."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from t4_e2e_devkit.common.dataclasses import T4Scene, Trajectory
from t4_e2e_devkit.evaluation.metric_cache import MetricCache
from t4_e2e_devkit.planning.simulation.closed_loop import T4ClosedLoopResult

if TYPE_CHECKING:
    from t4_e2e_devkit.evaluation.pdm_score import T4PDMScorer
    from t4_e2e_devkit.evaluation.tier4_metrics import RewardConfig


@dataclass(frozen=True)
class MetricContext:
    """Inputs available to one metric computation."""

    token: str
    prediction: Optional[Trajectory] = None
    ground_truth: Optional[Trajectory | T4Scene] = None
    scene: Optional[T4Scene] = None
    closed_loop: Optional[T4ClosedLoopResult] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    pdm_scorer: Optional["T4PDMScorer"] = None
    tier4_config: Optional["RewardConfig"] = None

    @property
    def signature(self) -> str:
        """Stable signature for metric inputs and caller configuration.

        The first version of the engine keyed its cache only by ``metadata``.
        That made two different predictions for the same token collide when a
        caller did not provide a manual version string.  Include the numeric
        inputs and scorer settings here so cache reuse remains correct while
        keeping the cache key independent of Python object identities.
        """
        payload = {
            "metadata": _jsonable(dict(self.metadata)),
            "prediction": _trajectory_signature(self.prediction),
            "ground_truth": _ground_truth_signature(self.ground_truth),
            "scene": _scene_signature(self.scene),
            "closed_loop": _closed_loop_signature(self.closed_loop),
            "pdm_scorer": _scorer_signature(self.pdm_scorer),
            "tier4_config": _jsonable(self.tier4_config),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


MetricComputer = Callable[[MetricContext], Mapping[str, float] | float]
MetricGroup = Callable[["MetricRecord"], str]


@dataclass(frozen=True)
class MetricDefinition:
    """One registered metric family and its computation function."""

    name: str
    compute: MetricComputer
    family: str = "default"
    supports: Optional[Callable[[MetricContext], bool]] = None


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
        explicit_selection = wanted_names is not None or wanted_families is not None
        definitions = tuple(
            definition
            for definition in self.definitions
            if (wanted_names is None or definition.name in wanted_names)
            and (wanted_families is None or definition.family in wanted_families)
            and (
                explicit_selection
                or definition.supports is None
                or definition.supports(context)
            )
        )
        if not definitions:
            raise ValueError("no registered metrics match metric_names/families")
        context_signature = context.signature if cache is not None else None
        for definition in definitions:
            cache_key = None
            if cache is not None:
                assert context_signature is not None
                cache_key = cache.key(context.token, definition.name, context_signature)
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

    def evaluate_history(
        self,
        history: Any,
        builders: Sequence[Any],
        *,
        scenario_token: Optional[str] = None,
    ) -> tuple[Any, ...]:
        """Run NuPlan-shaped metric builders on one simulation history.

        The family engine above emits compact scalar records.  This adapter
        keeps richer time-series/statistic results separate so simulation
        consumers can use the same engine without flattening their reports.
        """

        from t4_e2e_devkit.evaluation.metric_api import MetricBuilderRegistry

        registry = MetricBuilderRegistry(builders)
        return registry.compute(history, scenario_token=scenario_token)

    @staticmethod
    def aggregate_history_metrics(
        results: Iterable[Any],
        *,
        weights: Optional[Mapping[str, float]] = None,
        include_failures: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Aggregate rich builder results by metric family and statistic."""

        from t4_e2e_devkit.evaluation.metric_api import MetricAggregator

        return MetricAggregator().aggregate(
            results,
            weights=weights,
            include_failures=include_failures,
        )

    @classmethod
    def t4_default(cls) -> "MetricEngine":
        """Create an engine with all built-in, independent metric families.

        PDM and T4 metrics are registered as adapters rather than folded into
        open-loop or closed-loop records.  Callers can select one family with
        ``families=...`` and can inject a configured scorer through
        :class:`MetricContext` when the default CPU scorer is not appropriate.
        """
        return cls.with_t4_metrics()

    @classmethod
    def with_t4_metrics(
        cls,
        *,
        pdm_scorer: Optional["T4PDMScorer"] = None,
        tier4_config: Optional["RewardConfig"] = None,
        include_pdm: bool = True,
        include_tier4: bool = True,
    ) -> "MetricEngine":
        """Build the standard engine with explicit family registration.

        ``pdm_scorer`` is optional to keep construction cheap.  When the PDM
        family is evaluated without a scorer, one CPU scorer is created lazily.
        This means an open-loop-only run never initializes the PDM stack.
        """
        from t4_e2e_devkit.evaluation.closed_loop import compute_closed_loop_metrics
        from t4_e2e_devkit.evaluation.open_loop import compute_open_loop_metrics

        engine = cls()
        scorer = pdm_scorer

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

        def pdm(context: MetricContext) -> Mapping[str, float]:
            nonlocal scorer
            if context.prediction is None:
                raise ValueError("pdm requires prediction")
            scene = context.scene
            if scene is None and isinstance(context.ground_truth, T4Scene):
                scene = context.ground_truth
            if scene is None:
                raise ValueError("pdm requires scene or a T4Scene ground_truth")
            active_scorer = context.pdm_scorer
            if active_scorer is None:
                if scorer is None:
                    from t4_e2e_devkit.evaluation.pdm_score import T4PDMScorer

                    scorer = T4PDMScorer(backend="cpu")
                active_scorer = scorer
            result = active_scorer.score(context.prediction, scene)
            return {**result.components, "score": float(result.score)}

        def tier4(context: MetricContext) -> Mapping[str, float]:
            if context.prediction is None:
                raise ValueError("tier4 requires prediction")
            scene = context.scene
            if scene is None and isinstance(context.ground_truth, T4Scene):
                scene = context.ground_truth
            if scene is None:
                raise ValueError("tier4 requires scene or a T4Scene ground_truth")
            from t4_e2e_devkit.evaluation.tier4_metrics import compute_tier4_metrics

            return compute_tier4_metrics(
                context.prediction,
                scene,
                config=context.tier4_config or tier4_config,
            )

        engine.register(
            MetricDefinition(
                "open_loop",
                open_loop,
                family="open_loop",
                supports=lambda context: context.prediction is not None
                and context.ground_truth is not None,
            )
        )
        engine.register(
            MetricDefinition(
                "closed_loop",
                closed_loop,
                family="closed_loop",
                supports=lambda context: context.closed_loop is not None,
            )
        )
        if include_pdm:
            engine.register(
                MetricDefinition(
                    "pdm",
                    pdm,
                    family="pdm",
                    supports=lambda context: context.prediction is not None
                    and (
                        context.scene is not None
                        or isinstance(context.ground_truth, T4Scene)
                    ),
                )
            )
        if include_tier4:
            engine.register(
                MetricDefinition(
                    "tier4",
                    tier4,
                    family="tier4",
                    supports=lambda context: context.prediction is not None
                    and (
                        context.scene is not None
                        or isinstance(context.ground_truth, T4Scene)
                    ),
                )
            )
        return engine


def _array_signature(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _jsonable(value: Any) -> Any:
    """Convert common config/dataclass values to deterministic JSON data."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return _array_signature(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(
            {name: getattr(value, name) for name in value.__dataclass_fields__}
        )
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _trajectory_signature(trajectory: Optional[Trajectory]) -> Any:
    if trajectory is None:
        return None
    sampling = trajectory.trajectory_sampling
    return {
        "poses": _array_signature(trajectory.poses),
        "num_poses": int(sampling.num_poses),
        "interval_length": float(sampling.interval_length),
    }


def _ground_truth_signature(value: Optional[Trajectory | T4Scene]) -> Any:
    if value is None:
        return None
    if isinstance(value, Trajectory):
        return _trajectory_signature(value)
    return _scene_signature(value)


def _scene_signature(scene: Optional[T4Scene]) -> Any:
    if scene is None:
        return None
    frame = scene.current_frame
    map_signature = None
    if frame.map_tensors is not None:
        map_signature = {
            name: _array_signature(value)
            for name, value in frame.map_tensors.as_dict().items()
        }
    annotations = frame.annotations
    annotation_signature = None
    if annotations is not None:
        annotation_signature = {
            "boxes": _array_signature(annotations.boxes),
            "labels": _array_signature(annotations.labels),
            "track_tokens": _jsonable(annotations.track_tokens),
        }
    return {
        "token": scene.scene_metadata.token,
        "future_ego_poses": _array_signature(scene.future_ego_poses),
        "future_annotations": None
        if scene.future_annotations is None
        else [
            {
                "boxes": _array_signature(item.boxes),
                "labels": _array_signature(item.labels),
            }
            for item in scene.future_annotations
        ],
        "map": map_signature,
        "annotations": annotation_signature,
        "goal_pose": _array_signature(scene.goal_pose),
        "pdm_progress": scene.pdm_progress,
    }


def _closed_loop_signature(result: Optional[T4ClosedLoopResult]) -> Any:
    if result is None:
        return None
    return {
        "source_frames": _array_signature(result.source_frames),
        "poses": _array_signature(result.realized_poses_world),
        "dt_s": float(result.dt_s),
        "goal": _array_signature(result.goal_pose_world),
        "collision_steps": _jsonable(result.collision_steps),
        "timeout": result.timeout,
        "termination_reason": result.termination_reason,
    }


def _scorer_signature(scorer: Any) -> Any:
    if scorer is None:
        return None
    return {
        "type": f"{type(scorer).__module__}.{type(scorer).__qualname__}",
        "backend": getattr(scorer, "backend", None),
        "device": str(getattr(scorer, "device", "")),
        "config": _jsonable(getattr(scorer, "config", None)),
    }


__all__ = [
    "MetricContext",
    "MetricDefinition",
    "MetricEngine",
    "MetricRecord",
    "MetricReport",
]
