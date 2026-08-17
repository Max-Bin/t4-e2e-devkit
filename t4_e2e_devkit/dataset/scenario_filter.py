"""Scenario filtering and deterministic sampling for T4 scenario lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np


def _as_tuple(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or ()) if str(value))


@dataclass(frozen=True)
class ScenarioFilter:
    """NuPlan-style scenario restrictions over materialized T4 scenarios."""

    scenario_types: tuple[str, ...] = ()
    log_names: tuple[str, ...] = ()
    map_names: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()
    include_events: tuple[str, ...] = ()
    exclude_events: tuple[str, ...] = ()
    max_scenarios: Optional[int] = None
    max_per_type: Optional[int] = None
    shuffle: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "scenario_types",
            "log_names",
            "map_names",
            "tokens",
            "include_events",
            "exclude_events",
        ):
            object.__setattr__(self, field_name, _as_tuple(getattr(self, field_name)))
        for name in ("max_scenarios", "max_per_type"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when provided")

    def matches(self, scenario: Any) -> bool:
        """Return whether a materialized scenario satisfies the restrictions."""

        token = str(getattr(scenario, "token", ""))
        scenario_type = str(getattr(scenario, "scenario_type", ""))
        log_name = str(getattr(scenario, "log_name", ""))
        map_name = str(getattr(scenario, "map_name", ""))
        if self.tokens and token not in self.tokens:
            return False
        if self.scenario_types and scenario_type not in self.scenario_types:
            return False
        if self.log_names and log_name not in self.log_names:
            return False
        if self.map_names and map_name not in self.map_names:
            return False
        tags = getattr(getattr(scenario, "scene", None), "scene_metadata", None)
        scene_tags = getattr(tags, "scene_tags", ())
        events = {str(event) for tag in scene_tags for event in getattr(tag, "events", ())}
        if self.include_events and not set(self.include_events).issubset(events):
            return False
        if self.exclude_events and events.intersection(self.exclude_events):
            return False
        return True


@dataclass(frozen=True)
class ScenarioSampling:
    """Deterministic sampling policy applied after filtering."""

    num_samples: Optional[int] = None
    strategy: str = "ordered"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_samples is not None and self.num_samples < 0:
            raise ValueError("num_samples must be non-negative when provided")
        if self.strategy not in {"ordered", "uniform", "type_balanced"}:
            raise ValueError("strategy must be ordered, uniform or type_balanced")


def filter_scenarios(scenarios: Iterable[Any], config: Optional[ScenarioFilter] = None) -> list[Any]:
    """Filter scenarios and apply deterministic limits."""

    config = config or ScenarioFilter()
    selected = [scenario for scenario in scenarios if config.matches(scenario)]
    if config.shuffle:
        rng = np.random.default_rng(config.seed)
        order = rng.permutation(len(selected))
        selected = [selected[int(index)] for index in order]
    if config.max_per_type is not None:
        counts: dict[str, int] = {}
        limited: list[Any] = []
        for scenario in selected:
            key = str(getattr(scenario, "scenario_type", ""))
            if counts.get(key, 0) >= config.max_per_type:
                continue
            counts[key] = counts.get(key, 0) + 1
            limited.append(scenario)
        selected = limited
    if config.max_scenarios is not None:
        selected = selected[: config.max_scenarios]
    return selected


def sample_scenarios(
    scenarios: Sequence[Any],
    sampling: Optional[ScenarioSampling] = None,
) -> list[Any]:
    """Sample materialized scenarios using a stable random generator."""

    sampling = sampling or ScenarioSampling()
    values = list(scenarios)
    if sampling.num_samples is None or sampling.num_samples >= len(values):
        return values
    if sampling.num_samples == 0:
        return []
    rng = np.random.default_rng(sampling.seed)
    if sampling.strategy == "ordered":
        return values[: sampling.num_samples]
    if sampling.strategy == "uniform":
        indices = np.sort(rng.choice(len(values), sampling.num_samples, replace=False))
        return [values[int(index)] for index in indices]

    groups: dict[str, list[Any]] = {}
    for scenario in values:
        groups.setdefault(str(getattr(scenario, "scenario_type", "")), []).append(scenario)
    keys = sorted(groups)
    selected: list[Any] = []
    while len(selected) < sampling.num_samples and keys:
        next_keys: list[str] = []
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                if len(selected) >= sampling.num_samples:
                    break
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def select_scenarios(
    scenarios: Iterable[Any],
    scenario_filter: Optional[ScenarioFilter] = None,
    sampling: Optional[ScenarioSampling] = None,
) -> list[Any]:
    """Apply filtering followed by sampling in one explicit operation."""

    return sample_scenarios(filter_scenarios(scenarios, scenario_filter), sampling)


def filter_scenarios_for_rank(
    scenarios: Iterable[Any], *, rank: int = 0, world_size: int = 1
) -> list[Any]:
    """Return the stable subset owned by one distributed rank."""

    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    values = list(scenarios)
    return values[int(rank) : len(values) : int(world_size)]


__all__ = [
    "ScenarioFilter",
    "ScenarioSampling",
    "filter_scenarios",
    "sample_scenarios",
    "select_scenarios",
    "filter_scenarios_for_rank",
]
