"""Typed, serializable experiment configuration for T4 runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class DatasetConfig:
    data_list: str
    history_frames: int = 31
    future_frames: int = 80
    frame_interval: int = 5
    reader: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.data_list):
            raise ValueError("dataset.data_list must not be empty")
        if self.history_frames < 1 or self.future_frames < 0 or self.frame_interval < 1:
            raise ValueError("dataset frame settings are invalid")


@dataclass(frozen=True)
class EvaluationConfig:
    families: tuple[str, ...] = ("open_loop", "pdm")
    backend: str = "auto"
    device: Optional[str] = None
    checkpoint: Optional[str] = None
    submission_dir: Optional[str] = None
    max_retries: int = 0

    def __post_init__(self) -> None:
        allowed = {"open_loop", "pdm"}
        families = tuple(dict.fromkeys(str(item) for item in self.families))
        if not families or not set(families).issubset(allowed):
            raise ValueError(f"evaluation.families must be a non-empty subset of {sorted(allowed)}")
        if self.backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("evaluation.backend must be auto, cpu or gpu")
        if self.max_retries < 0:
            raise ValueError("evaluation.max_retries must be non-negative")
        object.__setattr__(self, "families", families)


@dataclass(frozen=True)
class SimulationConfig:
    enabled: bool = False
    num_steps: int = 200
    replan_interval: int = 1
    max_speed_mps: float = 20.0
    goal_radius_m: float = 2.0
    ttc_horizon_s: Optional[float] = 1.0
    traffic_policy: str = "replay"
    stop_on_collision: bool = False
    stop_on_goal: bool = False

    def __post_init__(self) -> None:
        if self.num_steps < 1 or self.replan_interval < 1:
            raise ValueError("simulation steps and replan_interval must be positive")
        if self.max_speed_mps <= 0.0 or self.goal_radius_m <= 0.0:
            raise ValueError("simulation speed and goal radius must be positive")
        if self.ttc_horizon_s is not None and self.ttc_horizon_s <= 0.0:
            raise ValueError("simulation.ttc_horizon_s must be positive or null")
        if self.traffic_policy not in {"replay", "constant_velocity", "idm"}:
            raise ValueError("simulation.traffic_policy must be replay, constant_velocity or idm")


@dataclass(frozen=True)
class WorkerConfig:
    world_size: int = 1
    workers: int = 1
    backend: str = "serial"
    launcher_backend: str = "sequential"
    timeout_s: Optional[float] = None
    max_retries: int = 0
    resume: bool = True

    def __post_init__(self) -> None:
        if self.world_size < 1 or self.workers < 1:
            raise ValueError("worker world_size and workers must be positive")
        if self.backend not in {"serial", "thread", "process", "ray"}:
            raise ValueError("worker.backend must be serial, thread, process or ray")
        if self.launcher_backend not in {"sequential", "process"}:
            raise ValueError("worker.launcher_backend must be sequential or process")
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise ValueError("worker.timeout_s must be positive when provided")
        if self.max_retries < 0:
            raise ValueError("worker.max_retries must be non-negative")


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "results/run"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.directory):
            raise ValueError("output.directory must not be empty")


@dataclass(frozen=True)
class ExperimentConfig:
    """One resolved run configuration shared by CLI and library callers."""

    agent: str
    dataset: DatasetConfig
    mode: str = "evaluate"
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    agent_params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode != "score_submission" and not str(self.agent):
            raise ValueError("agent must not be empty")
        if self.mode not in {"evaluate", "closed_loop", "submit", "score_submission"}:
            raise ValueError("mode must be evaluate, closed_loop, submit or score_submission")
        object.__setattr__(self, "agent", str(self.agent))
        object.__setattr__(self, "agent_params", dict(self.agent_params))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "mode": self.mode,
            "agent_params": _jsonable(self.agent_params),
            "dataset": _dataclass_dict(self.dataset),
            "evaluation": _dataclass_dict(self.evaluation),
            "simulation": _dataclass_dict(self.simulation),
            "workers": _dataclass_dict(self.workers),
            "output": _dataclass_dict(self.output),
            "metadata": _jsonable(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        dataset = _build_dataclass(DatasetConfig, value.get("dataset", {}))
        evaluation = _build_dataclass(EvaluationConfig, value.get("evaluation", {}))
        simulation = _build_dataclass(SimulationConfig, value.get("simulation", {}))
        workers = _build_dataclass(WorkerConfig, value.get("workers", {}))
        output = _build_dataclass(OutputConfig, value.get("output", {}))
        return cls(
            agent=str(value.get("agent", "")),
            mode=str(value.get("mode", "evaluate")),
            agent_params=value.get("agent_params", {}),
            dataset=dataset,
            evaluation=evaluation,
            simulation=simulation,
            workers=workers,
            output=output,
            metadata=value.get("metadata", {}),
        )


def load_experiment_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Load YAML through OmegaConf and apply ``key=value`` overrides."""

    try:
        from omegaconf import OmegaConf
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("OmegaConf is required to load experiment configs") from error
    config = OmegaConf.load(str(path))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"experiment config must resolve to a mapping: {path}")
    return ExperimentConfig.from_mapping(value)


def _build_dataclass(cls, value: Any):
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    values = dict(value)
    if cls is EvaluationConfig and "families" in values:
        values["families"] = tuple(values["families"])
    return cls(**values)


def _dataclass_dict(value: Any) -> dict[str, Any]:
    from dataclasses import fields

    return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"configuration value is not JSON serializable: {type(value).__name__}")


__all__ = [
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "OutputConfig",
    "SimulationConfig",
    "WorkerConfig",
    "load_experiment_config",
]
