"""Small, typed configuration and component-registry boundary."""

from .registry import ComponentRegistry, resolve_component
from .schema import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    OutputConfig,
    SimulationConfig,
    WorkerConfig,
    load_experiment_config,
)

__all__ = [
    "ComponentRegistry",
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "OutputConfig",
    "SimulationConfig",
    "WorkerConfig",
    "load_experiment_config",
    "resolve_component",
]
