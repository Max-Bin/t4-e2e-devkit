"""Agent interface, builders and registry."""

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent
from t4_e2e_devkit.agents.builders import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
    CameraFeatureBuilder,
    EgoStatusFeatureBuilder,
    FeatureBuilderRegistry,
    LidarFeatureBuilder,
    MapFeatureBuilder,
    OracleTargetBuilder,
    TargetBuilderRegistry,
    TrajectoryTargetBuilder,
)
from t4_e2e_devkit.agents.registry import (
    available_agents,
    build_agent,
    register_agent,
)

__all__ = [
    "AbstractT4Agent",
    "AbstractFeatureBuilder",
    "AbstractTargetBuilder",
    "CameraFeatureBuilder",
    "EgoStatusFeatureBuilder",
    "FeatureBuilderRegistry",
    "LidarFeatureBuilder",
    "MapFeatureBuilder",
    "OracleTargetBuilder",
    "TrajectoryTargetBuilder",
    "TargetBuilderRegistry",
    "available_agents",
    "build_agent",
    "register_agent",
]
