"""Feature/target preprocessing boundary."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from t4_e2e_devkit.agents.builders import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
    FeatureBuilderRegistry,
    TargetBuilderRegistry,
)
from t4_e2e_devkit.common.dataclasses import T4AgentInput, T4Scene
from t4_e2e_devkit.planning.training.data_augmentation import Augmentor
from t4_e2e_devkit.planning.training.modeling import FeatureMapping


class FeaturePreprocessor:
    """Run feature builders, target builders and optional augmentors."""

    def __init__(
        self,
        feature_builders: Sequence[AbstractFeatureBuilder],
        target_builders: Sequence[AbstractTargetBuilder] = (),
        *,
        augmentor: Optional[Augmentor] = None,
    ) -> None:
        self.feature_builders = FeatureBuilderRegistry(feature_builders)
        self.target_builders = TargetBuilderRegistry(target_builders)
        self.augmentor = augmentor

    def process(self, scene: T4Scene) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        input_view: T4AgentInput = scene.get_agent_input()
        if self.augmentor is not None:
            input_view, scene = self.augmentor(input_view, scene)
        return self.feature_builders.compute(input_view), self.target_builders.compute(scene)

    def process_features(self, agent_input: T4AgentInput) -> dict[str, torch.Tensor]:
        if self.augmentor is not None:
            agent_input, _ = self.augmentor(agent_input, None)
        return self.feature_builders.compute(agent_input)

    def process_targets(self, scene: T4Scene) -> dict[str, torch.Tensor]:
        return self.target_builders.compute(scene)

    @staticmethod
    def collate(features: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> tuple[FeatureMapping, FeatureMapping]:
        return FeatureMapping.collate([FeatureMapping(item) for item in features]), FeatureMapping.collate([FeatureMapping(item) for item in targets])


FeatureBuilder = AbstractFeatureBuilder
TargetBuilder = AbstractTargetBuilder

__all__ = ["FeatureBuilder", "FeaturePreprocessor", "TargetBuilder"]
