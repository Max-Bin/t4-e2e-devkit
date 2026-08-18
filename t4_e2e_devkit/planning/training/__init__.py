"""Training glue with lazy imports to keep dataset reads lightweight."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str]] = {
    "T4DataModule": ("t4_e2e_devkit.planning.training.datamodule", "T4DataModule"),
    "T4LightningModule": (
        "t4_e2e_devkit.planning.training.lightning_module",
        "T4LightningModule",
    ),
    "PredictionVizCallback": (
        "t4_e2e_devkit.planning.training.callbacks",
        "PredictionVizCallback",
    ),
    "OfficialDevkitScoreCallback": (
        "t4_e2e_devkit.planning.training.official_score",
        "OfficialDevkitScoreCallback",
    ),
    "TrajectoryVizCallback": (
        "t4_e2e_devkit.planning.training.callbacks",
        "TrajectoryVizCallback",
    ),
    "FeatureCache": ("t4_e2e_devkit.planning.training.feature_cache", "FeatureCache"),
    "AbstractModelFeature": ("t4_e2e_devkit.planning.training.modeling", "AbstractModelFeature"),
    "FeatureMapping": ("t4_e2e_devkit.planning.training.modeling", "FeatureMapping"),
    "TensorFeature": ("t4_e2e_devkit.planning.training.modeling", "TensorFeature"),
    "FeaturePreprocessor": ("t4_e2e_devkit.planning.training.preprocessing", "FeaturePreprocessor"),
    "FeatureBuilder": ("t4_e2e_devkit.planning.training.preprocessing", "FeatureBuilder"),
    "TargetBuilder": ("t4_e2e_devkit.planning.training.preprocessing", "TargetBuilder"),
    "Augmentor": ("t4_e2e_devkit.planning.training.data_augmentation", "Augmentor"),
    "ComposeAugmentor": ("t4_e2e_devkit.planning.training.data_augmentation", "ComposeAugmentor"),
    "RandomSE2Augmentor": ("t4_e2e_devkit.planning.training.data_augmentation", "RandomSE2Augmentor"),
    "AbstractObjective": ("t4_e2e_devkit.planning.training.objectives", "AbstractObjective"),
    "AbstractTrainingMetric": ("t4_e2e_devkit.planning.training.objectives", "AbstractTrainingMetric"),
    "MeanSquaredTrajectoryObjective": ("t4_e2e_devkit.planning.training.objectives", "MeanSquaredTrajectoryObjective"),
    "TrajectoryErrorMetric": ("t4_e2e_devkit.planning.training.objectives", "TrajectoryErrorMetric"),
}

__all__ = sorted(_LAZY)

if TYPE_CHECKING:
    from t4_e2e_devkit.planning.training.callbacks import (
        PredictionVizCallback,
        TrajectoryVizCallback,
    )
    from t4_e2e_devkit.planning.training.datamodule import T4DataModule
    from t4_e2e_devkit.planning.training.feature_cache import FeatureCache
    from t4_e2e_devkit.planning.training.lightning_module import T4LightningModule
    from t4_e2e_devkit.planning.training.official_score import OfficialDevkitScoreCallback


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
