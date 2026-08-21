"""Shared config plumbing for the entry points.

These helpers keep the Hydra entry points aligned on reader and metric
configuration.
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from t4_e2e_devkit.evaluation.navsim_score import T4NavSimScorerConfig


def _container(node: Any) -> Dict[str, Any]:
    """A plain dict from a config node, dropping keys left null.

    ``OmegaConf.to_container`` rejects a node that is already a plain dict --
    which is what an empty mapping in the config tree resolves to -- so the type
    is checked rather than assumed.  Null keys are dropped so a component's own
    default applies instead of being overridden with ``None``.
    """
    if node is None:
        return {}
    values = OmegaConf.to_container(node) if OmegaConf.is_config(node) else dict(node)
    return {key: value for key, value in (values or {}).items() if value is not None}


def build_reader_config(
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Assemble the reader settings from a resolved config.
    :param cfg: the resolved configuration.
    :return: keyword settings for :class:`~t4_e2e_devkit.dataset.scene.T4SceneReader`.
    """
    return _container(cfg.get("reader"))


def build_scorer_config(cfg: DictConfig) -> T4NavSimScorerConfig:
    """
    Assemble the scorer settings from a resolved config.
    :param cfg: the resolved configuration.
    :return: the PDM/NavSim scorer configuration.
    """
    overrides: Dict[str, Any] = _container(cfg.get("scorer"))
    overrides.setdefault(
        "version", str(cfg.get("pdm_version") or "navsim-v2").removeprefix("navsim-")
    )
    overrides.setdefault("backend", str(cfg.get("backend") or "gpu"))
    if cfg.get("device") is not None:
        overrides.setdefault("device", str(cfg.device))
    return T4NavSimScorerConfig(**overrides)
