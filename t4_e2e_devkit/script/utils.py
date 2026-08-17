"""Shared config plumbing for the entry points.

These exist so that ``run_training`` and ``run_pdm_score`` cannot disagree about
how a config becomes a reader or a scorer.  The drivable-area buffer is the
reason: it is one number that three components must agree on -- the cache stores
it in its signature, the reader validates the cache against it, and the scorer
scores with it -- and duplicating the plumbing per entry point is exactly how
two of the three end up reading different fields.
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from t4_e2e_devkit.evaluation.pdm_score import T4PDMScorerConfig


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
    *,
    load_oracle_targets: bool = False,
) -> Dict[str, Any]:
    """
    Assemble the reader settings from a resolved config.
    :param cfg: the resolved configuration.
    :return: keyword settings for :class:`~t4_e2e_devkit.dataset.scene.T4SceneReader`.
    """
    reader: Dict[str, Any] = _container(cfg.get("reader"))
    reader["t4_drivable_area_buffer_m"] = float(cfg.get("drivable_area_buffer_m") or 0.0)
    if load_oracle_targets or cfg.get("pdm_cache_path"):
        reader["t4_load_oracle_targets"] = True
        reader.setdefault(
            "t4_oracle_device",
            "gpu" if str(cfg.get("backend") or "gpu").lower() == "gpu" else "cpu",
        )
    if cfg.get("pdm_cache_path"):
        reader["t4_pdm_reference_cache_dir"] = str(cfg.pdm_cache_path)
        # GPU oracle mode deliberately ignores this optional legacy artifact;
        # CPU mode uses it as its explicit offline reference source.
    return reader


def build_scorer_config(cfg: DictConfig) -> T4PDMScorerConfig:
    """
    Assemble the scorer settings from a resolved config.
    :param cfg: the resolved configuration.
    :return: the scorer configuration.
    """
    overrides: Dict[str, Any] = _container(cfg.get("scorer"))
    overrides.setdefault(
        "t4_drivable_area_buffer_m", float(cfg.get("drivable_area_buffer_m") or 0.0)
    )
    return T4PDMScorerConfig(**overrides)
