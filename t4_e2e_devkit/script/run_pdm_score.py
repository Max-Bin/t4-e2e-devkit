"""Score an agent over a data list.

Hydra entry point, matching the reference repositories' launch convention::

    python -m t4_e2e_devkit.script.run_pdm_score \\
        agent=constant_velocity \\
        data_list=/path/to/t4_val.json \\
        pdm_cache_path=/path/to/t4-pdm-reference-cache \\
        experiment_name=cv_baseline

The GPU backend computes the exact PDM-Closed reference online on CUDA and does
not require a cache. The optional ``pdm_cache_path`` is used only by an explicit
CPU/offline reference run.

Writes a per-window CSV and prints the aggregate.  The per-window file is the
point: an aggregate PDM score says a run got 0.42, and only the per-window
components say whether that came from a few zeroed windows or from uniformly
mediocre driving -- and those two call for opposite next steps.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Mapping

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from t4_e2e_devkit.agents.registry import build_agent
from t4_e2e_devkit.common.constants import PDM_COMPONENT_ORDER
from t4_e2e_devkit.common.dataclasses import PDMResults, SceneFilter
from t4_e2e_devkit.dataset.datalist import describe_data_list, load_data_list
from t4_e2e_devkit.dataset.dataset import T4Dataset
from t4_e2e_devkit.evaluation.open_loop import (
    OpenLoopMetrics,
    compute_open_loop_metrics,
)
from t4_e2e_devkit.evaluation.pdm_score import T4PDMScorer
from t4_e2e_devkit.evaluation.report import aggregate_evaluation
from t4_e2e_devkit.script.utils import build_reader_config, build_scorer_config

logger = logging.getLogger(__name__)

CONFIG_PATH = "config"
CONFIG_NAME = "pdm_score"


def run_pdm_score(cfg: DictConfig) -> Dict[str, Dict[str, float]]:
    """
    Score one agent over one data list.
    :param cfg: the resolved configuration.
    :return: the aggregate report.
    """
    data_list = load_data_list(cfg.data_list)
    if cfg.get("max_scenes"):
        data_list = data_list.filtered(max_rows=int(cfg.max_scenes))
    logger.info("data list:\n%s", describe_data_list(data_list))

    agent = build_agent(cfg.agent.name, **OmegaConf.to_container(cfg.agent.get("params", {})))
    if cfg.get("checkpoint_path"):
        _load_checkpoint(agent, cfg.checkpoint_path)
    agent.initialize()
    # `cfg.get(key, default)` returns None for a key that is present and null,
    # which every optional field in the config tree is -- so the fallback has
    # to be an `or`, not the get() default.
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent.to(device)

    reader_config = build_reader_config(cfg, load_oracle_targets=True)
    scene_filter = SceneFilter(**OmegaConf.to_container(cfg.scene_filter))
    dataset = T4Dataset(
        data_list,
        sensor_config=agent.get_sensor_config(),
        scene_filter=scene_filter,
        reader_config=reader_config,
    )

    scorer = T4PDMScorer(
        backend=cfg.get("backend") or "gpu",
        device=str(device),
        config=build_scorer_config(cfg),
        include_tier4_metrics=bool(cfg.get("include_tier4_metrics", False)),
    )

    batch_size = int(cfg.get("batch_size") or 8)
    results: List[PDMResults] = []
    open_loop_results: List[OpenLoopMetrics] = []
    tier4_results: List[Mapping[str, float]] = []
    tier4_rows: List[tuple[str | None, Mapping[str, float]]] = []
    failures: List[tuple] = []

    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        scenes, trajectories = [], []
        for index in indices:
            try:
                scene = dataset[index]
                trajectories.append(_plan(agent, scene))
                scenes.append(scene)
            except Exception as error:  # noqa: BLE001
                failures.append((dataset.data_list[index], repr(error)))
        if not scenes:
            continue
        try:
            batch_results = scorer.score_batch(trajectories, scenes)
            results.extend(batch_results)
            for trajectory, scene, result in zip(
                trajectories, scenes, batch_results, strict=True
            ):
                try:
                    open_loop_results.append(
                        compute_open_loop_metrics(
                            trajectory,
                            scene,
                            token=scene.scene_metadata.token,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning(
                        "open-loop metrics unavailable for %s: %s",
                        scene.scene_metadata.token,
                        error,
                    )
                if result.tier4_metrics:
                    tier4_results.append(result.tier4_metrics)
                    tier4_rows.append((result.token, result.tier4_metrics))
        except Exception as error:  # noqa: BLE001
            for scene in scenes:
                failures.append(((scene.scene_metadata.token,), repr(error)))
        logger.info("scored %d/%d windows", len(results), len(dataset))

    dataset.close()

    report = aggregate_evaluation(
        pdm=results,
        open_loop=open_loop_results,
        tier4=tier4_results,
        num_failed=len(failures),
    )
    output_dir = Path(cfg.get("output_dir") or ".") / str(cfg.get("experiment_name") or "pdm_score")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_window.csv", results)
    _write_open_loop_csv(output_dir / "open_loop.csv", open_loop_results)
    if tier4_rows:
        _write_tier4_csv(output_dir / "tier4.csv", tier4_rows)

    # Skipped windows are reported, never averaged away.  A run that scored 60%
    # of its list and reported the mean of those looks like a better run than
    # one that scored all of it.
    if failures:
        with (output_dir / "failures.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "error"])
            writer.writerows(failures)
        logger.warning(
            "%d of %d windows could not be scored; see %s",
            len(failures), len(dataset), output_dir / "failures.csv",
        )

    logger.info(
        "aggregate: %s",
        {
            family: {key: round(value, 4) for key, value in values.items()}
            for family, values in report.items()
        },
    )
    OmegaConf.save(OmegaConf.create(report), output_dir / "aggregate.yaml")
    return report


def _plan(agent, scene):
    """Plan with an agent, honouring whether it is an oracle."""
    if agent.requires_scene:
        return agent.compute_trajectory_from_scene(scene)
    return agent.compute_trajectory(scene.get_agent_input())


def _load_checkpoint(agent, path: str | Path) -> None:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    # Lightning prefixes every key with the module attribute name.
    state = {key.removeprefix("agent."): value for key, value in state.items()}
    missing, unexpected = agent.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(
            "checkpoint %s: %d missing and %d unexpected keys",
            path, len(missing), len(unexpected),
        )


def _write_csv(path: Path, results: List[PDMResults]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", *PDM_COMPONENT_ORDER, "score"])
        for result in results:
            components = result.components
            writer.writerow(
                [result.token]
                + [f"{components[name]:.6f}" for name in PDM_COMPONENT_ORDER]
                + [f"{result.score:.6f}"]
            )


def _write_open_loop_csv(path: Path, results: List[OpenLoopMetrics]) -> None:
    names = list(results[0].values) if results else []
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", *names])
        for result in results:
            writer.writerow([result.token] + [f"{result.values[name]:.6f}" for name in names])


def _write_tier4_csv(
    path: Path,
    results: List[tuple[str | None, Mapping[str, float]]],
) -> None:
    names = sorted({name for _, values in results for name in values})
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", *names])
        for token, values in results:
            writer.writerow([token] + [f"{values.get(name, float('nan')):.6f}" for name in names])


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.
    :param cfg: the resolved configuration.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pdm_score(cfg)


if __name__ == "__main__":
    main()
