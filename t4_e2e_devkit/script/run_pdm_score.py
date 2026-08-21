"""Score an agent with the unified PDM metric family."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from t4_e2e_devkit.agents.registry import build_agent
from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.datalist import describe_data_list, load_data_list
from t4_e2e_devkit.dataset.dataset import T4Dataset
from t4_e2e_devkit.evaluation.navsim_score import T4NavSimScorer
from t4_e2e_devkit.evaluation.open_loop import OpenLoopMetrics, compute_open_loop_metrics
from t4_e2e_devkit.evaluation.report import aggregate_evaluation
from t4_e2e_devkit.script.utils import (
    build_reader_config,
    build_scorer_config,
    load_agent_checkpoint,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config"
CONFIG_NAME = "pdm_score"


def run_pdm_score(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Score one agent over one data list and write family-separated reports."""

    data_list = load_data_list(cfg.data_list)
    if cfg.get("max_scenes"):
        data_list = data_list.filtered(max_rows=int(cfg.max_scenes))
    logger.info("data list:\n%s", describe_data_list(data_list))

    agent = build_agent(cfg.agent.name, **OmegaConf.to_container(cfg.agent.get("params", {})))
    if cfg.get("checkpoint_path"):
        load_agent_checkpoint(agent, cfg.checkpoint_path)
    agent.initialize()
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    agent.to(device)

    dataset = T4Dataset(
        data_list,
        sensor_config=agent.get_sensor_config(),
        scene_filter=SceneFilter(**OmegaConf.to_container(cfg.scene_filter)),
        reader_config=build_reader_config(cfg),
    )
    scorer = T4NavSimScorer(build_scorer_config(cfg))
    batch_size = int(cfg.get("batch_size") or 8)
    pdm_rows: list[Mapping[str, float]] = []
    open_loop_rows: list[OpenLoopMetrics] = []
    failures: list[tuple[Any, str]] = []

    for start in range(0, len(dataset), batch_size):
        scenes = []
        trajectories = []
        for index in range(start, min(start + batch_size, len(dataset))):
            try:
                scene = dataset[index]
                scenes.append(scene)
                trajectories.append(_plan(agent, scene))
            except Exception as error:  # noqa: BLE001 - row failures belong in the report
                failures.append((data_list.rows[index], repr(error)))
        if not scenes:
            continue
        try:
            results = scorer.score_batch(trajectories, scenes)
            pdm_rows.extend(result.values for result in results)
            for trajectory, scene in zip(trajectories, scenes, strict=True):
                try:
                    open_loop_rows.append(
                        compute_open_loop_metrics(
                            trajectory,
                            scene,
                            token=scene.scene_metadata.token,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - optional family
                    logger.warning(
                        "open-loop metrics unavailable for %s: %s",
                        scene.scene_metadata.token,
                        error,
                    )
        except Exception as error:  # noqa: BLE001 - row failures belong in the report
            failures.extend((scene.scene_metadata.token, repr(error)) for scene in scenes)
        logger.info("scored %d/%d windows", len(pdm_rows), len(dataset))

    dataset.close()
    report = aggregate_evaluation(
        pdm=pdm_rows,
        open_loop=open_loop_rows,
        num_failed=len(failures),
    )
    output_dir = Path(cfg.get("output_dir") or ".") / str(cfg.get("experiment_name") or "pdm_score")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pdm_csv(output_dir / "pdm.csv", pdm_rows)
    _write_open_loop_csv(output_dir / "open_loop.csv", open_loop_rows)
    if failures:
        with (output_dir / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "error"])
            writer.writerows(failures)
    OmegaConf.save(OmegaConf.create(report), output_dir / "aggregate.yaml")
    logger.info("aggregate: %s", report)
    return report


def _plan(agent: Any, scene: Any):
    with torch.inference_mode():
        if agent.requires_scene:
            return agent.compute_trajectory_from_scene(scene)
        return agent.compute_trajectory(scene.get_agent_input())


def _write_pdm_csv(path: Path, results: list[Mapping[str, float]]) -> None:
    names = sorted({str(name) for result in results for name in result})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", *names])
        for index, result in enumerate(results):
            writer.writerow([index, *[result.get(name, "") for name in names]])


def _write_open_loop_csv(path: Path, results: list[OpenLoopMetrics]) -> None:
    names = list(results[0].values) if results else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", *names])
        for result in results:
            writer.writerow([result.token, *[result.values[name] for name in names]])


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pdm_score(cfg)


if __name__ == "__main__":
    main()
