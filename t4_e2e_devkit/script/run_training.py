"""Train an agent.

Hydra entry point, matching the reference repositories' launch convention::

    python -m t4_e2e_devkit.script.run_training \\
        agent=my_agent \\
        train_data_list=/path/to/t4_train.json \\
        val_data_list=/path/to/t4_val.json \\
        experiment_name=my_run \\
        trainer.max_epochs=10

Everything agent-specific comes from the agent.  This script owns the trainer,
the loaders and checkpoint selection -- nothing else.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from t4_e2e_devkit.agents.registry import build_agent
from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.datalist import describe_data_list, load_data_list
from t4_e2e_devkit.planning.training.datamodule import T4DataModule
from t4_e2e_devkit.planning.training.lightning_module import T4LightningModule
from t4_e2e_devkit.script.utils import build_reader_config, build_scorer_config

logger = logging.getLogger(__name__)

CONFIG_PATH = "config"
CONFIG_NAME = "training"


def run_training(cfg: DictConfig) -> str:
    """
    Train one agent.
    :param cfg: the resolved configuration.
    :return: path to the best checkpoint.
    """
    pl.seed_everything(int(cfg.get("seed") or 0), workers=True)

    for key in ("train_data_list", "val_data_list"):
        path = cfg.get(key)
        if path:
            logger.info("%s:\n%s", key, describe_data_list(load_data_list(path)))

    agent = build_agent(cfg.agent.name, **OmegaConf.to_container(cfg.agent.get("params", {})))
    agent.initialize()

    visualize = cfg.get("visualize")
    score_training_batches = bool(cfg.get("score_training_batches"))
    reader_config = build_reader_config(cfg)

    datamodule = T4DataModule(
        agent=agent,
        train_data_list=cfg.train_data_list,
        val_data_list=cfg.get("val_data_list"),
        scene_filter=SceneFilter(**OmegaConf.to_container(cfg.scene_filter)),
        reader_config=reader_config,
        return_scenes=score_training_batches,
        **OmegaConf.to_container(cfg.dataloader),
    )

    scorer = None
    if score_training_batches:
        from t4_e2e_devkit.evaluation.navsim_score import T4NavSimScorer

        scorer = T4NavSimScorer(build_scorer_config(cfg))

    module = T4LightningModule(agent=agent, scorer=scorer)

    output_dir = Path(cfg.get("output_dir") or ".") / str(cfg.get("experiment_name") or "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Which metric selects the best checkpoint is a real decision, not a
    # default.  The PDM report is deliberately independent from the objective;
    # it is logged for comparison and does not select a checkpoint by itself.
    monitor = cfg.get("monitor")
    if monitor is None:
        monitor = "val/loss" if cfg.get("val_data_list") else "train/loss"
    callbacks = [_checkpoint_callback(cfg, output_dir, monitor)]
    viz = visualize
    if viz and viz.get("enabled") and cfg.get("val_data_list"):
        from t4_e2e_devkit.planning.training.callbacks import TrajectoryVizCallback

        callbacks.append(
            TrajectoryVizCallback(
                data_list=cfg.val_data_list,
                num_windows=int(viz.get("num_windows") or 4),
                every_n_epochs=int(viz.get("every_n_epochs") or 1),
                mode=str(viz.get("mode") or "summary"),
                output_dir=output_dir / "windows",
                reader_config=reader_config,
            )
        )
        logger.info("window plots every %s epoch(s)", viz.get("every_n_epochs") or 1)

    checkpoint = callbacks[0]

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        **OmegaConf.to_container(cfg.trainer),
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.get("resume_from"))

    logger.info("best checkpoint: %s (%s)", checkpoint.best_model_path, checkpoint.best_model_score)
    return checkpoint.best_model_path


def _checkpoint_callback(cfg: DictConfig, output_dir: Path, monitor: str) -> ModelCheckpoint:
    """
    Build the checkpoint callback.
    :param cfg: the resolved configuration.
    :param output_dir: run directory.
    :param monitor: metric selecting the best checkpoint.
    :return: the callback.
    """
    return ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor=monitor,
        mode=str(cfg.get("monitor_mode") or "min"),
        save_top_k=int(cfg.get("save_top_k") or 3),
        save_last=True,
        filename="{epoch:03d}-{step:07d}",
    )


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.
    :param cfg: the resolved configuration.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(cfg)


if __name__ == "__main__":
    main()
