"""Training callbacks.

The visualisation is only useful if it runs. A plotting library nobody calls
during a twelve-hour job is a plotting library that rots, so this is the hook
that puts a figure in front of whoever is watching the run.

:class:`TrajectoryVizCallback` renders fixed dataset windows for devkit agents.
:class:`PredictionVizCallback` accepts raw prediction samples and uses the
shared BEV renderer for a caller-provided image logger.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class TrajectoryVizCallback(pl.Callback):
    """Render a fixed set of windows during training.

    Runs on rank zero only and never raises into the training loop: a rendering
    failure at epoch 30 of a long job must not lose the run. Failures are logged
    once per epoch and the callback disables itself after repeated ones, since a
    broken renderer that logs every epoch buries the training log it shares.
    """

    def __init__(
        self,
        data_list: str | Path,
        centers: Optional[Sequence[int]] = None,
        num_windows: int = 4,
        every_n_epochs: int = 1,
        output_dir: Optional[str | Path] = None,
        mode: str = "summary",
        reader_config: Optional[Dict[str, Any]] = None,
        max_failures: int = 3,
    ) -> None:
        """
        :param data_list: the list to draw windows from; usually the validation one.
        :param centers: explicit ``(row index)`` selections; evenly spaced otherwise.
        :param num_windows: how many windows to render when ``centers`` is omitted.
        :param every_n_epochs: render cadence.
        :param output_dir: also write PNGs here.
        :param mode: ``"bev"``, ``"cameras"`` or ``"summary"``.
        :param reader_config: extra reader settings, e.g. an optional PDM cache,
            so the reference path can be drawn.
        :param max_failures: disable the callback after this many failed epochs.
        """
        super().__init__()
        self.data_list = data_list
        self.centers = list(centers) if centers is not None else None
        self.num_windows = int(num_windows)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.output_dir = Path(output_dir) if output_dir else None
        self.mode = mode
        self.reader_config = dict(reader_config or {})
        self.max_failures = int(max_failures)

        self._rows: Optional[List] = None
        self._failures = 0
        self._disabled = False

    # ------------------------------------------------------------------ #

    def _resolve_rows(self) -> List:
        """Pick the windows once, so every epoch renders the same ones."""
        from t4_e2e_devkit.dataset.datalist import load_data_list

        data_list = load_data_list(self.data_list)
        if self.centers is not None:
            return [data_list[index] for index in self.centers if index < len(data_list)]
        if len(data_list) <= self.num_windows:
            return list(data_list)
        # Evenly spaced across the list rather than the first N, which would all
        # come from one scene and show one road.
        step = len(data_list) // self.num_windows
        return [data_list[index * step] for index in range(self.num_windows)]

    def _render(self, agent, epoch: int) -> List:
        from t4_e2e_devkit.dataset.datalist import load_data_list
        from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene
        from t4_e2e_devkit.dataset.window import T4WindowBuilder
        from t4_e2e_devkit.visualization import (
            figure_to_rgb,
            plot_bev_frame,
            plot_cameras_frame,
            plot_scene_summary,
            reference_trajectories,
            save_figure,
        )

        data_list = load_data_list(self.data_list)
        needs_cameras = self.mode in ("cameras", "summary")
        images: List = []

        for scene_dir, center in self._rows or []:
            absolute = data_list.absolute_scene_dir(scene_dir)
            sensor_config = (
                sensor_config_for_scene(absolute) if needs_cameras else None
            )
            builder = T4WindowBuilder(
                absolute, data_list.root,
                sensor_config=sensor_config,
                reader_config=self.reader_config,
            )
            try:
                scene = builder.build(int(center))
                trajectories = reference_trajectories(scene)
                trajectories["prediction"] = (
                    agent.compute_trajectory_from_scene(scene)
                    if getattr(agent, "requires_scene", False)
                    else agent.compute_trajectory(scene.get_agent_input())
                )
                if self.mode == "bev":
                    figure, _ = plot_bev_frame(scene, trajectories)
                elif self.mode == "cameras":
                    figure, _ = plot_cameras_frame(scene, with_annotations=True)
                else:
                    figure, _ = plot_scene_summary(scene, trajectories)

                images.append((scene.scene_metadata.token, figure_to_rgb(figure)))
                if self.output_dir is not None:
                    name = f"epoch{epoch:03d}_{scene.scene_metadata.token.replace('/', '_')}.png"
                    save_figure(figure, self.output_dir / name)
                else:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
            finally:
                builder.close()
        return images

    @staticmethod
    def _log(trainer: pl.Trainer, images: List, epoch: int) -> None:
        """Send the figures wherever the run is already logging."""
        logger_obj = getattr(trainer, "logger", None)
        experiment = getattr(logger_obj, "experiment", None)
        if experiment is None:
            return
        log_image = getattr(experiment, "log_image", None)
        if callable(log_image):
            log_image(
                key="val/windows",
                images=[image for _, image in images],
                caption=[token for token, _ in images],
            )
            return
        # TensorBoard
        if hasattr(experiment, "add_image"):
            for token, image in images:
                experiment.add_image(
                    f"windows/{token}", image, global_step=trainer.global_step, dataformats="HWC"
                )

    # ------------------------------------------------------------------ #

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Render and log after a validation epoch.
        :param trainer: the Lightning trainer.
        :param pl_module: the module being trained.
        """
        if self._disabled or not trainer.is_global_zero or trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        agent = getattr(pl_module, "agent", pl_module)
        was_training = agent.training
        try:
            if self._rows is None:
                self._rows = self._resolve_rows()
            agent.eval()
            images = self._render(agent, trainer.current_epoch)
            if images:
                self._log(trainer, images, trainer.current_epoch)
            self._failures = 0
        except Exception as error:  # noqa: BLE001
            # Never raise into the training loop: losing a twelve-hour run to a
            # plotting bug is a strictly worse outcome than losing the plot.
            self._failures += 1
            logger.warning(
                "TrajectoryVizCallback failed at epoch %d (%d/%d): %r",
                trainer.current_epoch, self._failures, self.max_failures, error,
            )
            if self._failures >= self.max_failures:
                self._disabled = True
                logger.warning(
                    "TrajectoryVizCallback disabled after %d consecutive failures; "
                    "training continues without window plots",
                    self._failures,
                )
        finally:
            if was_training:
                agent.train()


class PredictionVizCallback(pl.Callback):
    """Render model-owned BEV prediction samples through a generic image logger.

    Model repositories can keep their own batch and inference code while
    exposing ``_viz_samples`` on the Lightning module. Each item must contain
    ``gt_xy`` and ``pred_xy`` plus optional raw ``lanes`` and ``route`` arrays.
    The callback owns the renderer but not the logging backend. It calls a
    trainer logger's ``log_image`` method when one is available. It is
    deliberately best-effort: a plotting or logging failure never aborts
    training.
    """

    def __init__(
        self,
        n_samples: int = 4,
        every_n_epochs: int = 1,
        view_range: float = 60.0,
        max_failures: int = 3,
    ) -> None:
        super().__init__()
        self.n_samples = int(n_samples)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.view_range = float(view_range)
        self.max_failures = max(1, int(max_failures))
        self._failures = 0
        self._disabled = False

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        has_media_logger = any(
            callable(getattr(candidate, "log_image", None))
            for candidate in (getattr(trainer, "loggers", []) or [])
        )
        active = (
            not self._disabled
            and trainer.is_global_zero
            and has_media_logger
            and trainer.current_epoch % self.every_n_epochs == 0
        )
        pl_module._viz_capacity = self.n_samples if active else 0
        pl_module._viz_samples = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        samples = getattr(pl_module, "_viz_samples", [])
        pl_module._viz_capacity = 0
        if not samples or not trainer.is_global_zero or trainer.sanity_checking:
            pl_module._viz_samples = []
            return

        log_image = None
        for candidate in getattr(trainer, "loggers", []) or []:
            method = getattr(candidate, "log_image", None)
            if callable(method):
                log_image = method
                break
        if log_image is None:
            pl_module._viz_samples = []
            return

        try:
            import numpy as np

            from t4_e2e_devkit.visualization import render_prediction_bev

            images, captions = [], []
            for index, sample in enumerate(samples):
                gt = sample["gt_xy"]
                pred = sample["pred_xy"]
                ade = float(np.linalg.norm(pred - gt, axis=-1).mean()) if len(gt) else float("nan")
                images.append(
                    render_prediction_bev(
                        gt,
                        pred,
                        lanes=sample.get("lanes"),
                        route=sample.get("route"),
                        view_range=self.view_range,
                        title=f"epoch {trainer.current_epoch} sample {index} ADE {ade:.2f} m",
                    )
                )
                captions.append(f"ep{trainer.current_epoch} s{index} ADE={ade:.2f}m")
            log_image(key="val/bev_trajectory", images=images, caption=captions)
            self._failures = 0
        except Exception as error:  # noqa: BLE001
            self._failures += 1
            logger.warning(
                "PredictionVizCallback failed at epoch %d (%d/%d): %r",
                trainer.current_epoch,
                self._failures,
                self.max_failures,
                error,
            )
            if self._failures >= self.max_failures:
                self._disabled = True
                logger.warning("PredictionVizCallback disabled after repeated failures")
        finally:
            pl_module._viz_samples = []
