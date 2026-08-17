"""The Lightning DataModule: two data lists in, two loaders out.

Nothing here is agent-specific.  The agent's sensor config and builders decide
what a sample contains, and this module only decides how samples become batches:
worker count, prefetch, the locality-preserving sampler, and the collate
function that keeps ragged streams ragged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent
from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.dataset import (
    T4Dataset,
    T4SceneLocalitySampler,
    build_dataset_from_agent,
    collate_t4,
)


class T4DataModule(pl.LightningDataModule):
    """Builds the train and validation loaders for one agent."""

    def __init__(
        self,
        agent: AbstractT4Agent,
        train_data_list: str | Path,
        val_data_list: Optional[str | Path] = None,
        scene_filter: Optional[SceneFilter] = None,
        reader_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        seed: int = 0,
    ) -> None:
        """
        :param agent: the agent whose sensors and builders drive the pipeline.
        :param train_data_list: the training data list.
        :param val_data_list: the validation data list; omit to skip validation.
        :param scene_filter: the window shape.
        :param reader_config: extra reader settings (frame cache, optional PDM cache).
        :param batch_size: samples per rank.
        :param num_workers: DataLoader workers per rank.
        :param prefetch_factor: batches prefetched per worker.
        :param pin_memory: pin host memory for the device copy.
        :param seed: sampler seed.
        """
        super().__init__()
        self.agent = agent
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.scene_filter = scene_filter or SceneFilter()
        self.reader_config = dict(reader_config or {})
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.seed = seed

        self._train: Optional[T4Dataset] = None
        self._val: Optional[T4Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build the datasets.
        :param stage: Lightning stage; both datasets are built regardless.
        """
        if self._train is None:
            self._train = build_dataset_from_agent(
                self.train_data_list,
                self.agent,
                scene_filter=self.scene_filter,
                reader_config=self.reader_config,
            )
        if self._val is None and self.val_data_list is not None:
            self._val = build_dataset_from_agent(
                self.val_data_list,
                self.agent,
                scene_filter=self.scene_filter,
                reader_config=self.reader_config,
            )

    def _loader(self, dataset: T4Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=T4SceneLocalitySampler(
                dataset, shuffle=shuffle, seed=self.seed, drop_last=shuffle
            ),
            num_workers=self.num_workers,
            collate_fn=collate_t4,
            pin_memory=self.pin_memory,
            # prefetch_factor is only valid with workers, and passing it with
            # num_workers=0 is a TypeError rather than a no-op.
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """:return: the training loader."""
        if self._train is None:
            self.setup()
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        """:return: the validation loader, or ``None`` when no list was given."""
        if self._val is None:
            self.setup()
        return None if self._val is None else self._loader(self._val, shuffle=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Close reader file handles.
        :param stage: Lightning stage.
        """
        for dataset in (self._train, self._val):
            if dataset is not None:
                dataset.close()
