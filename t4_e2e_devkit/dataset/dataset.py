"""The map-style dataset every T4 model trains and evaluates on.

One row is ``(scene_dir, center_frame)``, so the dataset is a plain
``torch.utils.data.Dataset`` with random access and no iterable-shard
bookkeeping.  What it yields depends on what the agent asked for:

* with feature/target builders -- the ``(features, targets)`` tensor dicts the
  training loop consumes;
* without them -- the raw :class:`~t4_e2e_devkit.common.dataclasses.T4Scene`,
  which is what evaluation, caching and visualisation want.

Two access patterns matter for throughput and they pull in opposite directions.
Opening a scene is expensive (calibration, pack indices, file handles) and rows
of one scene share all of it, so the dataset keeps a small per-worker cache of
open builders.  But consecutive centres in one scene are near-duplicates -- at
stride 1 they share 30 of 31 history frames -- so filling a batch from one scene
makes the effective batch size much smaller than the nominal one.
:class:`T4SceneLocalitySampler` resolves that: it shuffles rows globally for the
batch composition while grouping each worker's reads by scene.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from t4_e2e_devkit.common.dataclasses import SceneFilter, SensorConfig, T4Scene
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.dataset.window import T4WindowBuilder

#: Open scene builders held per worker.  Each keeps file handles and a decoded
#: frame cache, so this trades memory for the repeated open cost; four covers
#: the boundary between scenes under a locality-preserving sampler without
#: pinning a large fraction of a scene's decoded frames.
DEFAULT_SCENE_CACHE_SIZE = 4


class T4Dataset(Dataset):
    """Map-style dataset over a T4 data list."""

    def __init__(
        self,
        data_list: str | Path | DataList,
        sensor_config: Optional[SensorConfig] = None,
        scene_filter: Optional[SceneFilter] = None,
        feature_builders: Optional[Sequence[Any]] = None,
        target_builders: Optional[Sequence[Any]] = None,
        reader_config: Optional[Dict[str, Any]] = None,
        scene_cache_size: int = DEFAULT_SCENE_CACHE_SIZE,
    ) -> None:
        """
        :param data_list: a data list, or the path to one.
        :param sensor_config: which sensors to decode; usually
            ``agent.get_sensor_config()``.
        :param scene_filter: the window shape.
        :param feature_builders: agent feature builders; raw scenes when absent.
        :param target_builders: agent target builders.
        :param reader_config: extra reader settings (frame cache, image size).
        :param scene_cache_size: open scene builders to keep per worker.
        """
        self.data_list = (
            data_list if isinstance(data_list, DataList) else load_data_list(data_list)
        )
        self.sensor_config = sensor_config or SensorConfig.build_no_sensors()
        self.scene_filter = scene_filter or SceneFilter()
        self.feature_builders = list(feature_builders or [])
        self.target_builders = list(target_builders or [])
        self.reader_config = dict(reader_config or {})
        self.scene_cache_size = int(scene_cache_size)
        if self.scene_cache_size < 1:
            raise ValueError(
                f"scene_cache_size must be at least 1, got {scene_cache_size}"
            )

        self._builders: "OrderedDict[str, T4WindowBuilder]" = OrderedDict()

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        scene_dir, center = self.data_list[index]
        scene = self.build_scene(scene_dir, center)
        if not self.feature_builders and not self.target_builders:
            return scene
        return self.build_features(scene)

    # ------------------------------------------------------------------ #
    # Scene access
    # ------------------------------------------------------------------ #

    def build_scene(self, scene_dir: str, center: int) -> T4Scene:
        """
        Assemble one window.
        :param scene_dir: scene directory relative to the data-list root.
        :param center: centre frame index.
        :return: the assembled scene.
        """
        return self._builder(scene_dir).build(int(center))

    def build_features(self, scene: T4Scene) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run the agent's builders over one scene.
        :param scene: an assembled scene.
        :return: ``(features, targets)``.
        """
        agent_input = scene.get_agent_input()
        features: Dict[str, Any] = {}
        for builder in self.feature_builders:
            features.update(builder.compute_features(agent_input))
        targets: Dict[str, Any] = {}
        for builder in self.target_builders:
            targets.update(builder.compute_targets(scene))
        return features, targets

    def _builder(self, scene_dir: str) -> T4WindowBuilder:
        builder = self._builders.get(scene_dir)
        if builder is not None:
            self._builders.move_to_end(scene_dir)
            return builder

        builder = T4WindowBuilder(
            self.data_list.absolute_scene_dir(scene_dir),
            self.data_list.root,
            sensor_config=self.sensor_config,
            scene_filter=self.scene_filter,
            reader_config=self.reader_config,
        )
        self._builders[scene_dir] = builder
        while len(self._builders) > self.scene_cache_size:
            _, evicted = self._builders.popitem(last=False)
            evicted.close()
        return builder

    def close(self) -> None:
        """Close every cached scene builder."""
        for builder in self._builders.values():
            builder.close()
        self._builders.clear()

    def __getstate__(self) -> dict:
        # File handles do not survive the fork into a DataLoader worker; each
        # worker reopens what it needs.
        state = self.__dict__.copy()
        state["_builders"] = OrderedDict()
        return state


class T4SceneLocalitySampler(Sampler):
    """Shuffles rows globally while keeping each rank's reads scene-local.

    A uniform shuffle makes every consecutive read land in a different scene, so
    the scene cache never hits and each row pays the full open cost.  Grouping
    strictly by scene fixes that but fills batches with near-duplicate windows,
    which quietly shrinks the effective batch size.

    The compromise: shuffle scenes, shuffle rows inside each scene, then split
    the concatenation across ranks.  Reads stay local because a rank's slice
    spans few scenes, and batch composition stays varied because a batch is
    drawn across the whole slice rather than from one scene's block.
    """

    def __init__(
        self,
        dataset: T4Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to sample from.
        :param num_replicas: DDP world size; read from the process group by default.
        :param rank: this process's rank; read from the process group by default.
        :param shuffle: shuffle scenes and rows each epoch.
        :param seed: base seed; the epoch is mixed in.
        :param drop_last: drop the tail so every rank sees the same count.
        """
        if num_replicas is None or rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = num_replicas or torch.distributed.get_world_size()
                rank = torch.distributed.get_rank() if rank is None else rank
            else:
                num_replicas, rank = 1, 0

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas < 1:
            raise ValueError(f"num_replicas must be at least 1, got {num_replicas}")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self._scene_rows: Dict[str, List[int]] = {}
        for index, (scene_dir, _) in enumerate(dataset.data_list):
            self._scene_rows.setdefault(scene_dir, []).append(index)

        total = len(dataset)
        if drop_last:
            self.num_samples = total // self.num_replicas
        else:
            self.num_samples = math.ceil(total / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """
        Reseed the shuffle for a new epoch.
        :param epoch: the epoch index.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        scenes = list(self._scene_rows)
        indices: List[int] = []
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(scenes)
            for scene in scenes:
                rows = list(self._scene_rows[scene])
                rng.shuffle(rows)
                indices.extend(rows)
        else:
            for scene in scenes:
                indices.extend(self._scene_rows[scene])

        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices += indices[:padding]
        indices = indices[: self.total_size]
        return iter(indices[self.rank : self.total_size : self.num_replicas])


def collate_t4(batch: Sequence[Any]) -> Any:
    """Collate T4 samples without padding the ragged streams.

    Fixed-shape arrays stack into tensors.  Point clouds and per-frame object
    lists stay Python lists: padding them means either a per-batch maximum that
    changes the tensor shape run to run, or a fixed cap that silently drops
    objects, and both have produced real bugs.  Strings and scenes pass through
    as lists.

    :param batch: samples from :class:`T4Dataset`.
    :return: the collated batch, matching the input's structure.
    """
    if not batch:
        raise ValueError("cannot collate an empty T4 batch")
    first = batch[0]

    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[0], dict):
        features = _collate_mapping([sample[0] for sample in batch])
        targets = _collate_mapping([sample[1] for sample in batch])
        return features, targets
    if isinstance(first, dict):
        return _collate_mapping(list(batch))
    # Raw scenes and anything else: keep the list, let the caller decide.
    return list(batch)


def _collate_mapping(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    keys = list(samples[0])
    missing = [
        (index, sorted(set(keys) ^ set(sample)))
        for index, sample in enumerate(samples)
        if set(sample) != set(keys)
    ]
    if missing:
        raise KeyError(
            f"samples in one batch disagree about their keys; first mismatch at "
            f"sample {missing[0][0]}, differing keys {missing[0][1]}"
        )

    out: Dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        out[key] = _collate_values(values)
    return out


def _collate_values(values: List[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if all(value.shape == first.shape for value in values):
            return torch.stack(values, dim=0)
        return values  # ragged: point clouds, per-sample object tensors
    if isinstance(first, np.ndarray):
        if all(value.shape == first.shape for value in values):
            return torch.from_numpy(np.stack(values, axis=0))
        return [torch.from_numpy(np.ascontiguousarray(value)) for value in values]
    if isinstance(first, (bool, np.bool_)):
        return list(values)
    if isinstance(first, (int, np.integer)):
        return torch.as_tensor(values, dtype=torch.int64)
    if isinstance(first, (float, np.floating)):
        return torch.as_tensor(values, dtype=torch.float32)
    return list(values)


def build_dataset_from_agent(
    data_list: str | Path | DataList,
    agent,
    scene_filter: Optional[SceneFilter] = None,
    reader_config: Optional[Dict[str, Any]] = None,
    include_targets: bool = True,
) -> T4Dataset:
    """Construct the dataset an agent needs, from the agent itself.

    This is the seam that makes the interface unified in practice rather than
    only on paper: the agent declares its sensors and its builders, and nothing
    about the data pipeline has to be configured per model.

    :param data_list: a data list, or the path to one.
    :param agent: an :class:`~t4_e2e_devkit.agents.AbstractT4Agent`.
    :param scene_filter: the window shape.
    :param reader_config: extra reader settings.
    :param include_targets: build supervision targets (training) or not (inference).
    :return: the configured dataset.
    """
    return T4Dataset(
        data_list=data_list,
        sensor_config=agent.get_sensor_config(),
        scene_filter=scene_filter,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders() if include_targets else [],
        reader_config=reader_config,
    )
