"""Dataset protocol tests that do not require a T4 dataset on disk."""

from __future__ import annotations

import pytest

from t4_e2e_devkit.dataset.datalist import DataList
from t4_e2e_devkit.dataset.dataset import T4Dataset, T4SceneLocalitySampler, collate_t4


def _dataset(tmp_path):
    return T4Dataset(DataList(root=tmp_path, rows=[("prd_jt/a", 1), ("prd_jt/b", 2)]))


def test_collate_keeps_boolean_metadata_boolean(tmp_path):
    del tmp_path
    assert collate_t4([{"available": True}, {"available": False}])["available"] == [True, False]


def test_collate_rejects_empty_batches(tmp_path):
    del tmp_path
    with pytest.raises(ValueError, match="empty"):
        collate_t4([])


def test_dataset_zero_cache_builds_windows_online(tmp_path):
    dataset = T4Dataset(DataList(root=tmp_path, rows=[("prd_jt/a", 1)]), scene_cache_size=0)
    assert dataset.scene_cache_size == 0
    assert dataset._builders == {}


def test_dataset_cache_size_rejects_negative(tmp_path):
    with pytest.raises(ValueError, match="scene_cache_size"):
        T4Dataset(DataList(root=tmp_path, rows=[("prd_jt/a", 1)]), scene_cache_size=-1)


def test_sampler_validates_rank(tmp_path):
    dataset = _dataset(tmp_path)
    with pytest.raises(ValueError, match="rank"):
        T4SceneLocalitySampler(dataset, num_replicas=2, rank=2)
