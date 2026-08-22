"""Dataset protocol tests that do not require a T4 dataset on disk."""

from __future__ import annotations

from pathlib import Path

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


class TestLidarPackLocation:
    """Scene index to pack index, with the range that makes it meaningful."""

    @staticmethod
    def _location(**meta):
        from t4_e2e_devkit.dataset.lidar_pack import lidar_pack_location

        return lidar_pack_location(meta, Path("/t4"))

    def test_a_relative_pack_resolves_against_the_root(self):
        location = self._location(lidar_pack="s/data/LIDAR_CONCAT.pack", lidar_frames=10)
        assert location.path == Path("/t4/s/data/LIDAR_CONCAT.pack")

    def test_an_absolute_pack_is_left_alone(self):
        location = self._location(lidar_pack="/elsewhere/p.pack", lidar_frames=10)
        assert location.path == Path("/elsewhere/p.pack")

    def test_an_index_outside_the_range_carries_no_sweep(self):
        # The failure this type exists to prevent: with the offset alone, frame
        # 4 maps to pack index 6 and reads a sweep that is not its own.
        location = self._location(
            lidar_pack="p.pack", lidar_first_frame=5, lidar_frames=10, frame_offset=2
        )
        assert location.pack_index(4) is None
        assert location.pack_index(5) == 7
        assert location.pack_index(14) == 16
        assert location.pack_index(15) is None

    def test_no_pack_means_no_sweeps_rather_than_an_error(self):
        location = self._location()
        assert location.path is None
        assert location.pack_index(0) is None

    def test_a_declared_pack_with_an_empty_range_is_refused(self):
        with pytest.raises(ValueError, match="LiDAR frame range"):
            self._location(lidar_pack="p.pack", lidar_frames=0)
