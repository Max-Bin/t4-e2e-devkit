from __future__ import annotations

import numpy as np
import pytest

from t4_e2e_devkit.evaluation.executor import LocalExecutor, rank_indices
from t4_e2e_devkit.planning.training.feature_cache import FeatureCache


def _square(value: int) -> int:
    return value * value


def test_feature_cache_round_trip_signature_and_raw_payload_guard(tmp_path):
    cache = FeatureCache(tmp_path, version="feature-v2")
    key = cache.key("prd_jt/scene@10", signature="builder-a")
    path = cache.save(
        key,
        {
            "map": np.arange(6, dtype=np.float32).reshape(2, 3),
            "mask": np.array([True, False]),
        },
    )
    loaded = cache.load(key)
    assert path.is_file()
    assert loaded is not None
    np.testing.assert_array_equal(loaded["map"], np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(loaded["mask"], [True, False])
    assert cache.key("prd_jt/scene@10", signature="builder-b") != key

    with pytest.raises(TypeError, match="raw sensor bytes"):
        cache.save(cache.key("scene"), {"image": b"jpeg"})


def test_feature_cache_get_or_compute_only_runs_on_a_miss(tmp_path):
    cache = FeatureCache(tmp_path)
    key = cache.key("scene")
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"value": np.array([4.0], dtype=np.float32)}

    assert cache.get_or_compute(key, compute)["value"][0] == 4.0
    assert cache.get_or_compute(key, compute)["value"][0] == 4.0
    assert calls["count"] == 1


def test_local_executor_is_ordered_and_partitions_by_rank_deterministically():
    values = list(range(8))
    assert rank_indices(8, rank=1, world_size=3) == (1, 4, 7)
    executor = LocalExecutor(workers=1)
    assert executor.map(_square, values) == [value * value for value in values]
    assert executor.select_rank(values, rank=1, world_size=3) == [1, 4, 7]
    assert executor.map_indexed(_square, values, rank=1, world_size=3) == [
        (1, 1),
        (4, 16),
        (7, 49),
    ]


def test_local_executor_uses_multiple_processes_when_requested():
    values = list(range(6))
    result = LocalExecutor(workers=2, start_method="spawn").map(_square, values)
    assert result == [value * value for value in values]
