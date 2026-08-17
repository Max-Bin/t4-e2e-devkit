from __future__ import annotations

import numpy as np
import pytest

from t4_e2e_devkit.common.dataclasses import Trajectory
from t4_e2e_devkit.evaluation.metric_cache import MetricCache
from t4_e2e_devkit.evaluation.metric_engine import MetricContext, MetricEngine
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def test_metric_engine_registers_evaluates_aggregates_and_caches(tmp_path):
    calls = {"count": 0}

    def compute(context):
        calls["count"] += 1
        return {"error": float(context.metadata["error"]), "stable": 1.0}

    engine = MetricEngine()
    engine.register("toy", compute, family="open_loop")
    cache = MetricCache(tmp_path)
    context = MetricContext(token="scene@1", metadata={"error": 2.0})

    first = engine.evaluate(context, cache=cache)
    second = engine.evaluate(context, cache=cache)
    assert calls["count"] == 1
    assert first.aggregate() == {
        "open_loop": {"num_records": 1.0, "error": 2.0, "stable": 1.0}
    }
    assert second.records[0].as_dict() == first.records[0].as_dict()

    changed = engine.evaluate(
        MetricContext(token="scene@1", metadata={"error": 3.0}),
        cache=cache,
    )
    assert calls["count"] == 2
    assert changed.records[0].values["error"] == 3.0

    with pytest.raises(ValueError, match="already registered"):
        engine.register("toy", compute)


def test_default_open_loop_adapter_accepts_sparse_and_dense_grids():
    sampling = TrajectorySampling(num_poses=4, interval_length=0.5)
    prediction = Trajectory(
        poses=np.column_stack(
            (np.arange(1, 5, dtype=np.float32), np.zeros(4), np.zeros(4))
        ),
        trajectory_sampling=sampling,
    )
    engine = MetricEngine.t4_default()
    report = engine.evaluate(
        MetricContext(token="scene@1", prediction=prediction, ground_truth=prediction),
        families=("open_loop",),
    )
    assert report.records[0].family == "open_loop"
    assert report.records[0].values["ade_m"] == 0.0
