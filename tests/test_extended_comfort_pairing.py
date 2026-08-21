"""Extended comfort has to be asked for, in the right unit, to exist.

v2 EPDMS weights extended comfort at 2 of 16, and it had never been produced:
the manifest scorer passed no previous plan, so ``_aggregate`` dropped the term
and renormalised over the remaining 14 weights instead of failing -- every report
read as a full EPDMS while being a partial-weight one.

The pairing is where the unit matters. A data-list centre indexes the 10 Hz
source stream; the manifest's ``interval_seconds`` is the spacing between plan
poses, 0.5 s for the standard eight-pose plan. Deriving the cycle from the latter
gives 1 instead of 5, which pairs nothing on any strided list and, where it does
fire, compares a plan against one made 0.1 s earlier while the metric shifts by
0.5 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.data

CYCLE_SECONDS = 0.5
SOURCE_HZ = 10.0


def _stride5_inputs(tmp_path: Path, t4_root: Path, first: int = 300, count: int = 8):
    """A stride-5 list over one scene, and a manifest that replays its future."""
    from t4_e2e_devkit.common.dataclasses import SensorConfig
    from t4_e2e_devkit.dataset.datalist import DataList
    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.evaluation.prediction_manifest import PredictionManifestWriter

    scene = next(
        (
            candidate
            for candidate in sorted(t4_root.glob("prd_jt*/*/*/*"))
            if (candidate / "derived" / "meta.json").is_file()
        ),
        None,
    )
    if scene is None:
        pytest.skip(f"no prd_jt scene under {t4_root}")
    scene_rel = str(scene.relative_to(t4_root))

    builder = T4WindowBuilder(scene, t4_root, sensor_config=SensorConfig(cameras={}, lidar=False))
    try:
        centers = [int(c) for c in builder.valid_centers()]
        step = int(round(CYCLE_SECONDS * SOURCE_HZ))
        rows = [(scene_rel, c) for c in centers if c >= first and (c - first) % step == 0][:count]
        if len(rows) < 3:
            pytest.skip("scene is too short for a stride-5 run")
        list_path = DataList(root=t4_root, rows=rows, manifest={"center_stride": step}).write(
            tmp_path / "stride5.datalist.json"
        )
        manifest_path = tmp_path / "predictions.jsonl"
        with PredictionManifestWriter(
            manifest_path, data_list=list_path, num_poses=8, interval_seconds=CYCLE_SECONDS
        ) as writer:
            for _, center in rows:
                poses = builder.build(center).get_future_trajectory().poses
                writer.write(scene_rel, center, np.asarray(poses, dtype=np.float64))
    finally:
        builder.close()
    return list_path, manifest_path, rows


def test_a_stride_five_run_produces_extended_comfort(tmp_path, t4_root):
    from t4_e2e_devkit.evaluation.prediction_scoring import score_prediction_manifest

    list_path, manifest_path, rows = _stride5_inputs(tmp_path, t4_root)
    report = score_prediction_manifest(
        data_list_path=list_path,
        predictions_path=manifest_path,
        output_dir=tmp_path / "score",
        version="v2",
        backend="cpu",
        batch_size=4,
        write_per_window=False,
    )

    # The cycle is a duration converted through the source rate, so five frames
    # -- not one, which is what the plan-pose interval would have given.
    assert report["extended_comfort_cycle_frames"] == int(round(CYCLE_SECONDS * SOURCE_HZ))
    # Every row but the first has a predecessor one cycle earlier.
    assert report["extended_comfort_pairs"] == len(rows) - 1
    # And the report names what it aggregated: the union across results, not the
    # first result's, which is the one window that cannot have a predecessor.
    assert "extended_comfort" in report["metric_names"]


def test_a_subsampled_run_reports_no_pairs_rather_than_a_full_epdms(tmp_path, t4_root):
    """A stride-50 list has no pairable row, and that is a fact worth reporting.

    The term is then genuinely unavailable -- which is not the same as never
    having been asked for, and a reader can tell the two apart from the count.
    """
    from t4_e2e_devkit.common.dataclasses import SensorConfig
    from t4_e2e_devkit.dataset.datalist import DataList
    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.evaluation.prediction_manifest import PredictionManifestWriter
    from t4_e2e_devkit.evaluation.prediction_scoring import score_prediction_manifest

    list_path, _, rows = _stride5_inputs(tmp_path, t4_root)
    scene_rel = rows[0][0]
    scene = t4_root / scene_rel
    sparse_rows = [(scene_rel, center) for _, center in rows[::4]]
    if len(sparse_rows) < 2:
        pytest.skip("scene is too short for a subsampled run")

    sparse_path = DataList(root=t4_root, rows=sparse_rows, manifest={"center_stride": 20}).write(
        tmp_path / "sparse.datalist.json"
    )
    builder = T4WindowBuilder(scene, t4_root, sensor_config=SensorConfig(cameras={}, lidar=False))
    try:
        manifest_path = tmp_path / "sparse.jsonl"
        with PredictionManifestWriter(
            manifest_path, data_list=sparse_path, num_poses=8, interval_seconds=CYCLE_SECONDS
        ) as writer:
            for _, center in sparse_rows:
                poses = builder.build(center).get_future_trajectory().poses
                writer.write(scene_rel, center, np.asarray(poses, dtype=np.float64))
    finally:
        builder.close()

    report = score_prediction_manifest(
        data_list_path=sparse_path,
        predictions_path=manifest_path,
        output_dir=tmp_path / "sparse-score",
        version="v2",
        backend="cpu",
        batch_size=4,
        write_per_window=False,
    )
    assert report["extended_comfort_pairs"] == 0
    assert "extended_comfort" not in report["metric_names"]
