"""Sharing an open scene between the training handles and the scene builder.

A training loader holds ``TrainingSceneHandles`` (meta, scalars, bundle) for
the scene it is reading; a PDM oracle needs the rich ``T4Scene`` of the same
scene. Adoption lets the scene builder borrow that state instead of opening
``frames.pack``/``scalars.npz``/``meta.json`` a second time. These tests pin
the two invariants adoption must keep: the assembled scene is IDENTICAL to
the standalone build, and a borrowed bundle survives the builder's close.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

zstd = pytest.importorskip("zstandard")

from t4_e2e_devkit.common.dataclasses import SceneFilter, SensorConfig  # noqa: E402
from t4_e2e_devkit.dataset.pack_writers import write_bundle, write_lidar_pack  # noqa: E402
from t4_e2e_devkit.dataset.training_window import (  # noqa: E402
    TrainingSceneHandles,
    expected_map_shapes,
)
from t4_e2e_devkit.dataset.window import T4WindowBuilder  # noqa: E402

N_FRAMES = 60
HISTORY = 5
FUTURE = 10


def _make_scene(root: Path) -> Path:
    """A byte-valid scene WITH camera metadata (the rich reader requires it)."""

    rng = np.random.default_rng(11)
    relative = Path("prd_jt/session/2026-01-01/scene-000")
    scene = root / relative
    (scene / "data").mkdir(parents=True)
    (scene / "derived").mkdir()

    write_lidar_pack(
        scene / "data" / "LIDAR_CONCAT.pack",
        [rng.uniform(-30, 30, (8, 5)).astype(np.float32) for _ in range(N_FRAMES)],
    )

    fields = {
        name: rng.standard_normal((N_FRAMES, *shape)).astype(np.float32)
        for name, shape in expected_map_shapes().items()
    }
    for name in ("lanes_has_speed", "route_has_speed"):
        fields[name] = fields[name] > 0
    boxes = [
        np.array([[10.0 - f * 0.3, 2.0, 0.5, 1.8, 4.2, 1.5, 0.0, 0.0, 0.0]], np.float32)
        for f in range(N_FRAMES)
    ]
    labels = [np.array([0], np.int64) for _ in range(N_FRAMES)]
    write_bundle(
        scene / "derived" / "frames.pack",
        N_FRAMES,
        dict(fields, gt_boxes=boxes, gt_labels=labels),
    )

    trajectory = np.zeros((N_FRAMES, 4), np.float32)
    trajectory[:, 0] = np.arange(N_FRAMES, dtype=np.float32) * 0.5
    trajectory[:, 2] = 1.0
    camera_names = ["CAM_FRONT_WIDE"]
    np.savez(
        scene / "derived" / "scalars.npz",
        trajectory=trajectory,
        velocity=np.full((N_FRAMES, 2), 0.5, np.float32),
        turn=np.zeros(N_FRAMES, np.int32),
        goal=np.array([40.0, 1.0, 1.0, 0.0], np.float32),
        shape=np.array([2.75, 4.8, 1.9], np.float32),
        cam_intrinsics=np.eye(3, dtype=np.float64)[None].repeat(len(camera_names), 0),
        cam_extrinsics=np.eye(4, dtype=np.float64)[None].repeat(len(camera_names), 0),
    )
    (scene / "derived" / "cam_names.json").write_text(json.dumps(camera_names))
    (scene / "derived" / "meta.json").write_text(
        json.dumps(
            {
                "n_frames": N_FRAMES,
                "lidar_pack": (relative / "data/LIDAR_CONCAT.pack").as_posix(),
                "lidar_frames": N_FRAMES,
                "lidar_first_frame": 0,
                "frame_offset": 0,
            }
        )
    )
    return scene


def _builder(scene: Path, root: Path, handles: TrainingSceneHandles | None) -> T4WindowBuilder:
    shared = {}
    if handles is not None:
        shared = dict(
            shared_bundle=handles.bundle,
            shared_meta=handles.meta,
            shared_scalars=handles.scalars,
        )
    return T4WindowBuilder(
        scene,
        root,
        sensor_config=SensorConfig.build_no_sensors(),
        scene_filter=SceneFilter(num_history_frames=HISTORY, num_future_frames=FUTURE),
        **shared,
    )


def _assert_same_scene(a, b) -> None:
    fa, fb = a.current_frame, b.current_frame
    np.testing.assert_array_equal(fa.map_tensors.lanes, fb.map_tensors.lanes)
    np.testing.assert_array_equal(fa.map_tensors.route_lanes, fb.map_tensors.route_lanes)
    np.testing.assert_array_equal(fa.map_tensors.polygons, fb.map_tensors.polygons)
    assert fa.ego_status.speed == fb.ego_status.speed
    aa, ab = a.future_annotations, b.future_annotations
    assert (aa is None) == (ab is None)
    if aa is not None:
        assert len(aa) == len(ab)
        for x, y in zip(aa, ab, strict=True):
            np.testing.assert_array_equal(x.boxes, y.boxes)
            np.testing.assert_array_equal(x.labels, y.labels)


def test_adopted_build_is_identical_to_standalone(tmp_path):
    root = tmp_path / "t4-root"
    scene = _make_scene(root)
    handles = TrainingSceneHandles(scene, load_gt=False, t4_root=root)

    standalone = _builder(scene, root, None)
    adopted = _builder(scene, root, handles)
    for center in (HISTORY, N_FRAMES // 2, N_FRAMES - FUTURE - 1):
        _assert_same_scene(standalone.build(center), adopted.build(center))
    standalone.close()
    adopted.close()
    handles.close()


def test_borrowed_bundle_survives_the_builders_close(tmp_path):
    root = tmp_path / "t4-root"
    scene = _make_scene(root)
    handles = TrainingSceneHandles(scene, load_gt=False, t4_root=root)

    adopted = _builder(scene, root, handles)
    adopted.build(N_FRAMES // 2)
    adopted.close()
    # The training side keeps reading through the same bundle afterwards.
    frame = handles.bundle.frame(3)
    assert "lanes" in frame
    handles.close()
    # A standalone reader owns its bundle and does close it: reopening after
    # close must be the caller's job, which double-close would mask.
    standalone = _builder(scene, root, None)
    standalone.build(N_FRAMES // 2)
    standalone.close()
