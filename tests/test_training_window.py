"""Flat training-window assembly: writers round-trip, handles validation,
window numerics.

The strong bit-parity gate for the window transforms lives in the training
repository (golden windows captured from the pre-unification reader); these
tests pin the format round-trip and the assembly semantics on synthetic
scenes so they run without the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

zstd = pytest.importorskip("zstandard")

from t4_e2e_devkit.common.constants import FUTURE_FRAMES, MIN_T4_FRAMES, PAST_FRAMES  # noqa: E402
from t4_e2e_devkit.dataset.pack_writers import write_bundle, write_lidar_pack  # noqa: E402
from t4_e2e_devkit.dataset.scene import T4BundleReader, T4LidarPackReader  # noqa: E402
from t4_e2e_devkit.dataset.training_window import (  # noqa: E402
    TrainingSceneHandles,
    TrainingWindowBuilder,
    expected_map_shapes,
    valid_window_centers,
)

N_FRAMES = MIN_T4_FRAMES + 8


def _make_scene(root: Path, *, with_gt: bool = True) -> Path:
    """Fabricate one byte-valid T4 scene under ``root``."""

    rng = np.random.default_rng(7)
    relative = Path("prd_jt/session/2026-01-01/scene-000")
    scene = root / relative
    (scene / "data").mkdir(parents=True)
    (scene / "derived").mkdir()

    lidar = []
    for _ in range(N_FRAMES):
        points = np.empty((12, 5), np.float32)
        points[:, :3] = rng.uniform(-30, 30, (12, 3)).astype(np.float32)
        points[:, 3] = rng.integers(0, 255, 12)
        points[:, 4] = rng.integers(0, 8, 12)
        lidar.append(points)
    write_lidar_pack(scene / "data" / "LIDAR_CONCAT.pack", lidar)

    fields = {
        name: rng.standard_normal((N_FRAMES, *shape)).astype(np.float32)
        for name, shape in expected_map_shapes().items()
    }
    for name in ("lanes_has_speed", "route_has_speed"):
        fields[name] = fields[name] > 0

    if with_gt:
        boxes, labels = [], []
        for frame in range(N_FRAMES):
            if frame % 17 == 0:
                boxes.append(np.zeros((0, 9), np.float32))
                labels.append(np.zeros((0,), np.int64))
            else:
                boxes.append(
                    np.array([[18.0 - frame * 0.5, 4.0, 0.5, 1.8, 4.2, 1.5, 0, 0, 0]], np.float32)
                )
                labels.append(np.array([0], np.int64))
        fields = dict(fields, gt_boxes=boxes, gt_labels=labels)
    write_bundle(scene / "derived" / "frames.pack", N_FRAMES, fields)

    trajectory = np.zeros((N_FRAMES, 4), np.float32)
    trajectory[:, 0] = np.arange(N_FRAMES, dtype=np.float32) * 0.5
    trajectory[:, 2] = 1.0
    np.savez(
        scene / "derived" / "scalars.npz",
        trajectory=trajectory,
        velocity=np.full((N_FRAMES, 2), 0.5, np.float32),
        turn=np.arange(N_FRAMES, dtype=np.int32) % 3,
        goal=np.array([180.0, 3.0, 1.0, 0.0], np.float32),
        shape=np.array([2.75, 4.8, 1.9], np.float32),
    )
    meta = {
        "n_frames": N_FRAMES,
        "lidar_pack": (relative / "data/LIDAR_CONCAT.pack").as_posix(),
        "lidar_frames": N_FRAMES,
        "lidar_first_frame": 0,
        "frame_offset": 0,
    }
    (scene / "derived" / "meta.json").write_text(json.dumps(meta))
    return scene


def test_lidar_pack_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    frames = [
        np.column_stack(
            [
                rng.uniform(-50, 50, (9, 3)).astype(np.float32),
                rng.integers(0, 255, (9, 1)).astype(np.float32),
                rng.integers(-4, 8, (9, 1)).astype(np.float32),
            ]
        )
        for _ in range(4)
    ]
    path = tmp_path / "LIDAR_CONCAT.pack"
    write_lidar_pack(path, frames)
    reader = T4LidarPackReader(path)
    assert reader.n_frames == 4
    for index, frame in enumerate(frames):
        got = reader.read_frame(index)
        # intensity/ring pass through their integer storage types
        expected = frame.copy()
        expected[:, 3] = frame[:, 3].astype(np.uint8)
        expected[:, 4] = frame[:, 4].astype(np.int8)
        np.testing.assert_array_equal(got, expected.astype(np.float32))
    reader.close()


def test_bundle_round_trip_fixed_and_variable(tmp_path):
    n = 5
    fixed = [np.full((3, 2), i, np.float32) for i in range(n)]
    ragged = [np.arange(i, dtype=np.int64).reshape(-1) for i in range(n)]
    path = tmp_path / "frames.pack"
    write_bundle(path, n, {"fixed": fixed, "ragged": ragged})
    reader = T4BundleReader(path)
    assert reader.n_frames == n
    assert reader.field_spec["ragged"]["variable"] is True
    for i in range(n):
        frame = reader.frame(i)
        np.testing.assert_array_equal(frame["fixed"], fixed[i])
        np.testing.assert_array_equal(frame["ragged"], ragged[i])
    reader.close()


def test_valid_window_centers_bounds_and_stride():
    centers = valid_window_centers(N_FRAMES)
    assert centers.start == PAST_FRAMES - 1
    assert centers.stop - 1 <= N_FRAMES - 1 - FUTURE_FRAMES
    strided = valid_window_centers(N_FRAMES, stride=5)
    assert list(strided) == list(centers)[::5]
    assert valid_window_centers(MIN_T4_FRAMES - 1) == ()


def test_valid_window_centers_rejects_interior_hole():
    mask = np.ones(N_FRAMES, bool)
    mask[N_FRAMES // 2] = False
    with pytest.raises(ValueError, match="interior invalid"):
        valid_window_centers(N_FRAMES, valid_mask=mask)


def test_training_window_assembly(tmp_path):
    root = tmp_path / "t4-root"
    scene_dir = _make_scene(root)
    handles = TrainingSceneHandles(scene_dir, load_gt=True, t4_root=root)
    scene = handles.scene_dict("prd_jt/session/2026-01-01/scene-000")
    builder = TrainingWindowBuilder(goal_clamp_m=120.0)

    center = PAST_FRAMES - 1
    sample = builder.extract_window(scene, center)
    assert sample is not None
    assert sample["ego_agent_past"].shape == (PAST_FRAMES, 4)
    assert sample["ego_agent_future"].shape == (FUTURE_FRAMES, 3)
    assert sample["ego_current_state"].shape == (10,)
    assert sample["turn_indicators"].shape == (PAST_FRAMES,)
    assert sample["goal_pose"].shape == (4,)
    assert sample["points"].shape[1] == 5
    # The ego moves along +x at 0.5 m/frame with identity heading, so the
    # future is a straight x ramp in the centre frame.
    np.testing.assert_allclose(
        sample["ego_agent_future"][:, 0],
        np.arange(1, FUTURE_FRAMES + 1, dtype=np.float32) * 0.5,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(sample["ego_agent_future"][:, 1], 0.0, atol=1e-5)
    # Ring column is zeroed on the loaded sweep.
    assert (sample["points"][:, 4] == 0.0).all()
    # GT passes through at the centre frame.
    assert sample["gt_bboxes_3d"].shape[1] == 9
    # A window that does not fit returns None rather than raising.
    assert builder.extract_window(scene, 0) is None
    handles.close()


def test_goal_clamp_scales_to_radius(tmp_path):
    root = tmp_path / "t4-root"
    scene_dir = _make_scene(root)
    handles = TrainingSceneHandles(scene_dir, load_gt=False, t4_root=root)
    scene = handles.scene_dict("scene")
    clamped = TrainingWindowBuilder(goal_clamp_m=50.0).extract_window(scene, PAST_FRAMES - 1)
    free = TrainingWindowBuilder(goal_clamp_m=None).extract_window(scene, PAST_FRAMES - 1)
    assert clamped is not None and free is not None
    assert np.hypot(*clamped["goal_pose"][:2]) == pytest.approx(50.0, abs=1e-4)
    assert np.hypot(*free["goal_pose"][:2]) > 50.0
    handles.close()


def test_point_range_crops_the_sweep(tmp_path):
    root = tmp_path / "t4-root"
    scene_dir = _make_scene(root)
    handles = TrainingSceneHandles(scene_dir, load_gt=False, t4_root=root)
    scene = handles.scene_dict("scene")
    bounds = (-10.0, -10.0, -10.0, 10.0, 10.0, 10.0)
    cropped = TrainingWindowBuilder(point_range=bounds).extract_window(scene, PAST_FRAMES - 1)
    full = TrainingWindowBuilder().extract_window(scene, PAST_FRAMES - 1)
    assert cropped is not None and full is not None
    assert cropped["points"].shape[0] <= full["points"].shape[0]
    if cropped["points"].shape[0]:
        assert (np.abs(cropped["points"][:, :3]) < 10.0).all()
    handles.close()


def test_handles_reject_frame_count_mismatch(tmp_path):
    root = tmp_path / "t4-root"
    scene_dir = _make_scene(root)
    scalars = dict(np.load(scene_dir / "derived" / "scalars.npz"))
    scalars["trajectory"] = scalars["trajectory"][:-1]
    np.savez(scene_dir / "derived" / "scalars.npz", **scalars)
    with pytest.raises(ValueError, match="frames, expected"):
        TrainingSceneHandles(scene_dir, load_gt=False, t4_root=root)


def test_lidar_reader_rejects_corrupt_index_entry(tmp_path):
    path = tmp_path / "LIDAR_CONCAT.pack"
    write_lidar_pack(path, [np.zeros((3, 5), np.float32)])
    raw = bytearray(path.read_bytes())
    # Rewrite the index with an out-of-bounds frame size.
    reader_ok = T4LidarPackReader(path)
    frames = [dict(reader_ok.frames[0])]
    reader_ok.close()
    frames[0]["size"] = 1 << 40
    index_blob = zstd.ZstdCompressor(level=1).compress(
        json.dumps(
            {"format": "t4pack", "version": 1, "n_frames": 1, "frames": frames},
            separators=(",", ":"),
        ).encode()
    )
    import struct

    magic = b"T4PACK\x00\x01"
    body_end = len(raw) - (len(magic) + 16)
    offset, _ = struct.unpack("<QQ", raw[body_end : body_end + 16])
    new = raw[:offset] + index_blob + struct.pack("<QQ", offset, len(index_blob)) + magic
    path.write_bytes(new)
    with pytest.raises(ValueError, match="invalid bounds"):
        T4LidarPackReader(path)
