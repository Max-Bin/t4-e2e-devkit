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


def _dict_scene(n_frames: int = 130) -> dict:
    """A plain-dict scene mapping, exercising the builder without handles."""

    trajectory = np.zeros((n_frames, 4), np.float32)
    trajectory[:, 0] = np.arange(n_frames, dtype=np.float32) * 0.5
    trajectory[:, 2] = 1.0
    scene = {
        "meta": {"n_frames": n_frames},
        "trajectory": trajectory,
        "velocity": np.zeros((n_frames, 2), np.float32),
        "turn": np.arange(n_frames, dtype=np.int32) % 3,
        "goal": np.array([500.0, 300.0, 1.0, 0.0], np.float32),
        "shape": np.array([2.79, 4.34, 1.7], np.float32),
        "points": {f"frame_{i:04d}": np.ones((4, 5), np.float32) for i in range(n_frames)},
    }
    rng = np.random.default_rng(1)
    for name, shape in expected_map_shapes().items():
        scene[name] = rng.standard_normal((n_frames, *shape)).astype(np.float32)
    return scene


def test_map_features_are_indexed_at_center():
    scene = _dict_scene()
    center = 40
    out = TrainingWindowBuilder().extract_window(scene, center)
    assert out is not None
    np.testing.assert_array_equal(out["lanes"], scene["lanes"][center])
    np.testing.assert_array_equal(out["route_lanes"], scene["route"][center])
    np.testing.assert_array_equal(out["polygons"], scene["polygons"][center])
    np.testing.assert_array_equal(out["line_strings"], scene["lines"][center])


def test_ego_frame_transform_is_vectorized_and_fixed_shape():
    scene = _dict_scene()
    out = TrainingWindowBuilder().extract_window(scene, 40)
    assert out is not None
    assert out["ego_agent_past"].shape == (PAST_FRAMES, 4)
    assert out["ego_agent_future"].shape == (FUTURE_FRAMES, 3)
    # Straight motion at 0.5 m/frame: the current frame is the origin and the
    # first future point is exactly 0.5 m ahead.
    np.testing.assert_allclose(out["ego_agent_past"][-1], [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(out["ego_agent_future"][0], [0.5, 0.0, 0.0], atol=1e-6)


def test_schema_is_strict_about_required_map_fields():
    scene = _dict_scene()
    del scene["route"]
    with pytest.raises(KeyError, match="route"):
        TrainingWindowBuilder().extract_window(scene, 40)


def test_goal_clamp_preserves_bearing():
    scene = _dict_scene()
    raw = TrainingWindowBuilder().extract_window(scene, 40)
    clamped = TrainingWindowBuilder(goal_clamp_m=120.0).extract_window(scene, 40)
    assert raw is not None and clamped is not None
    assert np.isclose(np.hypot(*clamped["goal_pose"][:2]), 120.0, atol=1e-5)
    assert np.isclose(
        np.arctan2(*clamped["goal_pose"][1::-1]),
        np.arctan2(*raw["goal_pose"][1::-1]),
    )


def test_valid_window_centers_edge_mask_trims_history_or_future():
    leading = np.ones(N_FRAMES, bool)
    leading[:5] = False
    assert list(valid_window_centers(N_FRAMES, leading))[0] == PAST_FRAMES - 1 + 5
    trailing = np.ones(N_FRAMES, bool)
    trailing[-5:] = False
    last_valid = N_FRAMES - 6
    centers = list(valid_window_centers(N_FRAMES, trailing))
    assert centers[-1] == last_valid - FUTURE_FRAMES


def test_valid_window_centers_wrong_mask_length_fails_loudly():
    with pytest.raises(ValueError, match="expected"):
        valid_window_centers(N_FRAMES, np.ones(N_FRAMES - 1, bool))


def test_valid_window_centers_all_false_has_no_rows():
    assert list(valid_window_centers(N_FRAMES, np.zeros(N_FRAMES, bool))) == []


def test_pack_read_timeout_watchdog_path_returns_identical_frames(tmp_path, monkeypatch):
    frames = [np.arange(15, dtype=np.float32).reshape(3, 5) for _ in range(2)]
    path = tmp_path / "LIDAR_CONCAT.pack"
    write_lidar_pack(path, frames)
    reader = T4LidarPackReader(path)
    direct = reader.read_frame(0)
    monkeypatch.setenv("T4E2E_PACK_READ_TIMEOUT", "30")
    watched = reader.read_frame(0)
    np.testing.assert_array_equal(direct, watched)
    reader.close()


# --------------------------------------------------------------------------- #
# TemporalSpec
# --------------------------------------------------------------------------- #

from t4_e2e_devkit.common.temporal import DEFAULT_TEMPORAL_SPEC, TemporalSpec  # noqa: E402


def test_default_spec_reproduces_the_historical_contract():
    spec = DEFAULT_TEMPORAL_SPEC
    assert spec.past_frames == PAST_FRAMES
    assert spec.future_frames == FUTURE_FRAMES
    assert spec.frame_stride == 1
    assert spec.min_source_frames == MIN_T4_FRAMES
    assert spec.interval_seconds == pytest.approx(0.1)


def test_spec_validation():
    with pytest.raises(ValueError, match="divisor"):
        TemporalSpec(hz=3)
    with pytest.raises(ValueError, match="divisor"):
        TemporalSpec(hz=20)
    with pytest.raises(ValueError, match="grid"):
        TemporalSpec(future_seconds=4.3, hz=2)  # 8.6 frames
    with pytest.raises(ValueError, match="positive"):
        TemporalSpec(future_seconds=0.0)
    spec = TemporalSpec(history_seconds=2.0, future_seconds=4.0, hz=5)
    assert spec.past_frames == 11
    assert spec.future_frames == 20
    assert spec.frame_stride == 2
    assert spec.history_span == 20
    assert spec.future_span == 40
    assert spec.min_source_frames == 61
    assert TemporalSpec.from_dict(spec.as_dict()) == spec


def test_strided_window_is_the_stride_view_of_the_dense_window():
    """A 5 Hz window must be EXACTLY every second sample of the 10 Hz window
    with the same seconds spans — stride sampling, zero interpolation."""

    scene = _dict_scene(n_frames=200)
    center = 110
    dense = TrainingWindowBuilder().extract_window(scene, center)
    spec5 = TemporalSpec(history_seconds=3.0, future_seconds=8.0, hz=5)
    strided = TrainingWindowBuilder(spec=spec5).extract_window(scene, center)
    assert dense is not None and strided is not None

    assert strided["ego_agent_past"].shape == (16, 4)  # 3 s @ 5 Hz + current
    assert strided["ego_agent_future"].shape == (40, 3)  # 8 s @ 5 Hz
    # History: current frame anchors the grid, so the 5 Hz history is the
    # dense history sampled backward from the END.
    np.testing.assert_array_equal(strided["ego_agent_past"], dense["ego_agent_past"][::2])
    # Future: pose i sits at (i+1)/hz seconds, so the 5 Hz future is every
    # second dense pose STARTING at the second one.
    np.testing.assert_array_equal(strided["ego_agent_future"], dense["ego_agent_future"][1::2])
    np.testing.assert_array_equal(strided["turn_indicators"], dense["turn_indicators"][::2])
    # Centre-frame payloads are rate-independent.
    np.testing.assert_array_equal(strided["lanes"], dense["lanes"])
    np.testing.assert_array_equal(strided["goal_pose"], dense["goal_pose"])


def test_valid_window_centers_follow_the_spec_spans():
    spec = TemporalSpec(history_seconds=2.0, future_seconds=4.0, hz=5)
    n = spec.min_source_frames + 10
    centers = valid_window_centers(n, spec=spec)
    assert centers.start == spec.history_span
    assert max(centers) == n - 1 - spec.future_span
    assert valid_window_centers(spec.min_source_frames - 1, spec=spec) == ()


def test_short_horizon_spec_admits_short_scenes():
    """A 4 s @ 5 Hz spec must accept scenes the default 8 s contract rejects."""

    spec = TemporalSpec(history_seconds=2.0, future_seconds=4.0, hz=5)
    n = spec.min_source_frames  # 61 << MIN_T4_FRAMES (111)
    assert valid_window_centers(n) == ()
    assert list(valid_window_centers(n, spec=spec)) == [spec.history_span]


def test_handles_build_without_a_lidar_pack(tmp_path):
    """A camera/map-only copy has no pack, and must still produce handles.

    ``TrainingWindowBuilder(load_points=False)`` is the documented path for a
    camera or map backbone, but the handles used to open the pack in their
    constructor, so a trimmed scene failed before the builder ever saw the flag.
    """
    root = tmp_path / "t4-root"
    scene_dir = _make_scene(root)
    (scene_dir / "data" / "LIDAR_CONCAT.pack").unlink()

    handles = TrainingSceneHandles(scene_dir, load_gt=False, t4_root=root)
    try:
        scene = handles.scene_dict("scene@0")
        window = TrainingWindowBuilder(load_points=False).extract_window(scene, PAST_FRAMES - 1)
        assert window is not None
        assert window["points"].shape == (0, 5)
        # Asking for the sweep still fails, at the read and by name.
        with pytest.raises(FileNotFoundError, match="LIDAR_CONCAT.pack"):
            TrainingWindowBuilder(load_points=True).extract_window(scene, PAST_FRAMES - 1)
    finally:
        handles.close()


def test_a_scene_declaring_no_pack_has_no_sweeps(tmp_path):
    # An export that carries no LiDAR at all is a fact about the export; the
    # handles report empty sweeps rather than raising on a missing meta key.
    root = tmp_path / "t4-root"
    scene_dir = _make_scene(root)
    meta_path = scene_dir / "derived" / "meta.json"
    meta = json.loads(meta_path.read_text())
    for key in ("lidar_pack", "lidar_frames"):
        meta.pop(key, None)
    meta_path.write_text(json.dumps(meta))

    handles = TrainingSceneHandles(scene_dir, load_gt=False, t4_root=root)
    try:
        window = TrainingWindowBuilder().extract_window(
            handles.scene_dict("scene@0"), PAST_FRAMES - 1
        )
        assert window is not None and window["points"].shape == (0, 5)
    finally:
        handles.close()


def test_a_short_pack_is_still_rejected(tmp_path):
    # The range check moved to the first read; it must still happen.
    root = tmp_path / "t4-root"
    scene_dir = _make_scene(root)
    write_lidar_pack(scene_dir / "data" / "LIDAR_CONCAT.pack", [np.zeros((3, 5), np.float32)])

    handles = TrainingSceneHandles(scene_dir, load_gt=False, t4_root=root)
    try:
        with pytest.raises(ValueError, match="outside pack range"):
            TrainingWindowBuilder().extract_window(handles.scene_dict("scene@0"), PAST_FRAMES - 1)
    finally:
        handles.close()
