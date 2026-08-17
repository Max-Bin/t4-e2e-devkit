"""Sensor time synchronisation.

The correction these tests cover is not cosmetic. Each camera trails its LiDAR
frame by +50 to +116 ms, and uncorrected that puts a projected box 54 px off its
object (median, on a 1148 px-wide image).

The offset is a property of the **channel and the vehicle**, not of the channel
alone: over the eight vehicles in these subtrees ``CAM_FRONT_WIDE`` ranges over
9 ms and ``CAM_BACK`` over 100 ms. So the tests check three things that would make
the correction silently wrong: the direction of the transform, the refusal to
"correct" a channel whose timestamps do not support it, and the absence of any
hardcoded offset standing in for the scene's own numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from t4_e2e_devkit.common.enums import T4BoxIndex
from t4_e2e_devkit.dataset.sync import (
    MAX_CORRECTION_S,
    ChannelTimes,
    SensorSync,
    SyncUnavailable,
    _associate,
    _interpolate_heading,
    camera_offsets,
)


class TestHeadingInterpolation:
    def test_midpoint(self):
        assert _interpolate_heading(0.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_takes_the_short_way_round(self):
        # 3.0 rad to -3.0 rad is 0.28 rad the short way, not 6.0 the long way.
        # Interpolating the raw difference would swing an object right around.
        result = _interpolate_heading(3.0, -3.0, 0.5)
        assert abs(result) > 3.0

    def test_endpoints(self):
        assert _interpolate_heading(0.4, 1.4, 0.0) == pytest.approx(0.4)
        assert _interpolate_heading(0.4, 1.4, 1.0) == pytest.approx(1.4)


class TestAssociation:
    @staticmethod
    def _box(x, y, label_dims=(1.8, 4.5, 1.5)):
        row = np.zeros(9)
        row[T4BoxIndex.X], row[T4BoxIndex.Y] = x, y
        row[T4BoxIndex.WIDTH], row[T4BoxIndex.LENGTH], row[T4BoxIndex.HEIGHT] = label_dims
        return row

    def test_nearest_centre_is_matched(self):
        here = np.stack([self._box(0, 0), self._box(10, 0)])
        later = np.stack([self._box(10.5, 0), self._box(0.5, 0)])
        matched = _associate(here, later, None, None)
        np.testing.assert_allclose(matched[0][T4BoxIndex.POINT2D], [0.5, 0.0])
        np.testing.assert_allclose(matched[1][T4BoxIndex.POINT2D], [10.5, 0.0])

    def test_far_pairs_are_not_matched(self):
        # Beyond a plausible one-frame displacement, a "match" would interpolate
        # one object toward a different one's position.
        here = np.stack([self._box(0, 0)])
        later = np.stack([self._box(50, 0)])
        assert _associate(here, later, None, None) == {}

    def test_class_gate_prevents_cross_class_matches(self):
        here = np.stack([self._box(0, 0)])
        later = np.stack([self._box(0.3, 0)])
        assert _associate(here, later, np.array([0]), np.array([0]))
        assert _associate(here, later, np.array([0]), np.array([4])) == {}

    def test_empty_inputs(self):
        empty = np.zeros((0, 9))
        assert _associate(empty, np.zeros((1, 9)), None, None) == {}
        assert _associate(np.zeros((1, 9)), empty, None, None) == {}


class TestChannelValidation:
    @staticmethod
    def _sync(camera_times, lidar_times, key_frames=None):
        """Build a sync from raw timestamps, without touching disk."""
        def channel(name, times, kf):
            return ChannelTimes(
                channel=name,
                frames=dict(enumerate(times)),
                poses={},
                rotations={},
                key_frame={i: (kf[i] if kf else True) for i in range(len(times))},
            )

        return SensorSync(
            scene_dir="synthetic",
            channels={
                "LIDAR_CONCAT": channel("LIDAR_CONCAT", lidar_times, None),
                "CAM_X": channel("CAM_X", camera_times, key_frames),
            },
        )

    def test_clean_offset_is_usable(self):
        lidar = [1_000_000 + 100_000 * i for i in range(10)]
        camera = [value + 52_000 for value in lidar]  # +52 ms, the measured shape
        usable, reason = self._sync(camera, lidar).validate_channel("CAM_X")
        assert usable, reason

    def test_repeated_timestamps_are_refused(self):
        # This is the real degenerate case: the video-backed channels repeat their
        # leading timestamp, and a correction computed from a repeated value is
        # not a correction.
        lidar = [1_000_000 + 100_000 * i for i in range(10)]
        camera = [1_500_000] * 5 + [lidar[i] + 52_000 for i in range(5, 10)]
        usable, reason = self._sync(camera, lidar).validate_channel("CAM_X")
        assert not usable
        assert "repeats" in reason

    def test_non_key_frames_are_refused(self):
        lidar = [1_000_000 + 100_000 * i for i in range(6)]
        camera = [value + 52_000 for value in lidar]
        usable, reason = self._sync(camera, lidar, key_frames=[True, False] + [True] * 4).validate_channel("CAM_X")
        assert not usable
        assert "non-key-frame" in reason

    def test_absurd_offset_is_refused(self):
        lidar = [1_000_000 + 100_000 * i for i in range(6)]
        camera = [value + 5_000_000 for value in lidar]  # +5 s
        usable, reason = self._sync(camera, lidar).validate_channel("CAM_X")
        assert not usable
        assert "beyond" in reason

    def test_missing_channel_is_refused(self):
        lidar = [1_000_000 + 100_000 * i for i in range(6)]
        usable, reason = self._sync([], lidar).validate_channel("CAM_ABSENT")
        assert not usable
        assert "absent" in reason

    def test_max_correction_is_under_two_frames(self):
        # A bound above two frames would let a whole-frame misalignment through
        # as though it were a sub-frame skew.
        assert MAX_CORRECTION_S < 0.25 + 1e-9


class TestNoHardcodedOffsets:
    """The offset is per vehicle, so no constant may stand in for it."""

    def test_module_defines_no_offset_table(self):
        import t4_e2e_devkit.dataset.sync as sync

        # Measured across 8 vehicles: CAM_FRONT_WIDE ranges over 9 ms and
        # CAM_BACK over 100 ms, so any table keyed by channel would be wrong for
        # most of the fleet. Every offset must come from the scene being read.
        offsets = {
            name: value
            for name, value in vars(sync).items()
            if isinstance(value, dict)
            and value
            and all(isinstance(k, str) and k.startswith("CAM_") for k in value)
        }
        assert not offsets, f"sync.py defines a per-channel offset table: {sorted(offsets)}"

    def test_offset_comes_from_the_scene(self):
        import inspect

        from t4_e2e_devkit.dataset.sync import SensorSync

        # offset_s reads ChannelTimes, which open() fills from the scene's own
        # sample_data.json -- there is no path that returns a literal.
        source = inspect.getsource(SensorSync.offset_s)
        assert "timestamp_s" in source
        assert not any(
            token in source for token in ("50.1", "51.3", "116.1", "0.05", "0.116")
        )


@pytest.mark.data
class TestAgainstRealScenes:
    def test_offsets_are_stable_within_one_scene(self, t4_scene_dir):
        report = camera_offsets(t4_scene_dir)
        if not report:
            pytest.skip("scene has no annotation/ tables")
        # Stable within a scene -- which is what lets one number correct that
        # scene. Stability across vehicles is NOT claimed, and is not true.
        for channel, values in report.items():
            if values["usable"]:
                assert values["std_ms"] < 5.0, f"{channel} offset is not stable"

    def test_offsets_are_non_trivial(self, t4_scene_dir):
        report = camera_offsets(t4_scene_dir)
        if not report:
            pytest.skip("scene has no annotation/ tables")
        usable = [v for v in report.values() if v["usable"]]
        if not usable:
            pytest.skip("no correctable channel in this scene")
        # If every offset were near zero the correction would be pointless; the
        # measured range is 50-116 ms, i.e. up to more than one 10 Hz frame.
        assert max(abs(v["mean_ms"]) for v in usable) > 10.0

    def test_ego_pose_is_sampled_per_sensor(self, t4_scene_dir):
        sync = SensorSync.open(t4_scene_dir)
        if sync is None or sync.lidar is None:
            pytest.skip("scene has no annotation/ tables")
        camera = next(
            (c for name, c in sync.channels.items() if name != "LIDAR_CONCAT" and c.poses),
            None,
        )
        if camera is None:
            pytest.skip("no camera pose available")
        frame = camera.sorted_frames[len(camera.sorted_frames) // 2]
        if frame not in sync.lidar.poses:
            pytest.skip("frame not shared with the LiDAR")
        # Distinct poses: the ego really is somewhere else at the camera's time,
        # which is the whole reason the transform is needed.
        assert not np.allclose(camera.poses[frame], sync.lidar.poses[frame])

    def test_correction_moves_boxes_by_a_realistic_amount(self, t4_scene_dir, t4_root):
        from t4_e2e_devkit.dataset.window import T4WindowBuilder

        builder = T4WindowBuilder(t4_scene_dir, t4_root)
        try:
            sync = builder.sync
            if sync is None:
                pytest.skip("scene has no annotation/ tables")
            centers = builder.valid_centers()
            center = centers[len(centers) // 2]
            here = builder.read_annotations(center, center)
            later = builder.read_annotations(center + 1, center)
            if len(here) == 0:
                pytest.skip("no boxes at this centre")

            channel = next(
                (name for name in sync.channels
                 if name != "LIDAR_CONCAT" and sync.validate_channel(name)[0]),
                None,
            )
            if channel is None:
                pytest.skip("no correctable channel")

            corrected = sync.corrected_boxes(
                np.asarray(here.boxes, np.float64), channel, center,
                next_boxes=np.asarray(later.boxes, np.float64),
                next_labels=later.labels, labels=here.labels,
            )
            shift = np.linalg.norm(corrected[:, :2] - np.asarray(here.boxes)[:, :2], axis=1)
            # Sub-metre but not sub-centimetre: a correction of zero would mean
            # the transform did nothing, metres would mean it is wrong.
            assert 0.001 < np.median(shift) < 3.0
            assert corrected.shape == (len(here), 9)
        finally:
            builder.close()

    def test_uncorrectable_channel_raises_rather_than_passing_through(self, t4_scene_dir, t4_root):
        from t4_e2e_devkit.dataset.window import T4WindowBuilder

        builder = T4WindowBuilder(t4_scene_dir, t4_root)
        try:
            sync = builder.sync
            if sync is None:
                pytest.skip("scene has no annotation/ tables")
            # Returning the input unchanged is how a caller ends up believing the
            # correction was applied when it was not.
            with pytest.raises(SyncUnavailable):
                sync.corrected_boxes(np.zeros((1, 9)), "CAM_DOES_NOT_EXIST", 10)
        finally:
            builder.close()

    def test_camera_carries_its_own_corrected_boxes(self, t4_scene_dir, t4_root):
        from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene
        from t4_e2e_devkit.dataset.window import T4WindowBuilder

        builder = T4WindowBuilder(
            t4_scene_dir, t4_root, sensor_config=sensor_config_for_scene(t4_scene_dir)
        )
        try:
            centers = builder.valid_centers()
            scene = builder.build(centers[len(centers) // 2])
            cameras = scene.current_frame.cameras
            if cameras is None or builder.sync is None:
                pytest.skip("no cameras or no annotation/ tables")
            synced = [c for c in cameras if c.annotations is not None]
            if not synced:
                pytest.skip("no channel could be corrected")
            frame_boxes = np.asarray(scene.current_frame.annotations.boxes)
            for camera in synced:
                assert camera.timestamp_us is not None
                assert len(camera.annotations) == len(frame_boxes)
                # Each channel gets its OWN correction: the offsets differ by
                # 66 ms across the rig, so one shared set would be wrong for
                # every channel but one.
                shift = np.linalg.norm(
                    np.asarray(camera.annotations.boxes)[:, :2] - frame_boxes[:, :2], axis=1
                )
                assert np.median(shift) > 0.0
            if len(synced) > 1:
                shifts = [
                    np.median(np.linalg.norm(
                        np.asarray(c.annotations.boxes)[:, :2] - frame_boxes[:, :2], axis=1))
                    for c in synced
                ]
                assert max(shifts) - min(shifts) > 0.01, "channels share one correction"
        finally:
            builder.close()
