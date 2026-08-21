"""What a data-list build accepts, refuses, and records about cameras.

The camera gate is the only place a rig mismatch can be caught cheaply. Once a
list is written, a register its scenes cannot serve is indistinguishable from a
register they can -- until the first camera batch of a training run fails in a
DataLoader worker. So these tests check the gate against real scenes of both
rigs rather than the argument parsing around it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from t4_e2e_devkit.script import build_datalist


def _build(root: Path, scene_dir: Path, cameras: list[str], *extra: str):
    """Build a one-scene list, without writing it."""
    relative = Path(scene_dir).relative_to(root).as_posix()
    argv = [
        "--root",
        str(root),
        "--glob",
        relative,
        "--out",
        str(Path(root) / "unused.json"),
        "--camera-names",
        *cameras,
        "--limit-per-scene",
        "2",
        *extra,
    ]
    return build_datalist.build(build_datalist.parse_args(argv))


class TestRequiredArguments:
    """Neither the subtree nor the register may be inherited from a default."""

    def test_glob_is_required(self):
        with pytest.raises(SystemExit):
            build_datalist.parse_args(
                ["--root", "/nonexistent", "--out", "out.json", "--camera-names", "wide5"]
            )

    def test_camera_names_are_required(self):
        with pytest.raises(SystemExit):
            build_datalist.parse_args(
                ["--root", "/nonexistent", "--glob", "prd_jt/*/*/*", "--out", "out.json"]
            )


class TestPresenceGate:
    """The per-window half of the gate, on a stub reader.

    Real scenes rarely drop a surround frame, so the row-level rule needs a
    constructed case: without one, "no rows dropped" reads the same whether the
    gate works or does nothing at all -- which is exactly how it went unnoticed.
    """

    @staticmethod
    def _builder(presence):
        class _Reader:
            scalars = {"cam_presence": presence}
            # Register order is not the on-disk column order; the gate has to
            # index presence through camera_indices, not through the register.
            camera_names = ["CAM_FRONT", "CAM_BACK"]
            camera_indices = [3, 0]

        class _Builder:
            reader = _Reader()

            def valid_centers(self):
                return [40, 45]

        return _Builder()

    @staticmethod
    def _args():
        return argparse.Namespace(
            limit_per_scene=None,
            history_frames=31,
            future_frames=80,
            max_window_gap_frames=None,
        )

    def test_a_window_missing_a_required_frame_is_dropped(self):
        presence = np.ones((200, 4), dtype=bool)
        presence[40, 0] = False  # CAM_BACK at the first centre
        dropped = {"rows_dropped_by_camera": 0, "rows_dropped_by_window_gap": 0}
        rows = build_datalist._scene_rows(
            self._builder(presence),
            "x2_dev/scene",
            ["CAM_FRONT", "CAM_BACK"],
            self._args(),
            dropped,
        )
        assert rows == [("x2_dev/scene", 45)]
        assert dropped["rows_dropped_by_camera"] == 1

    def test_a_camera_nobody_required_does_not_drop_the_window(self):
        presence = np.ones((200, 4), dtype=bool)
        presence[40, 0] = False  # CAM_BACK absent, but only CAM_FRONT required
        dropped = {"rows_dropped_by_camera": 0, "rows_dropped_by_window_gap": 0}
        rows = build_datalist._scene_rows(
            self._builder(presence), "x2_dev/scene", ["CAM_FRONT"], self._args(), dropped
        )
        assert len(rows) == 2
        assert dropped["rows_dropped_by_camera"] == 0


@pytest.mark.data
class TestCameraGate:
    def test_a_fitting_profile_keeps_rows_and_records_the_register(self, rig_scene_dir, t4_root):
        from t4_e2e_devkit.common.constants import T4_CAMERA_PROFILES
        from t4_e2e_devkit.dataset.rigs import matching_profiles, readable_camera_names

        profiles = matching_profiles(readable_camera_names(rig_scene_dir))
        if not profiles:
            pytest.skip(f"{rig_scene_dir} fits no named profile")
        data_list = _build(t4_root, rig_scene_dir, [profiles[0]])
        assert data_list.rows
        # The manifest records what the request resolved to, not the request: a
        # list that says "wide5" tells a later reader nothing about which
        # channels its rows can actually serve.
        assert data_list.manifest["camera_registers"] == [
            {"register": list(T4_CAMERA_PROFILES[profiles[0]]), "scenes": 1}
        ]

    def test_the_wrong_rigs_profile_drops_the_scene(self, x2_scene_dir, t4_root):
        # wide5 cannot resolve on x2_dev. Before the register was resolved during
        # the build, this produced a list that claimed wide5 and still carried
        # every row -- the failure only surfaced at the first camera batch.
        data_list = _build(t4_root, x2_scene_dir, ["wide5"])
        assert data_list.rows == []
        assert data_list.manifest["filter"]["scene_without_cameras"] == 1

    def test_x2_dev_builds_with_its_own_profile(self, x2_scene_dir, t4_root):
        data_list = _build(t4_root, x2_scene_dir, ["x2_surround6"])
        assert data_list.rows
        registers = data_list.manifest["camera_registers"]
        assert registers[0]["register"] == [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

    def test_requiring_a_camera_the_rig_lacks_drops_the_scene(self, t4_scene_dir, t4_root):
        data_list = _build(t4_root, t4_scene_dir, ["wide5"], "--require-cameras", "CAM_BACK_WIDE")
        assert data_list.rows == []
        assert data_list.manifest["filter"]["scene_without_cameras"] == 1

    def test_no_camera_contract_still_builds(self, rig_scene_dir, t4_root):
        # A map/ego/LiDAR list declares no register; it must not inherit one.
        data_list = _build(t4_root, rig_scene_dir, ["none"])
        assert data_list.rows
        assert data_list.manifest["camera_registers"] == [{"register": [], "scenes": 1}]
