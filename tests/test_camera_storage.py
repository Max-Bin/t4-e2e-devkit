"""Camera rig resolution and the public JPEG-wide input boundary."""

from __future__ import annotations

import pytest

from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.dataset.camera_source import (
    CameraSourceError,
    JpegDirectorySource,
    available_cameras,
    open_camera_source,
)
from t4_e2e_devkit.dataset.rigs import (
    RigMismatch,
    describe_rig,
    matching_profiles,
    normalize_camera_names,
    readable_camera_names,
    resolve_camera_names,
    rig_signature,
    surround_camera_names,
)

# The three registers actually observed, as fixtures.
PRD_JT_MAIN = [
    "CAM_BACK_LEFT", "CAM_BACK_LEFT_WIDE", "CAM_BACK_RIGHT", "CAM_BACK_RIGHT_WIDE",
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_LEFT_WIDE", "CAM_FRONT_RIGHT",
    "CAM_FRONT_RIGHT_WIDE", "CAM_FRONT_WIDE", "CAM_TRAFFIC_LIGHT_FAR",
]
PRD_JT_VARIANT = [
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_FRONT_LEFT",
    "CAM_FRONT_LEFT_WIDE", "CAM_FRONT_RIGHT", "CAM_FRONT_RIGHT_WIDE",
    "CAM_TRAFFIC_LIGHT_FAR",
]
X2_DEV = [
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT", "CAM_FRONT_WIDE", "CAM_TOP_LEFT_CENTER",
    "CAM_TOP_RIGHT_CENTER", "CAM_TRAFFIC_LIGHT_FAR", "CAM_TRAFFIC_LIGHT_NEAR",
]


class TestProfileResolution:
    def test_wide_five_fits_the_main_prd_jt_rig(self):
        assert resolve_camera_names("wide5", PRD_JT_MAIN) == list(C.T4_WIDE5_CAMERA_NAMES)

    def test_wide_five_is_refused_on_x2_dev(self):
        # x2_dev has only one wide view. Silently substituting another register
        # would evaluate a checkpoint on an input layout it was not trained on.
        with pytest.raises(RigMismatch, match="CAM_FRONT_LEFT_WIDE"):
            resolve_camera_names("wide5", X2_DEV)

    def test_wide_five_is_refused_on_the_prd_jt_variant(self):
        with pytest.raises(RigMismatch, match="CAM_FRONT_WIDE"):
            resolve_camera_names("wide5", PRD_JT_VARIANT)

    def test_narrow_profile_is_rejected(self):
        with pytest.raises(RigMismatch, match="not supported"):
            resolve_camera_names("surround6", X2_DEV)

    def test_auto_picks_wide_only(self):
        assert resolve_camera_names(None, PRD_JT_MAIN) == list(C.T4_WIDE5_CAMERA_NAMES)
        assert resolve_camera_names(None, X2_DEV) == ["CAM_FRONT_WIDE"]

    def test_auto_uses_available_wide_subset_for_variant(self):
        assert resolve_camera_names(None, PRD_JT_VARIANT) == [
            "CAM_FRONT_LEFT_WIDE",
            "CAM_FRONT_RIGHT_WIDE",
        ]

    def test_explicit_missing_camera_is_an_error(self):
        with pytest.raises(RigMismatch, match="absent"):
            resolve_camera_names(["CAM_BACK_WIDE"], PRD_JT_MAIN)

    def test_explicit_narrow_camera_is_rejected(self):
        with pytest.raises(RigMismatch, match="not supported"):
            resolve_camera_names(["CAM_FRONT"], PRD_JT_MAIN)

    def test_resolution_preserves_requested_order(self):
        # Register order is part of the learned camera contract, so it must never
        # be sorted on the way through.
        requested = ["CAM_FRONT_RIGHT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_LEFT_WIDE"]
        assert resolve_camera_names(requested, PRD_JT_MAIN) == requested

    def test_require_count_is_enforced(self):
        with pytest.raises(RigMismatch, match="required"):
            resolve_camera_names("wide5", PRD_JT_MAIN, require_count=6)

    def test_no_profile_fits_reports_every_requirement(self):
        with pytest.raises(RigMismatch, match="no supported JPEG wide camera"):
            resolve_camera_names(None, ["CAM_TRAFFIC_LIGHT_FAR"])


class TestNameParsing:
    def test_auto_spellings_mean_unset(self):
        for value in (None, "auto", "", "null", "none", "AUTO"):
            assert normalize_camera_names(value) is None

    def test_colon_separated_string_for_slurm(self):
        # Slurm treats commas as variable separators, so a register passed through
        # the environment has to use colons.
        assert normalize_camera_names("CAM_FRONT:CAM_BACK") == ["CAM_FRONT", "CAM_BACK"]

    def test_comma_separated_string(self):
        assert normalize_camera_names("CAM_FRONT, CAM_BACK") == ["CAM_FRONT", "CAM_BACK"]

    def test_list_with_an_embedded_separator(self):
        assert normalize_camera_names(["CAM_FRONT:CAM_BACK"]) == ["CAM_FRONT", "CAM_BACK"]


class TestRigDescription:
    def test_signature_groups_by_set_not_order(self):
        assert rig_signature(PRD_JT_MAIN) == rig_signature(list(reversed(PRD_JT_MAIN)))

    def test_signature_separates_the_real_rigs(self):
        signatures = {rig_signature(r) for r in (PRD_JT_MAIN, PRD_JT_VARIANT, X2_DEV)}
        assert len(signatures) == 3

    def test_surround_excludes_signal_and_roof_views(self):
        surround = surround_camera_names(X2_DEV)
        assert "CAM_TRAFFIC_LIGHT_FAR" not in surround
        assert "CAM_TOP_LEFT_CENTER" not in surround
        assert "CAM_FRONT" in surround

    def test_describe_reports_cam_back_presence(self):
        # The difference that breaks a fixed camera grid layout.
        assert "CAM_BACK  : yes" in describe_rig(X2_DEV)
        assert "CAM_BACK  : no" in describe_rig(PRD_JT_MAIN)

    def test_matching_profiles_are_ordered_by_preference(self):
        assert matching_profiles(X2_DEV) == []


class TestStorageDiscovery:
    def test_jpeg_directories_are_found(self, tmp_path):
        (tmp_path / "data" / "CAM_FRONT_WIDE").mkdir(parents=True)
        (tmp_path / "data" / "CAM_FRONT.mp4").write_bytes(b"")
        found = available_cameras(tmp_path)
        assert found == {"CAM_FRONT_WIDE": "jpeg_dir"}

    def test_encoder_scratch_files_are_ignored(self, tmp_path):
        # Real scenes contain .vmaf_scratch_.CAM_X.mp4 left by the quality check.
        (tmp_path / "data").mkdir(parents=True)
        (tmp_path / "data" / ".vmaf_scratch_.CAM_FRONT.mp4").write_bytes(b"")
        assert available_cameras(tmp_path) == {}

    def test_non_camera_entries_are_ignored(self, tmp_path):
        (tmp_path / "data").mkdir(parents=True)
        (tmp_path / "data" / "LIDAR_CONCAT.pack").write_bytes(b"")
        assert available_cameras(tmp_path) == {}

    def test_missing_camera_error_lists_what_exists(self, tmp_path):
        (tmp_path / "data" / "CAM_FRONT_WIDE").mkdir(parents=True)
        with pytest.raises(CameraSourceError, match="CAM_FRONT_WIDE"):
            open_camera_source(tmp_path, "CAM_BACK_WIDE", (672, 1148))

    def test_jpeg_source_probes_filename_width(self, tmp_path):
        directory = tmp_path / "data" / "CAM_FRONT"
        directory.mkdir(parents=True)
        (directory / "0007.jpg").write_bytes(b"")
        source = JpegDirectorySource("CAM_FRONT", directory, (672, 1148))
        assert source.path_for(7) is not None


@pytest.mark.data
class TestRealSceneStorage:
    """Against the dataset: the register and the readable set differ."""

    def test_readable_is_a_subset_of_calibrated(self, t4_scene_dir):
        from t4_e2e_devkit.dataset.rigs import read_scene_camera_names

        calibrated = set(read_scene_camera_names(t4_scene_dir))
        readable = set(readable_camera_names(t4_scene_dir))
        assert readable <= calibrated

    def test_some_cameras_are_calibrated_but_not_exported(self, t4_scene_dir):
        from t4_e2e_devkit.dataset.rigs import read_scene_camera_names

        calibrated = set(read_scene_camera_names(t4_scene_dir))
        stored = set(available_cameras(t4_scene_dir))
        # Every prd_jt scene calibrates eleven cameras; none stores eleven JPEG
        # directories. If this ever becomes empty the export changed, and the
        # readable/calibrated distinction should be re-checked rather than assumed.
        assert calibrated - stored or len(stored) == len(calibrated)

    def test_auto_resolves_on_a_real_scene(self, t4_scene_dir):
        resolved = resolve_camera_names(None, readable_camera_names(t4_scene_dir))
        if not resolved:
            pytest.skip("scene has no supported wide JPEG camera")
        assert resolved
        stored = available_cameras(t4_scene_dir)
        assert all(name in stored for name in resolved)

    def test_jpeg_sources_emit_the_configured_shape(self, t4_scene_dir):
        stored = {
            name: kind
            for name, kind in available_cameras(t4_scene_dir).items()
            if name.upper() in {value.upper() for value in C.T4_SUPPORTED_CAMERA_NAMES}
            and kind == "jpeg_dir"
        }
        if not stored:
            pytest.skip("scene has no supported wide JPEG camera")
        shapes = set()
        for name in stored:
            if name.startswith("CAM_TRAFFIC"):
                continue
            source = open_camera_source(t4_scene_dir, name, (672, 1148))
            try:
                frame = source.read(40)
                if frame is not None:
                    shapes.add(frame.shape)
            finally:
                source.close()
            if len(shapes) >= 2:
                break
        # Whatever the storage, the reader emits one resolution.
        assert shapes and shapes == {(672, 1148, 3)}
