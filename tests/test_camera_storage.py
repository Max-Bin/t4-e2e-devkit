"""Camera rig resolution and the public JPEG input boundary."""

from __future__ import annotations

import pytest

from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.dataset.camera_source import (
    CameraSource,
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
    "CAM_BACK_LEFT",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT",
    "CAM_BACK_RIGHT_WIDE",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_FRONT_WIDE",
    "CAM_TRAFFIC_LIGHT_FAR",
]
PRD_JT_VARIANT = [
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_TRAFFIC_LIGHT_FAR",
]
X2_DEV = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_WIDE",
    "CAM_TOP_LEFT_CENTER",
    "CAM_TOP_RIGHT_CENTER",
    "CAM_TRAFFIC_LIGHT_FAR",
    "CAM_TRAFFIC_LIGHT_NEAR",
]

# What each rig stores as JPEG directories -- the set a register resolves
# against.  The difference from the calibration registers above is the whole
# point: prd_jt ships its narrow views as HEVC, x2_dev ships the same names as
# JPEG, so the same channel name is readable on one rig and not the other.
PRD_JT_MAIN_JPEG = [
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT_WIDE",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_FRONT_WIDE",
]
PRD_JT_VARIANT_JPEG = [
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT_WIDE",
]
X2_DEV_JPEG = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_WIDE",
    "CAM_TOP_LEFT_CENTER",
    "CAM_TOP_RIGHT_CENTER",
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

    def test_x2_surround_six_fits_x2_dev(self):
        assert resolve_camera_names("x2_surround6", X2_DEV_JPEG) == list(
            C.T4_X2_SURROUND6_CAMERA_NAMES
        )

    def test_x2_surround_six_is_refused_on_prd_jt(self):
        # The names are calibrated there, but prd_jt stores them as HEVC, so they
        # never reach the readable set.  A register that resolved anyway would
        # train on frames this reader cannot decode.
        with pytest.raises(RigMismatch, match="CAM_BACK"):
            resolve_camera_names("x2_surround6", PRD_JT_MAIN_JPEG)

    def test_unknown_profile_name_is_rejected(self):
        with pytest.raises(RigMismatch, match="not supported"):
            resolve_camera_names("surround6", X2_DEV)

    def test_auto_picks_the_profile_that_fits_the_rig(self):
        assert resolve_camera_names(None, PRD_JT_MAIN_JPEG) == list(C.T4_WIDE5_CAMERA_NAMES)
        assert resolve_camera_names(None, X2_DEV_JPEG) == list(C.T4_X2_SURROUND6_CAMERA_NAMES)

    def test_auto_uses_available_wide_subset_for_variant(self):
        assert resolve_camera_names(None, PRD_JT_VARIANT_JPEG) == [
            "CAM_FRONT_LEFT_WIDE",
            "CAM_FRONT_RIGHT_WIDE",
        ]

    def test_explicit_missing_camera_is_an_error(self):
        with pytest.raises(RigMismatch, match="absent"):
            resolve_camera_names(["CAM_BACK_WIDE"], PRD_JT_MAIN)

    def test_a_narrow_channel_resolves_where_it_is_jpeg(self):
        assert resolve_camera_names(["CAM_FRONT", "CAM_BACK"], X2_DEV_JPEG) == [
            "CAM_FRONT",
            "CAM_BACK",
        ]

    def test_a_narrow_channel_is_refused_where_it_is_video(self):
        with pytest.raises(RigMismatch, match="absent"):
            resolve_camera_names(["CAM_FRONT"], PRD_JT_MAIN_JPEG)

    def test_roof_views_stay_unsupported_even_as_jpeg(self):
        # x2_dev stores them as JPEG directories, but they point at signal heads
        # rather than the road, so they do not compose with a surround register.
        with pytest.raises(RigMismatch, match="not supported"):
            resolve_camera_names(["CAM_TOP_LEFT_CENTER"], X2_DEV_JPEG)

    def test_resolution_preserves_requested_order(self):
        # Register order is part of the learned camera contract, so it must never
        # be sorted on the way through.
        requested = ["CAM_FRONT_RIGHT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_LEFT_WIDE"]
        assert resolve_camera_names(requested, PRD_JT_MAIN) == requested

    def test_require_count_is_enforced(self):
        with pytest.raises(RigMismatch, match="required"):
            resolve_camera_names("wide5", PRD_JT_MAIN, require_count=6)

    def test_no_profile_fits_reports_every_requirement(self):
        with pytest.raises(RigMismatch, match="no supported JPEG camera"):
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


class TestNameTokenizer:
    """One tokenizer for every boundary a register arrives through."""

    def test_the_two_normalizers_agree_on_tokens(self):
        from t4_e2e_devkit.dataset.rigs import split_camera_names
        from t4_e2e_devkit.dataset.scene import normalize_t4_camera_names

        # dataset.scene and dataset.rigs each parsed this, and each had its own
        # copy of the splitting; only the upper-casing and the sentinels differ.
        value = "CAM_FRONT: cam_back ,"
        assert split_camera_names(value) == ["CAM_FRONT", "cam_back"]
        assert normalize_camera_names(value) == ["CAM_FRONT", "cam_back"]
        assert normalize_t4_camera_names(value) == ["CAM_FRONT", "CAM_BACK"]

    def test_a_sentinel_is_only_the_rig_module_s_business(self):
        from t4_e2e_devkit.dataset.scene import normalize_t4_camera_names

        # "auto" means "resolve against the scene" to the rig module, and is
        # never a register the reader may take.
        assert normalize_camera_names("auto") is None
        with pytest.raises(ValueError, match="unknown T4 camera name"):
            normalize_t4_camera_names("auto")


class TestConfigBool:
    """The drift the two copies had: one stripped, the other did not."""

    def test_whitespace_does_not_flip_a_flag(self):
        from t4_e2e_devkit.dataset.scene import as_config_bool

        for value in (" true", "true ", "TRUE", "1", "yes", "on", True):
            assert as_config_bool(value) is True
        for value in ("false", " false ", "0", "no", "off", "", None, False):
            assert as_config_bool(value) is False


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

    def test_each_rig_matches_its_own_profile(self):
        # One profile per rig, and the two are disjoint: no register spans both,
        # so a checkpoint cannot be moved between them by resolution alone.
        assert matching_profiles(PRD_JT_MAIN_JPEG) == ["wide5"]
        assert matching_profiles(X2_DEV_JPEG) == ["x2_surround6"]
        assert matching_profiles(PRD_JT_VARIANT_JPEG) == []


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


def _write_jpeg(path, width=64, height=48):
    """A real JPEG with structure, so a resize difference cannot hide in flat pixels."""
    import numpy as np
    from PIL import Image

    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack([rows * 5, columns * 3, (rows + columns) * 7], axis=-1).astype(np.uint8)
    Image.fromarray(pixels).save(path, quality=95)
    return pixels


class _UnsplittableSource(CameraSource):
    """A backend with no per-frame blob, e.g. inter-frame-compressed video."""

    def read(self, frame_index):
        return None

    def native_size(self):
        return None

    def describe(self):
        return {"camera": self.name, "storage": "fake"}


class TestUndecodedFrames:
    """``read_encoded`` is the boundary a training loop decodes on the GPU behind.

    Workers cannot use CUDA, so a fast camera path has to move the bytes across
    the worker boundary still compressed.  Two properties make that safe: the
    bytes are the stored file verbatim (no re-encode, so no second generation of
    JPEG loss), and they decode to what ``read`` decodes.
    """

    def test_bytes_are_the_stored_file_verbatim(self, tmp_path):
        directory = tmp_path / "data" / "CAM_FRONT_WIDE"
        directory.mkdir(parents=True)
        _write_jpeg(directory / "00007.jpg")
        source = JpegDirectorySource("CAM_FRONT_WIDE", directory, (24, 32))
        assert source.read_encoded(7) == (directory / "00007.jpg").read_bytes()

    def test_four_digit_names_are_found_too(self, tmp_path):
        # Raw exports mix five- and four-digit names, sometimes in one directory,
        # and the encoded path must follow the same probe as path_for.
        directory = tmp_path / "data" / "CAM_FRONT_WIDE"
        directory.mkdir(parents=True)
        _write_jpeg(directory / "0007.jpg")
        source = JpegDirectorySource("CAM_FRONT_WIDE", directory, (24, 32))
        assert source.read_encoded(7) == (directory / "0007.jpg").read_bytes()

    def test_an_absent_frame_is_none_not_an_error(self, tmp_path):
        directory = tmp_path / "data" / "CAM_FRONT_WIDE"
        directory.mkdir(parents=True)
        _write_jpeg(directory / "00007.jpg")
        source = JpegDirectorySource("CAM_FRONT_WIDE", directory, (24, 32))
        assert source.read_encoded(8) is None

    def test_the_bytes_decode_to_what_read_decodes(self, tmp_path):
        """Same pixels, exactly, once the caller applies the same resize.

        This is the property the GPU decode path is substituted into: it may
        resample differently, but it must start from these pixels.
        """
        import io

        import numpy as np
        from PIL import Image

        directory = tmp_path / "data" / "CAM_FRONT_WIDE"
        directory.mkdir(parents=True)
        _write_jpeg(directory / "00007.jpg")
        source = JpegDirectorySource("CAM_FRONT_WIDE", directory, (24, 32))

        blob = source.read_encoded(7)
        with Image.open(io.BytesIO(blob)) as image:
            from_bytes = np.asarray(
                image.convert("RGB").resize((source.width, source.height)), np.uint8
            )
        assert np.array_equal(from_bytes, source.read(7))

    def test_a_backend_without_per_frame_blobs_refuses_loudly(self):
        """Refusing, not ``None``: "unsupported" is not "this frame is absent".

        A caller that read ``None`` as absence would mean-fill every frame of the
        camera and train on a constant image that looks like data.
        """
        source = _UnsplittableSource("CAM_FRONT_WIDE", (672, 1148))
        with pytest.raises(CameraSourceError, match="cannot hand out undecoded"):
            source.read_encoded(0)


@pytest.mark.data
class TestRealSceneStorage:
    """Against the dataset: the register and the readable set differ."""

    def test_readable_is_a_subset_of_calibrated(self, rig_scene_dir):
        from t4_e2e_devkit.dataset.rigs import read_scene_camera_names

        calibrated = set(read_scene_camera_names(rig_scene_dir))
        readable = set(readable_camera_names(rig_scene_dir))
        assert readable <= calibrated

    def test_some_cameras_are_calibrated_but_not_exported(self, rig_scene_dir):
        from t4_e2e_devkit.dataset.rigs import read_scene_camera_names

        calibrated = set(read_scene_camera_names(rig_scene_dir))
        stored = set(available_cameras(rig_scene_dir))
        # Both rigs calibrate eleven cameras and neither stores eleven JPEG
        # directories: prd_jt exports five, x2_dev nine. If this ever becomes
        # empty the export changed, and the readable/calibrated distinction
        # should be re-checked rather than assumed.
        assert calibrated - stored or len(stored) == len(calibrated)

    def test_auto_resolves_on_a_real_scene(self, rig_scene_dir):
        resolved = resolve_camera_names(None, readable_camera_names(rig_scene_dir))
        if not resolved:
            pytest.skip("scene has no supported JPEG camera")
        assert resolved
        stored = available_cameras(rig_scene_dir)
        assert all(name in stored for name in resolved)

    def test_jpeg_sources_emit_the_configured_shape(self, rig_scene_dir):
        stored = {
            name: kind
            for name, kind in available_cameras(rig_scene_dir).items()
            if name.upper() in {value.upper() for value in C.T4_SUPPORTED_CAMERA_NAMES}
            and kind == "jpeg_dir"
        }
        if not stored:
            pytest.skip("scene has no supported JPEG camera")
        shapes = set()
        for name in stored:
            if name.startswith("CAM_TRAFFIC"):
                continue
            source = open_camera_source(rig_scene_dir, name, (672, 1148))
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

    def test_encoded_bytes_match_the_export_and_decode_to_read(self, t4_scene_dir):
        """The synthetic JPEG above proves the contract; this proves the export.

        A real wide frame is 2880x1860 and ~1.3 MB, so it exercises the paths a
        64x48 fixture cannot: the digit-width probe against real filenames, and a
        decode where any resampling disagreement would be visible.
        """
        import io

        import numpy as np
        from PIL import Image

        names = [
            name
            for name in available_cameras(t4_scene_dir)
            if name.upper() in {value.upper() for value in C.T4_SUPPORTED_CAMERA_NAMES}
            and not name.startswith("CAM_TRAFFIC")
        ]
        if not names:
            pytest.skip("scene has no supported JPEG camera")
        source = open_camera_source(t4_scene_dir, names[0], (672, 1148))
        try:
            blob = source.read_encoded(40)
            if blob is None:
                pytest.skip(f"{names[0]} has no frame 40")
            assert blob == source.path_for(40).read_bytes()
            with Image.open(io.BytesIO(blob)) as image:
                from_bytes = np.asarray(
                    image.convert("RGB").resize((source.width, source.height)), np.uint8
                )
            assert np.array_equal(from_bytes, source.read(40))
        finally:
            source.close()


@pytest.mark.data
class TestRealRigRegisters:
    """The rigs, as the dataset actually ships them.

    The synthetic registers above encode what was measured once; these assert it
    against the scenes on disk, so an export change shows up as a failing test
    rather than as a training run reading one camera where it wanted six.
    """

    def test_x2_dev_resolves_its_surround_profile(self, x2_scene_dir):
        readable = readable_camera_names(x2_scene_dir)
        assert resolve_camera_names("x2_surround6", readable, scene_dir=x2_scene_dir) == list(
            C.T4_X2_SURROUND6_CAMERA_NAMES
        )

    def test_wide_five_is_refused_on_a_real_x2_dev_scene(self, x2_scene_dir):
        # The single wide view is CAM_FRONT_WIDE; the four other wide channels do
        # not exist on this rig at all, and the refusal names the one that does
        # fit so the caller is not left guessing.
        with pytest.raises(RigMismatch, match="x2_surround6"):
            resolve_camera_names(
                "wide5", readable_camera_names(x2_scene_dir), scene_dir=x2_scene_dir
            )

    def test_x2_dev_narrow_views_decode(self, x2_scene_dir):
        source = open_camera_source(x2_scene_dir, "CAM_BACK", (672, 1148))
        try:
            frame = source.read(40)
            if frame is None:
                pytest.skip("CAM_BACK has no frame 40")
            assert frame.shape == (672, 1148, 3)
        finally:
            source.close()

    def test_prd_jt_narrow_views_are_video_and_say_so(self, t4_scene_dir):
        from t4_e2e_devkit.dataset.rigs import read_scene_camera_names

        if "CAM_FRONT" not in read_scene_camera_names(t4_scene_dir):
            pytest.skip("scene does not calibrate CAM_FRONT")
        assert "CAM_FRONT" not in readable_camera_names(t4_scene_dir)
        # Supported as a channel, unreadable on this rig: the message has to name
        # the storage, or the reader looks like it lost a converted camera.
        with pytest.raises(CameraSourceError, match="stored as video"):
            open_camera_source(t4_scene_dir, "CAM_FRONT", (672, 1148))

    def test_the_two_rigs_do_not_share_a_register(self, t4_scene_dir, x2_scene_dir):
        prd_jt = resolve_camera_names(None, readable_camera_names(t4_scene_dir))
        x2_dev = resolve_camera_names(None, readable_camera_names(x2_scene_dir))
        assert set(prd_jt).isdisjoint(x2_dev)
