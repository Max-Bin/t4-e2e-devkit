"""The one reader and the one writer every report path now shares.

Four atomic JSON writers existed with three levels of care; these tests pin the
properties the surviving one has to keep, and the leniency the readers owe a
dashboard rendering a run that died mid-write.
"""

from __future__ import annotations

import json

import pytest

from t4_e2e_devkit.common.artifact_io import (
    json_value,
    portable_value,
    read_csv_rows,
    read_mapping,
    write_json_atomic,
)


class TestWriteJsonAtomic:
    def test_writes_sorted_json_with_a_trailing_newline(self, tmp_path):
        path = write_json_atomic(tmp_path / "nested" / "run.json", {"b": 1, "a": 2})
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n") and text.index('"a"') < text.index('"b"')
        assert json.loads(text) == {"a": 2, "b": 1}

    def test_leaves_no_temporary_behind(self, tmp_path):
        write_json_atomic(tmp_path / "run.json", {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["run.json"]

    def test_a_failed_write_leaves_neither_file_nor_temporary(self, tmp_path):
        # allow_nan=False is how the rollout artifacts refuse a non-finite
        # number; the failure must not leave a temporary lying next to the
        # destination for a later reader to trip over.
        with pytest.raises(ValueError):
            write_json_atomic(tmp_path / "run.json", {"a": float("nan")}, allow_nan=False)
        assert list(tmp_path.iterdir()) == []

    def test_nan_is_allowed_by_default(self, tmp_path):
        # Metrics legitimately carry NaN for "no value"; only the artifact path
        # treats it as a bug.
        path = write_json_atomic(tmp_path / "metrics.json", {"score": float("nan")})
        assert "NaN" in path.read_text(encoding="utf-8")

    def test_replacing_an_existing_report_is_one_step(self, tmp_path):
        path = tmp_path / "run.json"
        write_json_atomic(path, {"generation": 1})
        write_json_atomic(path, {"generation": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"generation": 2}

    def test_two_writers_of_one_path_do_not_share_a_temporary(self, tmp_path, monkeypatch):
        """The bug in one of the four copies: a temporary named without the pid.

        Two ranks writing the same report would then use the same temporary
        path, and one could unlink or overwrite the other's half-written file.
        """
        names = []
        import tempfile as tempfile_module

        real_mkstemp = tempfile_module.mkstemp

        def recording_mkstemp(*args, **kwargs):
            descriptor, path = real_mkstemp(*args, **kwargs)
            names.append(path)
            return descriptor, path

        monkeypatch.setattr(tempfile_module, "mkstemp", recording_mkstemp)
        write_json_atomic(tmp_path / "run.json", {"a": 1})
        write_json_atomic(tmp_path / "run.json", {"a": 2})
        assert len(set(names)) == 2


class TestReadMapping:
    def test_reads_an_object(self, tmp_path):
        path = tmp_path / "run.json"
        path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        assert read_mapping(path) == {"status": "completed"}

    def test_a_missing_file_is_absent_not_fatal(self, tmp_path):
        assert read_mapping(tmp_path / "nothing.json") == {}

    def test_a_truncated_file_is_absent_not_fatal(self, tmp_path):
        # What a run killed mid-write leaves behind.  A dashboard renders the
        # panels that are fine rather than refusing the whole report.
        path = tmp_path / "run.json"
        path.write_text('{"status": "comple', encoding="utf-8")
        assert read_mapping(path) == {}

    def test_a_non_object_is_absent(self, tmp_path):
        path = tmp_path / "run.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert read_mapping(path) == {}

    def test_the_default_is_returned_instead(self, tmp_path):
        sentinel = {"status": "unknown"}
        assert read_mapping(tmp_path / "nothing.json", default=sentinel) is sentinel


class TestReadCsvRows:
    def test_reads_header_keyed_rows(self, tmp_path):
        path = tmp_path / "per_window.csv"
        path.write_text("token,score\na,1\nb,2\n", encoding="utf-8")
        assert read_csv_rows(path) == [
            {"token": "a", "score": "1"},
            {"token": "b", "score": "2"},
        ]

    def test_a_missing_file_is_no_rows(self, tmp_path):
        assert read_csv_rows(tmp_path / "nothing.csv") == []

    def test_a_header_only_file_is_no_rows(self, tmp_path):
        path = tmp_path / "per_window.csv"
        path.write_text("token,score\n", encoding="utf-8")
        assert read_csv_rows(path) == []


class TestPortableValue:
    """The tolerant coercion, for records a run writes about itself."""

    def test_arrays_and_nested_containers_become_data(self):
        import numpy as np

        value = {"poses": np.zeros((2, 2)), "labels": ("a", "b"), "n": 3}
        assert portable_value(value) == {
            "poses": [[0.0, 0.0], [0.0, 0.0]],
            "labels": ["a", "b"],
            "n": 3,
        }

    def test_an_object_with_as_dict_is_asked(self):
        class Report:
            def as_dict(self):
                return {"status": "completed"}

        assert portable_value(Report()) == {"status": "completed"}

    def test_a_plain_object_is_walked(self):
        class State:
            def __init__(self):
                self.x = 1.0

        assert portable_value(State()) == {"x": 1.0}

    def test_an_unknown_type_is_described_not_dropped(self):
        # A set of task ids is the realistic case: JSON has no set, and a record
        # that stringifies the field beats one that omits it.
        assert portable_value({"a"}) == str({"a"})


class TestJsonValue:
    """The strict coercion, for documents a reader parses back."""

    def test_nested_containers_pass_through(self):
        assert json_value({"a": [1, 2]}, what="configuration value") == {"a": [1, 2]}

    def test_an_unknown_type_is_an_error_naming_the_caller(self):
        with pytest.raises(TypeError, match="configuration value is not JSON serializable"):
            json_value(object(), what="configuration value")

    def test_non_finite_floats_pass_unless_refused(self):
        assert json_value(float("inf"), what="metric") == float("inf")
        with pytest.raises(ValueError, match="submission metadata must not"):
            json_value(float("nan"), what="submission metadata", require_finite=True)

    def test_the_refusal_reaches_into_containers(self):
        with pytest.raises(ValueError, match="must not contain non-finite"):
            json_value(
                {"scores": [1.0, float("nan")]}, what="submission metadata", require_finite=True
            )
