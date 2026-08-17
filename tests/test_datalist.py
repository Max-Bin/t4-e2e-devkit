"""Pure data-list validation tests."""

from __future__ import annotations

import json

import pytest

from t4_e2e_devkit.dataset.datalist import DataList, is_e2e_scene_path, load_data_list


def test_subtree_check_matches_path_components(tmp_path):
    assert is_e2e_scene_path("prd_jt/date/vehicle/scene")
    assert is_e2e_scene_path("prd_jt_val/date/vehicle/scene")
    assert not is_e2e_scene_path("prd_jt_backup/date/vehicle/scene")
    assert not is_e2e_scene_path("prd_jt/../annotated_data/scene")

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"root": str(tmp_path), "rows": [["prd_jt_backup/x", 1]]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="annotation-free"):
        load_data_list(path)


def test_absolute_and_traversal_rows_are_rejected(tmp_path):
    for scene in ("/tmp/prd_jt/scene", "prd_jt/../../outside"):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"root": str(tmp_path), "rows": [[scene, 1]]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="relative"):
            load_data_list(path)


def test_negative_runtime_limit_is_rejected(tmp_path):
    data_list = DataList(root=tmp_path, rows=[("prd_jt/a", 1)])
    with pytest.raises(ValueError, match="non-negative"):
        data_list.filtered(max_rows=-1)
