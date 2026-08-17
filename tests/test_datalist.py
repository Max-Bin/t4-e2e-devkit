"""Pure data-list validation tests."""

from __future__ import annotations

import json

import pytest

from t4_e2e_devkit.dataset.datalist import DataList, is_e2e_scene_path, load_data_list
from t4_e2e_devkit.dataset.scene_tags import T4SceneTagIndex


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


def test_scene_tag_filter_is_recorded_in_manifest(tmp_path):
    tag_root = tmp_path / "tags"
    tag_root.mkdir()
    (tag_root / "tags.json").write_text(
        json.dumps(
            {
                "date": "2025-12-24",
                "time_series": {
                    "scene-a": {
                        "whitelist_scenes": [
                            {
                                "scene_id": "scene_000",
                                "event": ["lane_change"],
                                "driving_decisions": {
                                    "lateral": "change_lane_left",
                                    "longitudinal": "driving_forward_keeping_speed",
                                },
                            }
                        ],
                        "blacklist_scenes": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    scene = tmp_path / "prd_jt" / "vehicle" / "2025-12-24" / "scene-a"
    (scene / "derived").mkdir(parents=True)
    (scene / "derived" / "meta.json").write_text(
        json.dumps({"date": "2025-12-24", "scene_name": "scene-a"}),
        encoding="utf-8",
    )
    index = T4SceneTagIndex(tag_root)
    data_list = DataList(
        root=tmp_path,
        rows=[("prd_jt/vehicle/2025-12-24/scene-a", 1), ("prd_jt/vehicle/2025-12-24/other", 2)],
    )
    filtered = data_list.filtered_by_scene_tags(index, include_events=["lane_change"])
    assert filtered.rows == [("prd_jt/vehicle/2025-12-24/scene-a", 1)]
    assert filtered.manifest["runtime_filter"]["scene_tags"]["rows_after"] == 1
    assert filtered.manifest["runtime_filter"]["scene_tags"]["source"] == "external-scene-tags"
    assert "root" not in filtered.manifest["runtime_filter"]["scene_tags"]
