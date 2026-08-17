"""Scene-tag taxonomy and matching tests."""

from __future__ import annotations

import json

from t4_e2e_devkit.dataset.scene_tags import T4SceneTagIndex


def _write_tag_file(path, *, debug=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": "2025-12-24",
        "vehicle_id": "e2e",
        "taxonomy_version": "0.1.1",
        "metadata": {"author": "test", "schema_revision": 3},
        "time_series": {
            "13_47_17-13_48_17": {
                "name": "13_47_17-13_48_17",
                "whitelist_scenes": [
                    {
                        "scene_id": "scene_000",
                        "key_time": "100",
                        "start_time": "100",
                        "end_time": "200",
                        "driving_decisions": {
                            "lateral": "change_lane_left",
                            "longitudinal": "driving_forward_accelerating",
                        },
                        "event": ["lane_change", "speed_transition"],
                        "dynamic_entities": [{"type": "pedestrian", "count": 2}],
                        "scenery": {"road_type": "urban"},
                        "future_taxonomy_field": {"value": True},
                    }
                ],
                "blacklist_scenes": [
                    {
                        "scene_id": "scene_001",
                        "key_time": "200",
                        "start_time": "200",
                        "end_time": "300",
                        "justification": "Other",
                        "event": ["stopped"],
                        "scenery": {},
                    }
                ],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scene_tags_keep_status_and_full_semantics(tmp_path):
    root = tmp_path / "tags"
    _write_tag_file(root / "2025-12-24.json")
    _write_tag_file(root / "debug" / "events.json")

    scene = tmp_path / "dataset" / "prd_jt" / "vehicle" / "2025-12-24" / "13_47_17-13_48_17"
    (scene / "derived").mkdir(parents=True)
    (scene / "derived" / "meta.json").write_text(
        json.dumps({"date": "2025-12-24", "scene_name": scene.name}),
        encoding="utf-8",
    )

    index = T4SceneTagIndex(root)
    tags = index.tags_for_scene(scene)
    assert [tag.status for tag in tags] == ["whitelist", "blacklist"]
    assert tags[0].lateral_decision == "change_lane_left"
    assert tags[0].source_path == "2025-12-24.json"
    assert tags[0].longitudinal_decision == "driving_forward_accelerating"
    assert tags[0].events == ("lane_change", "speed_transition")
    assert tags[0].dynamic_entities == [{"type": "pedestrian", "count": 2}]
    assert tags[0].scenery == {"road_type": "urban"}
    assert tags[0].raw["future_taxonomy_field"] == {"value": True}
    assert tags[0].metadata["schema_revision"] == 3
    assert index.has_event(scene, "lane_change")


def test_scene_tag_filter_is_semantic_and_preserves_unknown_statuses(tmp_path):
    root = tmp_path / "tags"
    _write_tag_file(root / "2025-12-24.json")
    scene = tmp_path / "dataset" / "prd_jt" / "vehicle" / "2025-12-24" / "13_47_17-13_48_17"
    (scene / "derived").mkdir(parents=True)
    (scene / "derived" / "meta.json").write_text(
        json.dumps({"date": "2025-12-24", "scene_name": scene.name}),
        encoding="utf-8",
    )

    index = T4SceneTagIndex(root)
    assert index.filter_scene_dirs(
        [scene],
        include_events=["lane_change"],
        include_lateral_decisions=["change_lane_left"],
        include_longitudinal_decisions=["driving_forward_accelerating"],
    ) == [scene]
    assert index.filter_scene_dirs([scene], statuses=["blacklist"]) == [scene]
    assert index.filter_scene_dirs([scene], include_events=["turn_left"]) == []
