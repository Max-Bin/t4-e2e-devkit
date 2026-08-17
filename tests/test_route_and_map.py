"""T4 route metadata and Lanelet2 ID recovery tests."""

from __future__ import annotations

import json

import numpy as np

from t4_e2e_devkit.common.dataclasses import MapObjectIds
from t4_e2e_devkit.common.t4_map import T4MapAPI
from t4_e2e_devkit.dataset.route import load_t4_route


def test_route_reader_keeps_ordered_primitive_ids_and_area_map(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "metadata.json").write_text(
        json.dumps({"area_map": {"id": 1423, "version_id": "v1"}}),
        encoding="utf-8",
    )
    (scene / "route.json").write_text(
        json.dumps(
            {
                "start_pose": {
                    "position": {"x": 1, "y": 2, "z": 3},
                    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                },
                "goal_pose": {
                    "position": {"x": 4, "y": 5, "z": 6},
                    "orientation": {"x": 0, "y": 0, "z": 1, "w": 0},
                },
                "segments": [
                    {
                        "preferred_primitive": {"id": 10, "primitive_type": "lane"},
                        "primitives": [
                            {"id": 10, "primitive_type": "lane"},
                            {"id": 11, "primitive_type": "area"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    route = load_t4_route(scene, strict=True)
    assert route is not None
    assert route.route_lane_ids == ("10",)
    assert route.primitive_ids == ("10", "11")
    assert route.preferred_lane_ids == ("10",)
    assert route.area_map_id == "1423"
    assert route.area_map_version_id == "v1"


def test_lanelet_map_parser_recovers_ids_geometry_speed_and_graph(tmp_path):
    osm = tmp_path / "lanelet2_map.osm"
    osm.write_text(
        """<?xml version="1.0"?>
<osm>
  <node id="1"><tag k="local_x" v="0"/><tag k="local_y" v="1"/></node>
  <node id="2"><tag k="local_x" v="10"/><tag k="local_y" v="1"/></node>
  <node id="3"><tag k="local_x" v="0"/><tag k="local_y" v="-1"/></node>
  <node id="4"><tag k="local_x" v="10"/><tag k="local_y" v="-1"/></node>
  <node id="5"><tag k="local_x" v="20"/><tag k="local_y" v="1"/></node>
  <node id="6"><tag k="local_x" v="20"/><tag k="local_y" v="-1"/></node>
  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="11"><nd ref="3"/><nd ref="4"/></way>
  <way id="12"><nd ref="2"/><nd ref="5"/></way>
  <way id="13"><nd ref="4"/><nd ref="6"/></way>
  <way id="14"><nd ref="5"/><nd ref="6"/></way>
  <relation id="100">
    <member type="way" role="left" ref="10"/>
    <member type="way" role="right" ref="11"/>
    <tag k="type" v="lanelet"/><tag k="speed_limit" v="36"/>
  </relation>
  <relation id="200">
    <member type="way" role="left" ref="12"/>
    <member type="way" role="right" ref="13"/>
    <tag k="type" v="lanelet"/>
  </relation>
</osm>
""",
        encoding="utf-8",
    )

    api = T4MapAPI(osm, route_lane_ids=["100"])
    lane = api.get_lane(100)
    assert lane is not None
    assert lane.id == "100"
    assert lane.speed_limit_mps == 10.0
    assert lane.polygon.area > 0
    assert api.get_lane(999) is None
    assert api.get_successors("100")[0].id == "200"
    assert api.get_predecessors("200")[0].id == "100"
    assert api.get_nearest_lane((4, 0)).id == "100"
    assert [candidate.id for candidate in api.get_proximal_lanes((4, 0), 0.2)] == ["100"]

    ids = api.match_local_centerlines(
        np.asarray([[[0, 0, 0], [5, 0, 0], [10, 0, 0]]], dtype=np.float32),
        [0, 0, 1, 0],
    )
    assert ids == ("100",)

    detailed = api.match_local_centerlines_detailed(
        np.asarray(
            [
                [[0, 0, 0], [5, 0, 0], [10, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=np.float32,
        ),
        [0, 0, 1, 0],
        frame_index=7,
    )
    assert detailed[0].source_object_id == "100"
    assert detailed[0].match_distance_m == 0.0
    assert detailed[0].candidate_ids[0] == "100"
    assert detailed[0].source_path == osm.name
    assert detailed[0].frame_index == 7
    assert detailed[0].reason == "matched"
    assert detailed[1].source_object_id is None
    assert detailed[1].reason == "padding"

    unsupported = api.unmatched_rows(
        np.asarray([[[1, 1, 0], [2, 1, 0]], [[0, 0, 0], [0, 0, 0]]]),
        layer="polygons",
        frame_index=7,
    )
    assert unsupported[0].reason == "unsupported_source_type"
    assert unsupported[1].reason == "padding"

    sidecar = tmp_path / "results" / "cache" / "map_matches.json"
    object_ids = MapObjectIds(
        lane_ids=(detailed[0].source_object_id, detailed[1].source_object_id),
        source_path=str(osm.resolve()),
        frame_index=7,
        matches=detailed,
    )
    assert object_ids.write_json(sidecar) == str(sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["frame_index"] == 7
    assert payload["source_path"] == osm.name
    assert payload["matches"][0]["source_object_id"] == "100"
