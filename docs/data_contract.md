# Data contract

The reader exposes one `T4Scene` per `(scene_dir, center_frame)`. Shapes and
field names are defined in
[`common/constants.py`](../t4_e2e_devkit/common/constants.py).

## Scene layout

```text
<scene_dir>/
├── derived/meta.json
├── derived/scalars.npz
├── derived/frames.pack
├── derived/cam_names.json
├── data/LIDAR_CONCAT.pack
├── data/CAM_*/...
└── annotation/                 # optional calibration/timing tables
```

`scalars.npz` contains the global ego trajectory `(x, y, cos, sin)`, dynamic
state, destination, vehicle shape and camera calibration. `frames.pack`
contains per-frame map tensors and tracked-object annotations. The reader
converts all geometry into the current frame's ego coordinates.

Optional files include `metadata.json` (area-map identity), `route.json`
(ordered route primitive IDs, start and goal poses), and an external scene-tag
root. The external tag files are runtime inputs and are not copied into the
repository.

## Time ranges

| range | default | meaning |
|---|---:|---|
| history | 31 frames | current frame plus 30 past frames at 10 Hz |
| recorded future | 80 frames | 8 seconds at 10 Hz |
| PDM observation | current + 50 frames | TTC context |
| PDM score | 40 poses | 4 seconds at 10 Hz |

These ranges are independent. A trajectory producer may emit any uniform grid
that is covered by the recorded future:

```python
TrajectorySampling(num_poses=80, interval_length=0.1)
TrajectorySampling(num_poses=8, interval_length=0.5)
```

The evaluator resamples to its own grid. The visualizer keeps the producer's
grid. A trajectory shorter than the requested target horizon is an error.

## Map fields

All map fields are in the current ego frame:

| field | shape |
|---|---|
| `lanes` | `[140, 20, 33]` |
| `lanes_speed_limit`, `lanes_has_speed_limit` | `[140, 1]` |
| `route_lanes` | `[25, 20, 33]` |
| `route_lanes_speed_limit`, `route_lanes_has_speed_limit` | `[25, 1]` |
| `polygons` | `[10, 40, 3]` |
| `line_strings` | `[60, 20, 4]` |
| `goal_pose` | `[4]`, `(x, y, cos, sin)` |
| `ego_shape` | `[3]`, `(wheel_base, length, width)` |

The 33 columns of a lane point are:

```text
[x, y, dx, dy, left_offset_x, left_offset_y,
 right_offset_x, right_offset_y, traffic_light(5), left_line_type(10),
 right_line_type(10)]
```

Boundary coordinates are centerline plus offset. Zero rows are padding and are
not geometry. Missing required map fields raise an error; they are not
zero-filled.

### Source IDs and matching audit

The source of each identity is explicit:

| identity | source |
|---|---|
| area-map identity | `metadata.json` → `area_map.id/version_id` |
| ordered route IDs | `route.json` |
| lanelet IDs and geometry | the resolved `lanelet2_map.osm` `<relation id="…">` entries |

There is no separate precomputed match table in the dataset. When
`t4_attach_map_ids=true`, the reader parses all source objects from the OSM file
and matches scene-local rows geometrically. The full lanelet ID index is
available as `T4MapAPI.available_object_ids`; semantic IDs are available through
`available_ids(object_type=...)`; the row-level result is
`MapTensors.object_ids.matches`.

Each `MapObjectMatch` records the tensor layer and row, source ID, source-file
label, frame index, score, candidates and a reason. The label is only a
portable filename; local filesystem prefixes are not serialized. `None` means
padding, missing source geometry or an unsuccessful match; IDs are never
fabricated. Unmatched rows retain their candidate IDs and reason for audit.
`T4MapAPI.get_objects()` and `query_objects()` expose source tags and geometry
for lanelets, lane connectors, roadblocks, intersections, line strings,
crosswalks, stop lines, traffic lights, regulatory elements and areas when
present in the source map. Topology helpers expose successors, predecessors,
adjacency, route chains and lane-associated regulatory objects.

To export an audit sidecar, choose an ignored runtime directory explicitly:

```python
scene.current_frame.map_tensors.object_ids.write_json(
    "results/cache/map_matches/scene_000@100.json"
)
```

`SceneMetadata.route_metadata` exposes the route file's ordered primitive IDs
and a portable source-file label whenever `route.json` exists.

## Scene tags

Scene tags are optional external metadata. A tag has a curation `status`
(`whitelist` or `blacklist`) and independent semantic fields: `events`,
`lateral_decision`, `longitudinal_decision`, `dynamic_entities`, `scenery`,
the source interval and the complete original JSON record. The status is not a
replacement for the behavior taxonomy. Unknown future fields remain available
through `T4SceneTag.raw`.
Treat `raw` as runtime metadata; do not commit or publish external tag files
without checking their curator and project-specific fields.

Pass `t4_scene_tags_root` in `reader_config` to attach tags to
`SceneMetadata.scene_tags`. To build a semantically filtered list:

```bash
uv run t4e2e datalist \
  --root /path/to/t4_dataset \
  --scene-tags-root /path/to/scene_tags \
  --include-tag-event lane_change \
  --include-lateral-decision change_lane_left \
  --out lists/lane-change.json
```

The manifest records that external tags were used and stores the semantic
filters, but not the local taxonomy path.

## Annotations

Tracked boxes use the 9-column layout:

```text
[x, y, z, width, length, height, yaw, vx, vy]
```

Labels are `car`, `truck`, `bus`, `bicycle` and `pedestrian` class ids defined
by `T4TrackLabel`. Object counts are variable per frame. Empty annotations are
valid only when the source explicitly contains no objects; missing annotation
fields are an error.

## Sensors

Sensor decoding is declared by `SensorConfig`:

```python
SensorConfig.build_no_sensors()
SensorConfig.build_current_frame(lidar=False)
SensorConfig(cameras={"CAM_FRONT_WIDE": [-1, -5]}, lidar=False)
```

Camera images are RGB `uint8 [H, W, 3]` with calibration at the decoded
resolution. The public camera input currently supports only JPEG-backed wide
channels; narrow and video-backed channels are rejected. LiDAR, when requested, is ragged `[N, 5]` with
`(x, y, z, intensity, ring_or_time)`.

## Data lists

A list is a reproducible JSON manifest:

```json
{
  "format": "t4-e2e.datalist",
  "version": 1,
  "root": "/path/to/dataset",
  "rows": [["prd_jt/scene", 100]]
}
```

The manifest may also record window settings, camera requirements and filtering
statistics. Its dataset root is local to the machine that built the list;
keep generated lists outside version control. The reader accepts only the
current `t4-e2e.datalist` format and version.
