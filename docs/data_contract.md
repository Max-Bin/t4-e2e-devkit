# Data contract

The reader exposes one `T4Scene` per `(scene_dir, center_frame)`. Shapes and
field names are defined in
[`common/constants.py`](../t4_e2e_devkit/common/constants.py) and checked by
`tests/test_contract.py`.

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
statistics. Older compatible format strings are accepted on read.
