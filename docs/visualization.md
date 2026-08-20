# Visualization

The renderer produces a single current-frame view from T4 map tensors,
annotations and trajectories. It does not require LiDAR. Camera and LiDAR
overlays are separate opt-in layers.

## Quick start

```bash
uv run t4e2e visualize /path/to/t4/scene/date/time \
  --root /path/to/t4 \
  --center 100 \
  --mode bev \
  --no-lidar \
  --out results/visualization/window.png

uv run t4e2e visualize /path/to/t4/scene/date/time \
  --root /path/to/t4 \
  --center 100 \
  --mode summary \
  --no-lidar \
  --out results/visualization/summary.png
```

The equivalent Python API is intentionally small:

```python
from t4_e2e_devkit.dataset.window import T4WindowBuilder
from t4_e2e_devkit.visualization import plot_bev_frame, save_figure

builder = T4WindowBuilder(scene_dir, dataset_root)
try:
    scene = builder.build(builder.valid_centers()[0])
    figure, _ = plot_bev_frame(
        scene,
        {"ground_truth": scene.get_future_trajectory()},
        {"legend": True},
    )
    save_figure(figure, "results/visualization/window.png")
finally:
    builder.close()
```

`Trajectory` inputs preserve their declared sampling grid. The renderer does
not convert all trajectories to one point count.

## What each trajectory means

`plot_bev_frame` accepts these built-in keys:

| Key | Owner | Rendering |
| --- | --- | --- |
| `history` | Recorded ego history | Dashed line |
| `ground_truth` | Recorded ego future | Blue-to-red time-colored points |
| `prediction` | Model-planned ego trajectory | Solid orange line |

The `ground_truth` and `prediction` entries are both ego trajectories. They are
not trajectories of other agents. Other-agent future traces are generated from
privileged future annotations and are labeled `other-agent future` when a
dynamic scene provides them. They are purple point traces, not model output.

The arrows attached to object boxes show the recorded current velocity of that
object. They are not additional trajectory predictions. The ego vehicle is
drawn as the footprint at the origin, with its scene-specific length and width;
it is not hidden behind the trajectory layer.

## Available views and functions

| Function | Purpose |
| --- | --- |
| `plot_bev_frame` | Map, ego footprint, agents, goal and trajectory roles |
| `plot_scene_summary` | BEV plus decoded camera views when available |
| `plot_cameras_frame` | Camera images with boxes and optional projections |
| `plot_bev_with_score` | BEV beside PDM metric components |
| `plot_agent_comparison` | Several ego trajectories in one scene |
| `reference_trajectories` | Recorded ego history and future |
| `render_prediction_bev` | Lightweight map-and-trajectory image for callbacks |

The deterministic rich sample is [`assets/bev_sample.png`](assets/bev_sample.png).
It exercises lane boundaries, route lanes, signal states, road markings,
multiple agent classes, other-agent future traces, the ego footprint, goal and
the fixed semantic legend. It is map-and-annotation only; no LiDAR is read.

Regenerate it from the repository root:

```bash
MPLCONFIGDIR=/tmp/mplconfig uv run python docs/examples/generate_bev_sample.py
```

## Map and object layers

The current frame uses ego coordinates (`x` forward, `y` left). The BEV renderer
draws:

- lane centerlines and boundaries reconstructed from the stored offsets;
- route lanes and their traffic-light state;
- polygons and line strings, including road borders and markings;
- the ego footprint using the dimensions stored in the scene;
- tracked boxes for cars, trucks, buses, bicycles and pedestrians;
- heading ticks and velocity arrows for tracked objects;
- recorded future traces for neighboring agents when available;
- the destination marker and vehicle status text.

Missing map or annotation fields are not replaced with geometry at the origin.
Required fields raise a clear error; a scene without optional future data simply
omits that layer.

## Stable legends

Enable `legend=True` or call `add_fixed_bev_legend` directly. The legend uses an
explicit semantic vocabulary rather than the artists visible in one frame, so
its content and layout stay fixed across a video:

- ego history, ego ground truth and ego prediction;
- the ego footprint;
- all five T4 tracked classes, including classes absent from the current view;
- the goal pose when the scene provides one.

This prevents the lower-right label box from changing size or contents as
traffic enters and leaves the frame.

## Camera projection

Camera images must be decoded through the scene builder so calibration and
image resolution agree. Projection applies camera-to-ego extrinsics,
intrinsics and stored distortion coefficients. Trajectories use the same ego
coordinates as the BEV.

The public camera input supports the road-facing channels a rig exports as one
JPEG per frame; video-backed and roof channels are rejected by the current
contract. The rigs differ -- `wide5` on the main prd_jt rig, `x2_surround6` on
x2_dev -- so a plot resolves its register against the scene rather than assuming
one.

LiDAR is opt-in:

```python
from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene

# "auto" is right here: the question is what this scene has, not what a
# checkpoint was trained on.
sensor_config = sensor_config_for_scene(scene_dir, "auto", lidar=True)
```

Use `SensorConfig.build_no_sensors()` for map-only BEV plots, or set
`lidar=False` for camera-only input. A no-LiDAR BEV still contains the complete
map, ego, annotations, trajectory and legend layers.
