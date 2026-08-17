# Visualization

The renderer draws a single current-frame BEV from the scene's map tensors,
annotations and trajectories. It does not require LiDAR. Camera and LiDAR
overlays are optional.

## Quick start

```bash
uv run t4e2e visualize <scene_dir> --mode bev --out window.png
uv run t4e2e visualize <scene_dir> --mode summary --out window.png
```

```python
from t4_e2e_devkit.dataset.window import T4WindowBuilder
from t4_e2e_devkit.visualization import plot_bev_frame, save_figure

builder = T4WindowBuilder(scene_dir, root)
scene = builder.build(builder.valid_centers()[0])
figure, _ = plot_bev_frame(
    scene,
    {"prediction": prediction, "ground_truth": scene.get_future_trajectory()},
)
save_figure(figure, "window.png")
builder.close()
```

`prediction` may be a `Trajectory` on any valid time grid. Visualization keeps
that grid; it does not convert every trajectory to a common point count.

## Available views

| function | purpose |
|---|---|
| `plot_bev_frame` | map, ego footprint, agents, goal and trajectories |
| `plot_scene_summary` | BEV plus decoded camera views when available |
| `plot_cameras_frame` | camera images with optional boxes, LiDAR and trajectory projection |
| `plot_bev_with_score` | BEV beside PDM components |
| `plot_agent_comparison` | several trajectories on one scene |
| `reference_trajectories` | recorded history, GT and available reference paths |

The rich synthetic sample is [bev_sample.png](assets/bev_sample.png). It
contains lane boundaries, route lanes, signal states, road markings, agents of
different classes, neighbour future traces, the ego footprint, goal and
multiple trajectory roles. It is map-and-annotation only; no LiDAR is used. The
legend identifies the ego GT future, other-agent history/future and class
colours. Regenerate it with:

```bash
MPLCONFIGDIR=/tmp/mplconfig uv run python docs/examples/generate_bev_sample.py
```

## Trajectory roles

`plot_bev_frame` accepts these built-in keys:

| key | source | rendering |
|---|---|---|
| `history` | recorded ego history | dashed line |
| `ground_truth` | recorded ego future | time-coloured points |
| `prediction` | planned trajectory | solid line |
| `pdm_reference` | optional reference path | dash-dot line |

Each role has a distinct style. A custom trajectory can be supplied as a
`Trajectory` or an `[N, 2+]` array; arrays use the default plotting interval and
cannot carry sampling metadata.

## Map and objects

The renderer uses the current frame's ego coordinates (`x` forward, `y` left).
It draws:

- lane centerlines and reconstructed left/right boundaries;
- route lanes and their traffic-light state;
- polygons and line strings, including road borders;
- the ego footprint with its scene-specific length and width;
- tracked object boxes, class colours, heading ticks and optional future traces;
- tracked-object velocity arrows in the same colour as their boxes;
- the destination marker and vehicle status text.

When a legend is enabled, its semantic entries are fixed across frames: both
trajectory roles and all five T4 agent classes remain present even when a class
is temporarily outside the view. This keeps BEV videos from resizing or
changing their legend as traffic changes.

Missing map or annotation fields are not replaced with geometry at the origin.
The reader raises for missing required fields, while a scene without optional
future data simply omits that layer.

## Camera projection

Camera images must be decoded through the scene builder so their calibration and
image resolution agree. The projection path applies camera-to-ego extrinsics,
intrinsics and any stored distortion coefficients. A trajectory is projected
from the same ego-frame coordinates as the BEV.

LiDAR is opt-in at construction time:

```python
sensor_config = SensorConfig.build_current_frame(lidar=True)
```

Leave it disabled for camera-only or map-only plots.
