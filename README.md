# t4-e2e-devkit

A model-neutral dataset, planning, evaluation and visualization toolkit for
T4 scenes. It provides one contract for scene windows, sensor inputs,
trajectories, scoring and local reports.

```text
T4 scene files -> data list -> T4Scene -> model -> prediction manifest
                                      └── evaluator / visualizer
```

The repository supports the T4 scene format only. It does not require a
NuPlan database or an experiment-tracking service. Evaluation returns ordinary
Python dictionaries and local files, so a caller may send the results to any
logger or dashboard.

## Public entry points

| Capability | Entry point | Result |
| --- | --- | --- |
| Build a data list | `t4e2e datalist` | Reproducible evaluation windows |
| Inspect a data list | `t4e2e inspect` | Window count, root and filters |
| Render one window | `t4e2e visualize` | BEV, camera or summary image |
| Render planning videos | `t4e2e visualize-video` | Per-scene mp4 with prediction overlays |
| Score external predictions | `t4e2e score-manifest` | v2 metrics from JSONL predictions |
| Run a registered agent | `t4e2e evaluate` | Open-loop and PDM reports |
| Run sensor-replay closed loop | `t4e2e evaluate-closed-loop` | Kinematic rollout and metrics |
| Score during training | `OfficialDevkitScoreCallback` | Distributed GPU scoring and generic logger output |

LiDAR is opt-in. Map-only and camera-only visualization do not decode it. The
public camera path currently supports the configured wide, JPEG-backed camera
channels; narrow and video-backed channels are not part of the input contract.

## Install and verify

Python 3.10–3.12 is supported. The recommended development workflow uses
`uv`:

```bash
uv sync
uv run t4e2e check
uv run pytest -q
uv run ruff check .
```

Install optional dependencies only when they are needed:

```bash
uv sync --extra camera
uv sync --extra lidar
```

The repository does not contain scenes, maps, scene tags, checkpoints or
generated reports. Keep data lists, manifests, images and reports under an
ignored `results/` or `reports/` directory.

## Quick start

### 1. Build and inspect a data list

```bash
uv run t4e2e datalist \
  --root /path/to/t4 \
  --glob 'prd_jt/*/*/*' \
  --out results/val.datalist.json

uv run t4e2e inspect results/val.datalist.json
```

A data list fixes the `(scene, center_frame)` windows and records the policy
used to construct them. Evaluation does not rediscover windows by scanning the
dataset at runtime.

### 2. Render one BEV frame

```bash
uv run t4e2e visualize /path/to/t4/prd_jt/scene/date/time \
  --root /path/to/t4 \
  --center 100 \
  --mode bev \
  --no-lidar \
  --out results/visualization/window.png
```

Use `--mode cameras` for camera panels or `--mode summary` for a combined
view. The Python equivalent is [`examples/render_bev.py`](examples/render_bev.py).
The deterministic synthetic sample is [`docs/assets/bev_sample.png`](docs/assets/bev_sample.png);
regenerate it with:

```bash
uv run python docs/examples/generate_bev_sample.py
```

### 3. Score predictions from any model repository

A model repository can stay independent of the devkit and export the shared
`t4-e2e.predictions` JSONL manifest:

```bash
uv run t4e2e score-manifest results/val.datalist.json \
  --predictions results/model/predictions.jsonl \
  --output-dir results/model/score \
  --version v2 \
  --backend auto
```

Omitting `--metrics` computes the complete v2 metric set. Use
`--backend gpu` to require CUDA or `--backend cpu` for the reference path.
The command writes `aggregate.json` and, by default, `per_window.csv`.
[`examples/write_prediction_manifest.py`](examples/write_prediction_manifest.py)
converts neutral NumPy outputs to the shared format.

### 4. Let the devkit call a registered agent

Implement `AbstractT4Agent`, declare its sensors and trajectory sampling, and
register it through the package entry-point group:

```bash
uv run t4e2e evaluate results/val.datalist.json \
  --agent my_agent \
  --output-dir results/my_agent/evaluation \
  --families open_loop pdm \
  --pdm-version navsim-v2 \
  --backend auto
```

Use this path when the model can be installed into the evaluation environment.
Use the prediction-manifest path when the model owns its inference runtime.
Both paths use the same trajectory and scoring contracts.

## Trajectory contract

Every trajectory declares its sampling grid; time is never inferred from the
number of points:

```python
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

TrajectorySampling(num_poses=80, interval_length=0.1)  # 8 seconds
TrajectorySampling(num_poses=8, interval_length=0.5)   # 4 seconds
```

Poses use the current ego frame as `(x, y, heading)`: `x` is forward, `y` is
left and heading is in radians. The current pose at `t=0` is implicit; pose
`i` is at `(i + 1) * interval_length`. The scorer resamples any uniform grid
that covers its four-second horizon and rejects shorter trajectories. The
visualizer preserves the source grid.

## Distributed evaluation

For a small run, use `evaluate` directly. For larger runs, partition rows by
rank and merge the portable reports:

```bash
uv run t4e2e evaluate results/val.datalist.json \
  --agent my_agent \
  --output-dir results/evaluation/rank-0 \
  --rank 0 --world-size 4

uv run t4e2e merge-evaluation \
  --input-dir results/evaluation/rank-0 results/evaluation/rank-1 \
               results/evaluation/rank-2 results/evaluation/rank-3 \
  --output-dir results/evaluation/merged
```

`--rank/--world-size` define deterministic data sharding. `--workers` controls
local execution within one rank. `distribute` adds rank launching, retries,
resume and merge orchestration without requiring a scheduler or tracking
service.

## Documentation

| Document | Contents |
| --- | --- |
| [`examples/README.md`](examples/README.md) | Copyable evaluation and BEV examples |
| [`docs/integration.md`](docs/integration.md) | Integration routes, manifests, agents and training callbacks |
| [`docs/data_contract.md`](docs/data_contract.md) | Scene layout, tensors, sensors and time ranges |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics, sampling, backends and distributed scoring |
| [`docs/visualization.md`](docs/visualization.md) | BEV, camera projection, legend and trajectory roles |
| [`docs/agents.md`](docs/agents.md) | Agent, feature-builder and target-builder interfaces |
| [`docs/closed_loop.md`](docs/closed_loop.md) | Sensor replay and kinematic closed loop |
| [`docs/runtime.md`](docs/runtime.md) | Feature cache and local execution |
| [`docs/internal_runtime.md`](docs/internal_runtime.md) | Orchestration, submissions and configuration |
| [`docs/architecture.md`](docs/architecture.md) | Module boundaries and ownership |
| [`docs/environment.md`](docs/environment.md) | Environments, data checks and optional dependencies |

Keep generated artifacts in ignored runtime directories. Do not publish local
dataset roots, scene-tag files, checkpoint paths or other machine-specific
metadata in source files or public documentation.
