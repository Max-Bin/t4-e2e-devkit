# t4-e2e-devkit

A shared dataset, agent, evaluation and visualization interface for end-to-end
planning on T4 scenes.

```text
scene files -> T4Scene -> T4AgentInput -> agent -> Trajectory -> scorer
                 \-> target builders                         \-> PDMResults
```

The reader, contract and scorer are shared. An agent contributes only its
sensor declaration, feature builders, target sampling, network, loss and
optimizer.

## Install

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run t4e2e check --vendor
```

Optional dependencies:

```bash
uv sync --extra camera
uv sync --extra lidar
```

See [`docs/environment.md`](docs/environment.md) for dataset and CUDA tests.

## Quick start

```bash
uv run t4e2e datalist \
  --root /path/to/t4_dataset \
  --glob 'prd_jt/*/*/*' \
  --out lists/val.json

# Optional semantic scene filtering
uv run t4e2e datalist \
  --root /path/to/t4_dataset \
  --scene-tags-root /path/to/scene_tags \
  --include-tag-event lane_change \
  --out lists/lane-change.json

uv run t4e2e evaluate lists/val.json \
  --agent my_agent \
  --output-dir results/evaluation \
  --families open_loop pdm tier4 \
  --backend auto

# Optional offline PDM reference cache and metadata sources
uv run t4e2e evaluate lists/val.json \
  --agent my_agent \
  --output-dir results/evaluation \
  --pdm-reference-cache-dir /path/to/pdm-cache \
  --maps-root /path/to/maps \
  --scene-tags-root /path/to/scene-tags \
  --attach-map-ids

uv run t4e2e visualize \
  /path/to/t4_dataset/prd_jt/scene \
  --mode bev \
  --out results/visualization/window.png

# Run independent ranks, then merge their portable reports.
uv run t4e2e evaluate lists/val.json --agent my_agent \
  --output-dir results/evaluation/rank-0 --rank 0 --world-size 4 --resume
uv run t4e2e merge-evaluation \
  --input-dir results/evaluation/rank-0 results/evaluation/rank-1 \
              results/evaluation/rank-2 results/evaluation/rank-3 \
  --output-dir results/evaluation/merged
uv run t4e2e dashboard results/evaluation/merged \
  --out results/evaluation/merged/dashboard.html

uv run t4e2e evaluate-closed-loop \
  lists/val.json \
  --agent my_agent \
  --output-dir results/closed_loop

uv run t4e2e merge-closed-loop \
  --input-dir results/closed_loop/rank-0 results/closed_loop/rank-1 \
  --output-dir results/closed_loop/merged
```

`evaluate` defaults to `auto`: it selects the GPU scorer when CUDA is available
and otherwise uses the CPU audit scorer. An explicit `--backend gpu` never falls
back; it fails clearly when CUDA is unavailable. `evaluate` writes one record
per row, family CSV files, `aggregate.json` and a rank manifest.
`--rank/--world-size` partition the data list deterministically;
`--workers/--worker-backend` control local execution and `--resume` reuses only
successful records with the same resolved configuration. `merge-evaluation`
validates rank completeness, manifest membership, duplicate tokens and common
configuration before recomputing the aggregate. All generated files belong in
the ignored `results/` or `reports/` directories.

Build a reference cache only when an offline CPU run needs one. GPU evaluation
can generate the reference online.

## Trajectory contract

Every `Trajectory` contains poses and its sampling metadata:

```python
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

TrajectorySampling(num_poses=80, interval_length=0.1)  # 8 s
TrajectorySampling(num_poses=8, interval_length=0.5)   # 4 s
```

The evaluator converts either grid to its configured scoring grid. The
visualizer keeps the original grid. A source trajectory must cover the target
horizon; otherwise the call fails with a clear error.

The default scene ranges are 31 history frames, 80 recorded future frames and a
4-second PDM scoring horizon. These are separate ranges, not one fixed model
shape.

## Add an agent

```python
from t4_e2e_devkit.agents import AbstractT4Agent, MapFeatureBuilder
from t4_e2e_devkit.common.dataclasses import SensorConfig


class MyAgent(AbstractT4Agent):
    def name(self):
        return "my_agent"

    def get_sensor_config(self):
        return SensorConfig.build_current_frame(lidar=False)

    def get_feature_builders(self):
        return [MapFeatureBuilder()]

    def forward(self, features):
        return {"trajectory": self.network(features)}
```

Declare `trajectory_sampling` and implement the remaining training methods as
described in [`docs/agents.md`](docs/agents.md). Register external agents via
the `t4_e2e_devkit.agents` entry-point group.

## Documentation

| document | contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | package boundaries and ownership |
| [`docs/data_contract.md`](docs/data_contract.md) | scene files, tensors and time ranges |
| [`docs/agents.md`](docs/agents.md) | agent interface and builders |
| [`docs/evaluation.md`](docs/evaluation.md) | scoring backends and sampling |
| [`docs/closed_loop.md`](docs/closed_loop.md) | sensor-replay closed-loop rollout |
| [`docs/runtime.md`](docs/runtime.md) | feature cache and local execution |
| [`docs/visualization.md`](docs/visualization.md) | BEV, cameras and sample image |
| [`docs/vendor_audit.md`](docs/vendor_audit.md) | vendored TODO and provenance audit |
| [`docs/migration.md`](docs/migration.md) | integrating an existing agent |
| [`docs/environment.md`](docs/environment.md) | setup and validation |

The rich BEV sample is [`docs/assets/bev_sample.png`](docs/assets/bev_sample.png).
