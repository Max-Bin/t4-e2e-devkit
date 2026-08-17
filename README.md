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

uv run t4e2e score \
  agent=constant_velocity \
  data_list=lists/val.json \
  experiment_name=baseline \
  backend=cpu

uv run t4e2e visualize \
  /path/to/t4_dataset/prd_jt/scene \
  --mode bev \
  --out window.png

uv run t4e2e evaluate-closed-loop \
  lists/val.json \
  --agent constant_velocity \
  --output-dir reports/closed_loop
```

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
| [`docs/visualization.md`](docs/visualization.md) | BEV, cameras and sample image |
| [`docs/vendor_audit.md`](docs/vendor_audit.md) | vendored TODO and provenance audit |
| [`docs/migration.md`](docs/migration.md) | integrating an existing agent |
| [`docs/environment.md`](docs/environment.md) | setup and validation |

The rich BEV sample is [`docs/assets/bev_sample.png`](docs/assets/bev_sample.png).
