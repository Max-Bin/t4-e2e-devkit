# Agents

An agent declares its sensors, feature builders, target builders, prediction
sampling, loss and optimizer. The dataset, evaluator and visualizer use the
same interface for every agent.

## Minimal interface

```python
import torch

from t4_e2e_devkit.agents import (
    AbstractT4Agent,
    CameraFeatureBuilder,
    MapFeatureBuilder,
    TrajectoryTargetBuilder,
)
from t4_e2e_devkit.common.dataclasses import SensorConfig
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


class MyAgent(AbstractT4Agent):
    @property
    def trajectory_sampling(self):
        return TrajectorySampling(num_poses=80, interval_length=0.1)

    def name(self):
        return "my_agent"

    def get_sensor_config(self):
        return SensorConfig.build_current_frame(lidar=False)

    def get_feature_builders(self):
        return [CameraFeatureBuilder(), MapFeatureBuilder()]

    def get_target_builders(self):
        return [TrajectoryTargetBuilder(trajectory_sampling=self.trajectory_sampling)]

    def forward(self, features):
        return {"trajectory": self.network(features)}  # [B, 80, 3]

    def compute_loss(self, features, targets, predictions):
        return torch.nn.functional.smooth_l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
```

`trajectory_sampling` is part of the agent contract. A prediction must have
`[B, sampling.num_poses, 3]`; the inference wrapper returns a `Trajectory`
with the same sampling. The current pose at `t=0` is implicit. Pose `i` is at
`(i + 1) * interval_length` seconds.

For a different grid, change only the sampling declaration. For example:

```python
TrajectorySampling(num_poses=8, interval_length=0.5)  # 4 s
TrajectorySampling(num_poses=80, interval_length=0.1)  # 8 s
```

The target builder samples the recorded future on that grid. It checks the
available horizon and never infers time from the number of points alone.

## Registration

Register an in-process agent:

```python
from t4_e2e_devkit.agents import register_agent

register_agent("my_agent", MyAgent)
```

An external package can expose the same constructor without becoming a runtime
dependency of this repository:

```toml
[project.entry-points."t4_e2e_devkit.agents"]
my_agent = "my_package.agent:MyAgent"
```

## Builders

Feature builders receive `T4AgentInput`, which contains no future data. Target
builders receive `T4Scene`, which contains the recorded future.

Built-in builders include:

- `EgoStatusFeatureBuilder`
- `CameraFeatureBuilder`
- `LidarFeatureBuilder`
- `MapFeatureBuilder`
- `TrajectoryTargetBuilder`
- `OracleTargetBuilder`

LiDAR is opt-in. A camera-only agent should set `lidar=False`; a map-only agent
can use `SensorConfig.build_no_sensors()`.

## Training-time PDM reporting

`score_proposals` accepts raw tensors and a sampling declaration:

```python
components = scorer.score_proposals(
    proposals,  # [B, N, P, 3]
    scenes,
    metric_names=("ego_progress", "score"),
    trajectory_sampling=agent.trajectory_sampling,
)
```

The scorer adapts every proposal to its configured evaluation grid before
scoring. A trajectory shorter than that grid is rejected. The result has a
detached `.values` tensor and an explicit `.metric_names` column order; it is
not part of the differentiable agent loss. `score_batch` gets sampling directly
from each `Trajectory`, so it does not need producer-specific configuration.

## Oracles and deployment

An oracle that needs future data sets `requires_scene=True` and implements
`compute_trajectory_from_scene(scene)`. A deployable agent must plan from
`T4AgentInput` only. `compute_control` is optional for one-step actuator output.
