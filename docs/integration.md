# Model integration

The devkit exposes two integration paths. Both produce the same trajectory
contract and v2 evaluation fields:

| Path | Use when | Model repository responsibility |
| --- | --- | --- |
| Prediction manifest | Inference already has its own runtime | Export keyed local-frame trajectories |
| Registered agent | The devkit should load scenes and call inference | Implement `AbstractT4Agent` |

The manifest path has the smallest dependency surface and is the recommended
choice for an existing training or inference repository.

## 1. Install the devkit

```bash
uv sync
```

The data is not packaged with the repository. Build a local data list once and
keep it in an ignored runtime directory:

```bash
uv run t4e2e datalist \
  --root /path/to/t4 \
  --glob 'prd_jt/*/*/*' \
  --out results/val.datalist.json
```

The data list is the evaluation boundary. It contains relative scene keys and
center frames, plus the local dataset root needed by the reader. The root is
not portable and should not be committed.

## 2. Export a prediction manifest

Write one JSONL header followed by one row for every data-list key. The public
format is:

```json
{"format":"t4-e2e.predictions","version":1,"trajectory":{"num_poses":8,"interval_seconds":0.5,"pose_format":"x_y_heading"}}
{"scene":"prd_jt/example-scene","center":100,"poses":[[1.0,0.0,0.01],[2.0,0.1,0.02],[3.0,0.2,0.03],[4.0,0.3,0.04],[5.0,0.4,0.05],[6.0,0.5,0.06],[7.0,0.6,0.07],[8.0,0.7,0.08]]}
```

The writer adds the data-list content hash when the data-list path is
provided. A prediction row contains only:

- `scene`: the relative scene key from the data list;
- `center`: the integer center frame;
- `poses`: `[T, 3]` values in `(x, y, heading)` form.

Coordinates are in the current ego frame: `x` points forward, `y` points left,
and heading is in radians. Pose `i` is sampled at
`(i + 1) * interval_seconds`; the current pose at `t=0` is implicit. A model
may emit `[T, 4]` values in memory, where the last two columns are
`cos(heading), sin(heading)`; convert them before writing or use
`trajectory_to_poses`.

The declared grid is part of the manifest. `80 × 0.1 s` and `8 × 0.5 s` are
both valid inputs. The evaluator resamples uniform grids that cover the
four-second scoring horizon and rejects shorter trajectories. Point count
alone never determines time.

The repository contains a NumPy conversion example:

```bash
uv run python examples/write_prediction_manifest.py \
  --data-list results/val.datalist.json \
  --input results/model_predictions.npz \
  --output results/model/predictions.jsonl \
  --interval-seconds 0.5
```

The input archive must contain `scene [N]`, `center [N]` and `poses [N,T,3|4]`.
The example validates duplicate keys, data-list coverage and finite values.

## 3. Score the manifest

```bash
uv run t4e2e score-manifest results/val.datalist.json \
  --predictions results/model/predictions.jsonl \
  --output-dir results/model/score \
  --version v2 \
  --backend auto
```

The default is the complete v2 metric set. Pass `--metrics` only when a caller
intentionally wants an ordered subset:

```bash
uv run t4e2e score-manifest results/val.datalist.json \
  --predictions results/model/predictions.jsonl \
  --output-dir results/model/score-selected \
  --version v2 \
  --metrics ego_progress score
```

`auto` selects the GPU backend when CUDA is available and otherwise uses the
CPU reference implementation. `gpu` is strict and fails when CUDA is not
available. The report contains `aggregate.json` and, unless disabled,
`per_window.csv`. It records content hashes and sampling metadata rather than
local file paths.

For independent workers, give every worker the same manifest and data list and
use disjoint deterministic shards:

```bash
uv run t4e2e score-manifest results/val.datalist.json \
  --predictions results/model/predictions.jsonl \
  --output-dir results/model/score-rank-0 \
  --version v2 --backend gpu \
  --shard-index 0 --num-shards 4
```

The caller can combine the shard aggregates using the recorded metric counts.
For rank launching, retries, resume and validated aggregate merging, use the
registered-agent workflow described below.

## 4. Register an agent

Implement the public interface when the devkit should own scene loading and
inference:

```python
from t4_e2e_devkit.agents import AbstractT4Agent, register_agent
from t4_e2e_devkit.common.dataclasses import SensorConfig
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


class MyAgent(AbstractT4Agent):
    @property
    def trajectory_sampling(self):
        return TrajectorySampling(num_poses=8, interval_length=0.5)

    def name(self):
        return "my_agent"

    def get_sensor_config(self):
        return SensorConfig.build_current_frame(lidar=False)

    def forward(self, features):
        return {"trajectory": self.network(features)}


register_agent("my_agent", MyAgent)
```

The complete training surface, including feature and target builders, is in
[`agents.md`](agents.md). A deployable agent receives `T4AgentInput` only;
future labels are available to target builders and evaluation orchestration,
not to inference.

External packages can register an entry point without becoming a dependency of
the devkit:

```toml
[project.entry-points."t4_e2e_devkit.agents"]
my_agent = "my_package.agent:MyAgent"
```

Run the agent through the shared evaluator:

```bash
uv run t4e2e evaluate results/val.datalist.json \
  --agent my_agent \
  --output-dir results/my_agent/evaluation \
  --families open_loop pdm \
  --pdm-version navsim-v2 \
  --backend auto
```

Use `t4e2e distribute evaluate` for multi-rank execution. The evaluator keeps
open-loop, PDM and closed-loop outputs in separate report sections; the PDM
version selects the formula version, not a model category.

## 5. Training-time scoring and visualization

The official scoring callback is logger-neutral. During validation, expose
these Lightning-module attributes for each prediction:

```python
module._official_score_predictions.append(prediction)  # [T, 3] or [T, 4]
module._official_score_keys.append((scene, center_frame))
```

Then add one callback to the trainer:

```python
from t4_e2e_devkit.planning.training import OfficialDevkitScoreCallback

callbacks.append(
    OfficialDevkitScoreCallback(
        data_list="results/val.datalist.json",
        output_dir="results/training/devkit-score",
        version="v2",
        metric_names=None,
        interval_seconds=0.5,
    )
)
```

The callback validates and merges rank predictions, scores them on each rank's
GPU, writes a single aggregate and sends stable scalar keys through any
Lightning logger that implements `log_metrics`. It does not import a tracking
client. The default logged keys are `devkit/pdms`,
`devkit/nc`, `devkit/dac`, `devkit/ddc`, `devkit/tlc`, `devkit/ttc`,
`devkit/ep`, `devkit/lk`, `devkit/comfort` and `devkit/ec` when those fields are
available.

For validation images, use `PredictionVizCallback` and expose up to the
configured number of samples in `module._viz_samples`. Each sample contains
`gt_xy` and `pred_xy`, with optional `lanes` and `route` arrays. The callback
calls a generic logger `log_image` method with the stable key
`val/bev_trajectory`; it owns rendering, not experiment tracking.

## 6. Output and privacy

Keep scenes, checkpoints, manifests, images and reports under ignored
`results/` or `reports/` directories. The public manifest and reports contain
relative keys, content hashes and numeric outputs. Do not add local dataset
roots, scene-tag files, checkpoint paths or private metadata to source files or
documentation.
