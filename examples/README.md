# Integration examples

These examples use only public devkit interfaces. They do not assume a model
framework, experiment tracker or scheduler. Replace the placeholder paths with
paths in your own environment; do not copy dataset paths into source files.

## 1. Install

```bash
uv sync
```

The repository does not include T4 scenes. The commands below expect an
existing T4 dataset root and a writable ignored `results/` directory.

## 2. Build and inspect a data list

```bash
uv run t4e2e datalist \
  --root /data/t4 \
  --glob 'prd_jt/*/*/*' \
  --camera-names wide5 \
  --out results/val.datalist.json

uv run t4e2e inspect results/val.datalist.json
```

A data list is the reproducible set of `(scene, center_frame)` windows used by
an evaluation. It also records the dataset root and filtering policy.

## 3. Render one BEV window

The command-line path is enough for most users:

```bash
uv run t4e2e visualize /data/t4/prd_jt/scene/date/time \
  --root /data/t4 \
  --center 100 \
  --mode bev \
  --no-lidar \
  --out results/visualization/window.png
```

The equivalent Python API is in [`render_bev.py`](render_bev.py). Pass a
`[T, 3]` or `[T, 4]` `.npy`/`.npz` prediction to overlay a model trajectory
without registering an agent:

```bash
uv run python examples/render_bev.py \
  --scene /data/t4/prd_jt/scene/date/time \
  --root /data/t4 \
  --center 100 \
  --prediction results/model/prediction.npy \
  --prediction-interval 0.5 \
  --out results/visualization/prediction.png
```

## 4. Export a model's predictions

The devkit boundary is a small JSONL manifest. A model repository can write it
directly with `PredictionManifestWriter`; the conversion example accepts a
neutral NumPy file when the model already exports arrays:

```bash
uv run python examples/write_prediction_manifest.py \
  --data-list results/val.datalist.json \
  --input results/model_predictions.npz \
  --output results/model/predictions.jsonl \
  --interval-seconds 0.5
```

The input `.npz` must contain:

```text
scene   [N]       relative scene paths from the data list
center  [N]       integer center frames
poses   [N,T,3]   x, y, heading
```

`poses[..., :2]` are metres in the current ego frame; heading is radians. A
`[N,T,4]` array is also accepted when its last two values are `cos(heading)`
and `sin(heading)`. The declared interval, not `T` alone, defines time.

## 5. Score without importing the model

```bash
uv run t4e2e score-manifest results/val.datalist.json \
  --predictions results/model/predictions.jsonl \
  --output-dir results/model/score \
  --version v2 \
  --backend auto
```

Omit `--metrics` for the complete v2 result. Use `--backend gpu` to require
CUDA or `--backend cpu` for the reference path. The output contains
`aggregate.json` and, unless disabled, `per_window.csv`.

For independent workers, use matching `--shard-index` and `--num-shards`.
Each worker validates the complete manifest before scoring its deterministic
row partition. Merge the resulting reports in the calling job, or use the
higher-level `evaluate`/`distribute` workflow when the model is registered as
an agent.

## 6. Render a planning video

Once a manifest exists, replay it against the recorded future frame by frame.
One mp4 per scene, camera and BEV side by side; `--manifest` is repeatable, so
two models can be compared in one video, and omitting it replays the ground
truth alone. Requires the `ffmpeg` binary on `PATH`:

```bash
uv run t4e2e visualize-video \
  --data-list results/val.datalist.json \
  --scene prd_jt/scene/date/time \
  --manifest baseline=results/model/predictions.jsonl \
  --out results/visualization/videos
```

The equivalent Python API is in
[`render_planning_video.py`](render_planning_video.py).

## 7. Register an agent instead

If the model should be invoked by the devkit, implement
`AbstractT4Agent`, declare its `SensorConfig` and `TrajectorySampling`, then
register it. The minimal contract is documented in [`docs/agents.md`](../docs/agents.md).

```bash
uv run t4e2e evaluate results/val.datalist.json \
  --agent my_agent \
  --output-dir results/my_agent/evaluation \
  --families open_loop pdm \
  --pdm-version navsim-v2 \
  --backend auto
```

Use this route when the devkit should own scene loading and inference. Use the
manifest route when the model repository already owns inference or its
environment is separate.

## 8. Training integration

Lightning-based training code can use the shared callback directly:

```python
from t4_e2e_devkit.planning.training import OfficialDevkitScoreCallback

callbacks.append(
    OfficialDevkitScoreCallback(
        data_list="results/val.datalist.json",
        output_dir="results/training/devkit-score",
        version="v2",
        metric_names=None,  # complete official evaluation
        interval_seconds=0.5,
    )
)
```

During validation, expose each model's local ego trajectory and its
`(scene_dir, center_frame)` key to the callback contract. The callback handles
manifest construction, DDP rank merging, GPU scoring and logger-neutral scalar
logging. The devkit itself does not import a tracking client.

## Output and privacy

Keep scenes, checkpoints, generated manifests, images and reports outside Git,
under ignored `results/` or `reports/` directories. Prediction manifests store
data-list content hashes rather than local absolute paths. Do not paste local
dataset roots, scene tags or checkpoint paths into examples or documentation.
