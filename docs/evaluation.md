# Evaluation

The devkit keeps evaluation families independent so their inputs and
aggregates cannot be confused:

| Family | Measures |
| --- | --- |
| `open_loop` | Prediction error against the recorded ego future |
| `pdm` | Simulated safety, progress, compliance and comfort |
| `closed_loop` | Sensor-replay rollout and realized ego behavior |

The PDM version is a formula version, not a separate model or result family.
The default official version is v2 with the complete metric set.

## PDM scoring

```python
from t4_e2e_devkit.evaluation import T4NavSimScorer, T4NavSimScorerConfig

scorer = T4NavSimScorer(
    T4NavSimScorerConfig(version="v2", backend="auto")
)
result = scorer.score(trajectory, scene)
print(result.values)  # complete v2 result
```

`v1` produces the original score fields. `v2` adds lane keeping, history
comfort and extended comfort. Extended comfort compares consecutive plans; the
first window in a sequence has no value for that term and aggregation uses the
available terms. Set `require_extended_comfort=True` when every result must
contain it.

`metric_names` is optional. Omit it for every metric in the selected version;
pass an ordered tuple when a caller intentionally needs only selected fields:

```python
scorer = T4NavSimScorer(
    T4NavSimScorerConfig(
        version="v2",
        metric_names=("ego_progress", "score"),
        backend="gpu",
    )
)
```

Requesting `score` computes its dependencies internally, but those dependencies
are not added to `result.values`. An unavailable extended-comfort value is
omitted, never replaced with zero. Diagnostic coverage and availability flags
are stored in `result.metadata`.

The scorer accepts any uniform trajectory grid that covers the four-second
evaluation horizon:

| Input | Evaluation |
| --- | --- |
| 80 poses at 0.1 s (8 s) | First 4 s, resampled |
| 8 poses at 0.5 s (4 s) | Direct |
| Another declared grid covering 4 s | Interpolated |
| A grid shorter than 4 s | Rejected |

Headings are interpolated after unwrapping. The current pose at `t=0` is an
implicit knot.

For training-time diagnostics, `score_proposals` accepts `[B, N, T, 3]` or
`[B, N, T, 4]` proposals and returns `T4NavSimProposalResult`. Its `.values`
tensor has shape `[B, N, len(metric_names)]`, and `.metric_names` defines the
column order. The result is detached and is not part of the differentiable
model loss.

## Backends

`gpu` batches geometry and simulation on CUDA. `cpu` is the scalar reference
implementation for audit runs and machines without CUDA. An explicit GPU
request fails when CUDA is unavailable; it never silently falls back to CPU.
`auto` selects GPU when available and CPU otherwise.

## Open-loop metrics

```python
from t4_e2e_devkit.evaluation import compute_open_loop_metrics

metrics = compute_open_loop_metrics(prediction, scene)
```

The comparison uses the declared trajectory intervals and their common horizon.
The report includes ADE, FDE, heading error, miss rate, horizon and pose count.

## Prediction manifests

Model repositories can exchange predictions through the model-neutral
`t4-e2e.predictions` JSONL format. The header declares `num_poses`,
`interval_seconds` and `pose_format`; each row is keyed by `scene` and
`center` and contains current-ego-frame `(x, y, heading)` poses. The writer
stores a data-list content hash, never a local data-list or checkpoint path.
The scorer validates an exact one-to-one key match before reading any scene.

```python
from t4_e2e_devkit.evaluation import PredictionManifestWriter

with PredictionManifestWriter(
    "results/predictions.jsonl",
    data_list="results/val.datalist.json",
    num_poses=8,
    interval_seconds=0.5,
) as writer:
    writer.write("prd_jt/scene", 100, trajectory)
```

Score the manifest without importing the model repository:

```bash
uv run t4e2e score-manifest results/val.datalist.json \
  --predictions results/predictions.jsonl \
  --output-dir results/score \
  --version v2 \
  --backend auto
```

Omit `--metrics` for the complete v2 result. Pass `--metrics` only for an
explicit ordered subset. `--shard-index` and `--num-shards` score disjoint
deterministic row partitions; the output records metric counts for a caller
that combines shard aggregates.

Sampling belongs to each manifest, so dense and sparse model outputs use the
same scoring entry point. The full integration workflow is documented in
[`integration.md`](integration.md).

## Batch evaluation

```bash
uv run t4e2e evaluate results/val.datalist.json \
  --agent my_agent \
  --output-dir results/evaluation \
  --families open_loop pdm \
  --pdm-version navsim-v2 \
  --backend auto
```

Each row produces a privacy-safe JSON record under `records/`. The run also
writes family CSV files, `aggregate.json`, `aggregate.yaml`, `run.json`, a
worker manifest and `failures.csv`. Generated result directories are ignored
by Git.

`--pdm-version` accepts `navsim-v1` or `navsim-v2`; these names identify the
formula version. Use `--pdm-metrics` only when a run intentionally selects a
subset. Omitting it uses the complete set for the selected version.

## Sharding, workers and resume

Ranks deterministically partition data-list rows:

```bash
uv run t4e2e evaluate results/val.datalist.json --agent my_agent \
  --output-dir results/evaluation/rank-0 \
  --rank 0 --world-size 4 --workers 1

uv run t4e2e merge-evaluation \
  --input-dir results/evaluation/rank-0 results/evaluation/rank-1 \
               results/evaluation/rank-2 results/evaluation/rank-3 \
  --output-dir results/evaluation/merged
```

`--workers` controls serial, thread, process or Ray execution inside one rank.
`--rank/--world-size` are the only partitioning inputs; no scheduler or
tracking service is required. `--resume` reuses a row only when its token and
resolved configuration fingerprint match. Merging rejects missing or
duplicate ranks, duplicate tokens and stale records. Failed rows remain
visible and make the run unsuccessful.

## Closed loop

`T4ClosedLoopRunner` performs sensor replay with a kinematic perfect tracker.
At each source tick it calls the agent, converts the plan to the source grid,
advances the ego state and feeds the realized state into the next tick. It does
not re-render sensors. Recorded traffic is replayed by default; constant
velocity and IDM policies are available for controlled experiments.

```bash
uv run t4e2e evaluate-closed-loop results/val.datalist.json \
  --agent my_agent \
  --output-dir results/closed-loop
```

Closed-loop metrics and rollout artifacts stay in their own report section and
are merged with `t4e2e merge-closed-loop`.

## Library engine

```python
from t4_e2e_devkit.evaluation import MetricContext, MetricEngine

engine = MetricEngine.t4_default()
report = engine.evaluate(
    MetricContext(
        token=token,
        prediction=prediction,
        ground_truth=scene,
        pdm_version="navsim-v2",
    ),
    families=("pdm",),
)
```

`MetricCache` stores metric outputs only. Its signature includes the
prediction, scene arrays, consecutive-plan inputs and metric version.
