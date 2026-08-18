# Evaluation

The devkit exposes three independent families: `open_loop`, `pdm` and
`closed_loop`. Their records and aggregates remain separate.

## PDM

`T4NavSimScorer` evaluates the PDM family with either published metric
version:

```python
from t4_e2e_devkit.evaluation import T4NavSimScorer, T4NavSimScorerConfig

scorer = T4NavSimScorer(
    T4NavSimScorerConfig(
        version="v2",
        backend="gpu",
        metric_names=("no_at_fault_collisions", "ego_progress", "score"),
    )
)
result = scorer.score(trajectory, scene)
print(result.values)
```

`v1` produces PDMS. `v2` produces EPDMS and adds lane keeping, history comfort
and extended comfort. Extended comfort compares consecutive plans; the first
window in a sequence has no extended-comfort value and is aggregated over the
remaining terms. Pass `require_extended_comfort=True` when every v2 result
must include it.

`metric_names` is optional. Omit it for the complete version-specific result;
provide an ordered subset when a caller only needs selected components. The
aggregate `score` is explicit: requesting it computes its formula dependencies,
but those dependencies are not added to `result.values`. An unavailable
extended-comfort value is omitted, never filled with zero. `result.metadata`
contains diagnostics such as coverage and availability flags.

Training-time detached diagnostics use the same field at
`scorer.metric_names`.

The scorer accepts any trajectory grid that covers the four-second evaluation
horizon. Sampling is carried by each `Trajectory`, not inferred from its point
count:

| input | evaluation |
| --- | --- |
| 80 poses at 0.1 s (8 s) | first 4 s, resampled |
| 8 poses at 0.5 s (4 s) | direct |
| another declared grid covering 4 s | interpolated |
| a grid shorter than 4 s | rejected |

Headings are interpolated after unwrapping. The current pose at `t=0` is an
implicit knot.

For training-time diagnostics, `score_proposals` accepts `[B, N, T, 3]` and
returns `T4NavSimProposalResult`. Its `.values` tensor has shape
`[B, N, len(.metric_names)]`, so the column order is never implicit. The
training module logs exactly those selected columns.

## Backends

`gpu` batches simulation and geometry on CUDA. `cpu` is the scalar reference
path for audit runs and machines without CUDA. An explicit GPU request raises
when CUDA is unavailable; it never silently falls back to CPU. The CLI's
`auto` backend selects GPU when available and CPU otherwise.

## Open loop

```python
from t4_e2e_devkit.evaluation import compute_open_loop_metrics

metrics = compute_open_loop_metrics(prediction, scene)
```

The comparison uses the declared source intervals and their common horizon.
ADE, FDE, heading errors, miss rate, horizon and pose count are reported.

## Prediction manifest

Model repositories exchange predictions through the model-neutral
`t4-e2e.predictions` JSONL format. The header declares
`num_poses`, `interval_seconds` and `pose_format`; each row is keyed by
`scene` and `center` and contains local `(x, y, heading)` poses. The writer
stores only the data-list content hash, never a local data-list or checkpoint
path. The scorer validates an exact key match before reading any scene.

```python
from t4_e2e_devkit.evaluation import PredictionManifestWriter

with PredictionManifestWriter(
    "results/predictions.jsonl",
    data_list="lists/val.json",
    num_poses=8,
    interval_seconds=0.5,
) as writer:
    writer.write("prd_jt/scene", 100, trajectory)
```

Sampling is per manifest, so dense and sparse model outputs use the same
scoring entry point. Rank processes can score disjoint row ranges with
`shard_index` and `num_shards`; reports retain only content hashes and
sampling metadata.

## Batch evaluation

```bash
uv run t4e2e evaluate /path/to/val.json \
  --agent my_agent \
  --output-dir results/evaluation \
  --families open_loop pdm \
  --pdm-version navsim-v2 \
  --backend auto
```

Each row produces a privacy-safe JSON record under `records/`. The run also
writes family CSV files, `aggregate.json`, `aggregate.yaml`, `run.json`, a
worker manifest and `failures.csv`. Result directories are ignored by Git.

`--pdm-version` accepts `navsim-v1` or `navsim-v2`; these are versions of the
PDM metric, not additional result families. Use `--pdm-metrics` to select a
subset, for example `--pdm-metrics no_at_fault_collisions ego_progress`.

## Sharding, workers and resume

Ranks deterministically partition data-list rows:

```bash
uv run t4e2e evaluate /path/to/val.json --agent my_agent \
  --output-dir results/evaluation/rank-0 \
  --rank 0 --world-size 4 --workers 1

uv run t4e2e merge-evaluation \
  --input-dir results/evaluation/rank-0 results/evaluation/rank-1 \
               results/evaluation/rank-2 results/evaluation/rank-3 \
  --output-dir results/evaluation/merged
```

`--workers` controls local serial, thread, process or Ray execution inside a
rank. `--rank/--world-size` are the only partitioning inputs; no scheduler or
tracking service is required. `--resume` reuses a row only when its token and
resolved configuration fingerprint match. Merging rejects missing/duplicate
ranks, duplicate tokens and stale records. Failed rows stay visible and make
the run non-successful.

## Closed loop

`T4ClosedLoopRunner` performs sensor replay with a kinematic perfect tracker.
At each source tick it calls the agent, converts the plan to the source grid,
advances the ego state and feeds the realized state into the next tick. It
does not re-render sensors. Recorded traffic is replayed by default; constant
velocity and IDM policies are available for controlled experiments.

```bash
uv run t4e2e evaluate-closed-loop /path/to/val.json \
  --agent my_agent \
  --output-dir results/closed-loop
```

Closed-loop metrics and rollout artifacts remain in their own report section
and are merged with `t4e2e merge-closed-loop`.

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
        pdm_metric_names=("ego_progress", "score"),
    ),
    families=("pdm",),
)
```

`MetricCache` stores only metric outputs. The cache signature includes the
prediction, scene arrays, consecutive-plan inputs and metric version.
