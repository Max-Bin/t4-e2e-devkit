# Evaluation

The public entry point is `T4PDMScorer`:

```python
from t4_e2e_devkit.evaluation import T4PDMScorer
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

scorer = T4PDMScorer(backend="cpu")
result = scorer.score(trajectory, scene)
```

`score_batch` accepts one `Trajectory` per scene. Each trajectory carries its
own `TrajectorySampling`; the scorer converts it to the evaluation grid before
calling the metric implementation.

## Sampling

The default evaluation grid is 40 poses at 0.1 s (4 s). This is independent of
the producer's grid:

| producer trajectory | evaluation input | result |
|---|---|---|
| 80 poses at 0.1 s (8 s) | first 4 s | accepted and resampled |
| 8 poses at 0.5 s (4 s) | same 4 s grid | accepted |
| any grid covering at least 4 s | first 4 s | accepted and interpolated |
| a grid shorter than 4 s | incomplete | rejected with `ScoringError` |

Interpolation is linear for position and angle-unwrapped for heading. The
current pose at `t=0` is used as an implicit interpolation knot. Time is never
inferred from the number of points.

For raw proposal tensors, pass their sampling explicitly when it differs from
the scorer configuration:

```python
components = scorer.score_proposals(
    proposals, scenes,
    trajectory_sampling=TrajectorySampling(num_poses=80, interval_length=0.1),
)
```

## Backends

| backend | use |
|---|---|
| `gpu` | batched scoring for large runs or training |
| `cpu` | audit implementation for environments without CUDA |

The backend is selected explicitly. A GPU request never falls back to the CPU
path silently. `compare_backends` scores the same windows on both paths and
reports the largest per-component difference.

## Metric families

The devkit exposes four independent families. They are written to separate
sections in `aggregate.yaml`; there is no cross-family total score.

### Open-loop trajectory error

```python
from t4_e2e_devkit.evaluation import compute_open_loop_metrics

metrics = compute_open_loop_metrics(prediction, scene)
```

The default comparison uses the recorded future's source interval and the
overlap of the two declared horizons. An 80-point trajectory at 0.1 s and an
8-point trajectory at 0.5 s are therefore both accepted. The result contains
ADE, FDE, mean/final heading error and a configurable final-position miss
indicator. Set `OpenLoopMetricConfig.target_sampling` to require a fixed
benchmark horizon.

### PDM-Score

The result contains `(nc, dac, ddc, ttc, ep, comfort)`:

- `nc` and `dac` are multiplicative gates;
- `ddc`, `ttc`, `ep` and `comfort` form the weighted average;
- the default aggregate is `(nc * dac) * (5*ep + 5*ttc + 2*comfort) / 12`.

The `ep` denominator is the progress of the configured reference trajectory.
It must come from a PDM-Closed reference or the online GPU reference path; the
recorded future endpoint is not a substitute.

### T4 metrics

`compute_tier4_metrics` covers red-light compliance, feasibility, lane
departure and related terms. `aggregate_tier4_metrics` aggregates only this
family. `PDMResults.tier4_metrics` remains as a compatibility field for
per-window consumers, but these values are not part of the PDM aggregate.

### Closed loop

`compute_closed_loop_metrics` evaluates the realized states returned by
`T4ClosedLoopRunner`: duration, path length, displacement, speed, acceleration,
yaw rate and stuck detection. The T4 runner automatically derives goal status,
replayed-agent collisions and timeout from the scene and annotations. Custom
rollout harnesses can still pass explicit event data; unavailable events are
omitted instead of reported as false zeros.

## Running a score

```bash
uv run t4e2e score \
  agent=my_agent \
  data_list=/path/to/val.json \
  experiment_name=validation \
  backend=cpu
```

The command writes `per_window.csv` for PDM, `open_loop.csv` for trajectory
errors, and `tier4.csv` when requested. The aggregate report keeps the same
families under `pdm`, `open_loop`, `tier4` and `closed_loop`. Failed windows are
written separately instead of being silently omitted.

The PDM reference cache is optional for GPU evaluation and can be supplied for
an explicit CPU/offline run. Its geometry settings must match the scorer.

## Closed-loop rollout

For an ego-only closed loop with recorded sensor replay, use
`T4ClosedLoopRunner` from [`closed_loop.md`](closed_loop.md). The runner calls
the agent at each simulation tick, converts its trajectory to the source 10 Hz
grid, and advances the ego with `PerfectTracker`. It is distinct from PDM
proposal simulation: PDM rolls out a fixed proposal for scoring, while this
runner feeds the realized ego state into the next agent call.

For a data-list run, use:

```bash
uv run t4e2e evaluate-closed-loop \
  /path/to/val.json \
  --agent my_agent \
  --output-dir reports/closed_loop
```

The command writes `closed_loop.csv`, `aggregate.json`, `aggregate.yaml` and,
when needed, `failures.csv`. Closed-loop metrics remain in their own report
section and are never folded into PDM or open-loop scores.
