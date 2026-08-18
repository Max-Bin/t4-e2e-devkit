# Model integration guide

A model only needs to implement the `AbstractT4Agent` interface when it uses
the registered-agent path. Keep model code in its own package; this repository
owns T4 data loading, sampling, evaluation and visualization.

## Trajectory contract

Return a `Trajectory` with explicit `TrajectorySampling`. The scorer and
open-loop metrics use that metadata, so point count alone never determines
time. For example, both of these cover the standard four-second horizon:

```python
TrajectorySampling(num_poses=80, interval_length=0.1)
TrajectorySampling(num_poses=8, interval_length=0.5)
```

Longer trajectories are evaluated on the requested horizon; shorter ones are
rejected. Heading interpolation unwraps angles before interpolation.

## Metric entry points

Use `T4NavSimScorer` for the PDM family. Select `version="v1"` or
`version="v2"` in `T4NavSimScorerConfig`; omit `metric_names` for the complete
version-specific result. The default `backend="auto"` uses CUDA when
available; pass `backend="cpu"` or `backend="gpu"` to force a path. The CLI
uses `--pdm-version navsim-v1` or `--pdm-version navsim-v2`.

Use `compute_open_loop_metrics` for trajectory error and
`T4ClosedLoopRunner` plus `compute_closed_loop_metrics` for sensor-replay
kinematic closed loop. These families are reported independently.

## Batch and distributed runs

`t4e2e evaluate` supports deterministic rank partitioning with
`--rank/--world-size`, local workers with `--workers`, and atomic row resume.
`merge-evaluation` validates rank manifests, configuration fingerprints and
token uniqueness before recomputing aggregates. Results are runtime artifacts
and are ignored by Git.

## Registration

```toml
[project.entry-points."t4_e2e_devkit.agents"]
my_agent = "my_package.agent:MyAgent"
```

The agent declares its sensor configuration, feature/target builders and
trajectory sampling. Inference receives `T4AgentInput`; privileged scenes are
reserved for evaluation and closed-loop orchestration. For a model that already
owns inference, export a `t4-e2e.predictions` manifest instead; see
[`integration.md`](integration.md).
