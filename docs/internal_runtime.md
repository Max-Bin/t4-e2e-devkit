# Internal runtime

This package targets reproducible T4 experiments on internal infrastructure.
Generated state belongs under ignored `results/` or `reports/` directories;
scene data and sensor payloads are never copied into a report.

## Distributed execution

`evaluate` and `evaluate-closed-loop` execute one rank. `distribute` is the
machine-level launcher:

```bash
uv run t4e2e distribute evaluate lists/val.json \
  --agent my_agent --output-dir results/evaluation \
  --world-size 8 --workers 1 --worker-backend serial
```

The launcher starts `rank-0` through `rank-7`, writes one status JSON and log
per rank, retries failed rank processes, and resumes completed rank directories.
After all ranks finish, it validates and writes `merged/`. A rank failure or a
merge failure leaves the orchestration run failed and inspectable.

When CUDA is available and no indexed device is supplied, the launcher assigns
one visible GPU to each rank. Keep local workers at one for GPU runs; use
`world_size` for GPU parallelism.

`--worker-backend serial|thread|process|ray` uses one common worker interface.
Ray is optional and loaded only when selected. Rank partitioning remains
deterministic and is independent of the worker implementation.

## Traffic simulation

The default closed loop replays recorded traffic. For geometry-only traffic
stress tests, use:

```bash
uv run t4e2e evaluate-closed-loop lists/val.json \
  --agent my_agent --traffic-policy idm \
  --output-dir results/closed-loop-idm
```

The `idm` policy maintains each track in world coordinates, advances it with a
deterministic controller, handles short track gaps, and converts boxes back to
the current T4 frame. Camera and LiDAR bytes remain the recorded payloads; no
sensor renderer is involved. The rollout artifact records the traffic state
history so the result can be audited without rerunning the agent.

## Submissions and local ranking

Create a portable prediction package:

```bash
uv run t4e2e submit lists/val.json \
  --agent my_agent --output-dir results/submission
uv run t4e2e score-submission lists/val.json \
  --submission-dir results/submission \
  --output-dir results/submission-score
```

A package contains `manifest.json` and `predictions.jsonl`. It validates token
coverage, duplicate tokens, finite poses, and per-trajectory sampling metadata.
It contains predictions and run metadata only, not scene data. Rank packages
can be merged with `merge-submission`.

Submission scoring supports the same rank launcher and merge lifecycle:

```bash
uv run t4e2e distribute score-submission lists/val.json \
  --submission-dir results/submission --output-dir results/submission-score \
  --world-size 4 --workers 1 --worker-backend serial
```

Completed result directories can be ranked by one independent metric:

```bash
uv run t4e2e leaderboard results/run-a results/run-b \
  --family open_loop --metric ade_m --no-higher-is-better \
  --output-dir results/leaderboard
```

There is no composite score: family selection and direction are explicit.

## Typed configuration

`run-config` resolves a small typed schema and supports dot-list overrides:

```yaml
mode: evaluate
agent: my_agent
agent_params: {}
dataset:
  data_list: lists/val.json
  history_frames: 31
  future_frames: 80
  frame_interval: 5
evaluation:
  families: [open_loop, pdm]
  backend: auto
simulation:
  stop_on_collision: false
  stop_on_goal: false
workers:
  world_size: 4
  workers: 1
  backend: serial
output:
  directory: results/evaluation
```

```bash
uv run t4e2e run-config configs/evaluate.yaml \
  --override workers.world_size=8
```

Component and metric registries reject duplicate names and unknown selections.
`MetricCatalog.t4_default()` exposes the standard tracking, rollout, comfort,
progress, and safety builders while allowing internal additions through an
explicit registration call.
