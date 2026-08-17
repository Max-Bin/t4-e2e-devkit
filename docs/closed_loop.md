# Closed-loop rollout

The devkit supports an ego-only closed loop over T4 recordings:

```text
replayed T4 sensors + rebased vector map
                  ↓
            T4AgentInput
                  ↓
            agent trajectory
                  ↓
             PerfectTracker
                  ↓
       simulated ego state at t + 0.1 s
                  ↺
```

Other traffic participants are not simulated reactively. Their detections and
the sensor payload are replayed from the recorded source frame. Camera images
are therefore still the recorded pixels; the devkit does not render a new
image from the simulated pose. Vector map geometry and the goal are rebased to
the simulated ego frame when they are present.

When replay annotations are available, the runner transforms their boxes into
the rollout frame and records ego-agent collision ticks. It also attaches the
world goal, timeout status, geometry events and a termination reason to
`T4ClosedLoopResult`. The rollout remains ego-only: other agents do not react
to the simulated ego.

The optional per-tick geometry events include minimum signed agent clearance,
constant-velocity replay TTC, drivable-area and road-border violations. TTC is
a diagnostic projection of the recorded box velocity; it is not a traffic-agent
simulator. A field is omitted when its annotations or map layer is unavailable.
The T4 map is a local tensor contract, so these checks are limited to the
provided lane, polygon and road-border window rather than a global map query.

This is also the relevant NuPlan boundary: its standard sensor observation
paths replay recorded sensor samples or tracked objects. They do not generate
new camera views or LiDAR sweeps from the simulated pose. Reactive traffic in
NuPlan updates tracked-object states with a rule-based or user-provided agent
policy; it is not a photorealistic sensor renderer.

The tracker follows the lightweight perfect-tracker contract used by the
reference closed-loop evaluator:

1. Convert the agent's local trajectory to the global frame.
2. Use the first point's displacement to get a target speed.
3. Clamp the speed and integrate one Euler step using the current heading.
4. Set the new heading to the first reference point's heading.
5. Replan on the next tick.

The trajectory's declared sampling is respected. For example, an 8-point
trajectory at 0.5 s is interpolated to 40 points at 0.1 s before tracking; an
80-point trajectory at 0.1 s is used directly.

## Python API

```python
from t4_e2e_devkit import T4ClosedLoopConfig, T4ClosedLoopRunner

config = T4ClosedLoopConfig(history_frames=31, replan_interval=1)

with T4ClosedLoopRunner.from_scene_dir(
    agent,
    scene_dir="/data/t4_dataset/prd_jt/scene/date/time",
    root="/data/t4_dataset",
    config=config,
) as runner:
    result = runner.run(start_frame=100, num_steps=200)

realized = result.realized_trajectory()

from t4_e2e_devkit.evaluation import compute_closed_loop_metrics

metrics = compute_closed_loop_metrics(result)
print(metrics.values)
```

The agent must plan from `T4AgentInput`; privileged future-reading oracles are
rejected. Sensor storage follows the normal T4 reader contract, so the current
public camera input is limited to JPEG-backed wide cameras. LiDAR is requested
only when the agent's `SensorConfig` asks for it.

The rollout components are injectable:

```python
from t4_e2e_devkit.planning.simulation.interfaces import (
    ConstantVelocityTrafficPolicy,
    ReplayObservationProvider,
)

with T4ClosedLoopRunner.from_scene_dir(
    agent,
    scene_dir,
    root,
    observation_provider=ReplayObservationProvider(),
    traffic_policy=ConstantVelocityTrafficPolicy(),
) as runner:
    result = runner.run(start_frame=100, num_steps=40)
```

Use the default replay policy for recorded traffic. The constant-velocity
policy is a small controlled-test hook; it does not render sensors or model
reactive traffic.

For a local simulation-manager boundary with lifecycle callbacks:

```python
from t4_e2e_devkit import SimulationRequest, T4SimulationManager

with T4SimulationManager.from_scene_dir(agent, scene_dir, root) as manager:
    result = manager.run(SimulationRequest(start_frame=100, num_steps=40))
```

Callbacks may implement `on_start`, `on_step`, `on_end` and `on_error`.
`on_step` receives the replay scene, agent observation, plan and realized
states for that tick. `ReactiveTrafficPolicy` is available when a controlled
experiment needs per-track state updates; it must be explicitly injected. The
default manager remains deterministic and local.

`result.collision_steps` is `None` when the replay source has no annotations;
an empty tuple means annotations were present and no collision was observed.
`result.timeout` is available when the scene provides a goal and the requested
rollout horizon ends outside the goal radius.

## Batch artifacts

`evaluate-closed-loop` writes one JSON artifact under `rollouts/` for every
data-list row. It contains the realized ego states, plans at replan ticks,
termination events and the per-tick trace; raw sensor bytes stay in the T4
scene and are addressed by `source_frame`. The directory also contains:

- `run.json`: resolved run settings and configuration fingerprint;
- `closed_loop.csv`: one aggregate row per rollout;
- `closed_loop_ticks.csv`: one row per simulated action;
- `failures.csv`: rows that exhausted their retry budget;
- `report.html`: a self-contained local summary with no external service.

Large runs can be split deterministically and resumed:

```bash
uv run t4e2e evaluate-closed-loop lists/val.json \
  --agent my_agent \
  --output-dir reports/closed_loop \
  --num-shards 8 --shard-index 0 \
  --max-retries 1 --resume
```

`--resume` only reuses successful artifacts whose token and resolved run
configuration still match. A changed rollout setting or data-list selection
causes that row to run again.

Shard reports can be merged after all workers finish. The default requires the
complete declared shard set and rejects duplicate rollout tokens:

```bash
uv run t4e2e merge-closed-loop \
  --input-dir reports/shard-0 reports/shard-1 reports/shard-2 reports/shard-3 \
  --output-dir reports/merged
```

Use `--allow-incomplete` only when the resulting report is intentionally a
subset. `report-closed-loop reports/merged` regenerates the HTML from the CSV
and JSON files.

Visualization outputs are runtime artifacts. Keep generated videos under
`results/visualization/closed_loop/`; this directory is ignored by Git and is
not part of the source documentation.
