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
world goal, timeout status and a termination reason to
`T4ClosedLoopResult`. The rollout remains ego-only: other agents do not react
to the simulated ego.

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

`result.collision_steps` is `None` when the replay source has no annotations;
an empty tuple means annotations were present and no collision was observed.
`result.timeout` is available when the scene provides a goal and the requested
rollout horizon ends outside the goal radius.
