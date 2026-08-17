# Integrating an existing agent

An existing planning model only needs an adapter. Keep its network, loss and
optimizer in its own package; use this repository for data, sampling,
evaluation and visualization.

## Adapter checklist

Implement `AbstractT4Agent`:

1. return a stable name;
2. declare the camera and LiDAR streams it reads;
3. provide feature and target builders;
4. declare the emitted `TrajectorySampling`;
5. return `{"trajectory": tensor}` with shape `[B, num_poses, 3]`;
6. implement loss and optimizer methods when training is required.

Example sampling declarations:

```python
@property
def trajectory_sampling(self):
    return TrajectorySampling(num_poses=80, interval_length=0.1)
```

The target builder must use the same sampling:

```python
TrajectoryTargetBuilder(trajectory_sampling=self.trajectory_sampling)
```

## Compatibility rules

- Use `T4Scene` for privileged evaluation and `T4AgentInput` for inference.
- Keep boxes in `[x, y, z, width, length, height, yaw, vx, vy]` order.
- Keep all poses in the current ego frame.
- Do not infer trajectory time from point count.
- Do not pad variable-length object lists or point clouds in the reader.
- Do not replace missing map or annotation fields with zeros.

## Evaluation boundary

`T4PDMScorer.score_batch` reads sampling from each `Trajectory` and resamples
to the configured scoring grid. It accepts a long, dense trajectory and a short,
sparse trajectory as long as both cover the scoring horizon. The visualizer
draws the original samples, so spacing differences remain visible.

For raw proposal tensors, pass the source sampling explicitly:

```python
scorer.score_proposals(
    proposals,
    scenes,
    trajectory_sampling=agent.trajectory_sampling,
)
```

## Registration

```toml
[project.entry-points."t4_e2e_devkit.agents"]
my_agent = "my_package.agent:MyAgent"
```

Then use `agent=my_agent` with the training, evaluation or visualization commands.

## Verification

Run these checks before comparing a new adapter:

```bash
uv run ruff check .
uv run pytest -q tests/test_agents.py tests/test_evaluation.py
uv run t4e2e check --vendor
```

With data available, score the same windows using the CPU and GPU backends and
run `compare_backends` to check component agreement.
