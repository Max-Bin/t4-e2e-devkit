# Architecture

The package has four data boundaries:

```text
scene files -> dataset -> T4Scene -> T4AgentInput -> agent -> Trajectory
                                      \-> targets              \-> evaluator
```

Closed-loop evaluation adds a separate runtime path:

```text
replayed T4 window -> live T4AgentInput -> agent -> PerfectTracker
                              ^                           |
                              |------ simulated ego -----|
```

The runtime replays recorded sensors, rebases vector map inputs around the
simulated ego, and advances only the ego state. Other traffic participants are
not reactive agents.

## Layers

| layer | responsibility |
|---|---|
| `common` | arrays, enums, geometry and shared constants |
| `dataset` | scene reading, window assembly and data lists |
| `agents` | sensor declarations, builders and agent registry |
| `evaluation` | independent open-loop, PDM, T4 and closed-loop metric families |
| `planning` | simulation and training integration |
| `visualization` | BEV, camera overlays and reports |
| `script` | CLI and configuration entry points |

The evaluator does not import agents. The agent sees `T4AgentInput`, which has
no future data. Target builders and evaluators receive the privileged
`T4Scene`. This makes the training/deployment boundary structural.

## Core types

`T4Scene` contains the current frame, history, recorded future, map, annotations
and optional reference data. `T4AgentInput` contains only what an agent may use
at inference. `Trajectory` contains local `(x, y, heading)` poses and a
`TrajectorySampling`:

```python
TrajectorySampling(num_poses=80, interval_length=0.1)
TrajectorySampling(num_poses=8, interval_length=0.5)
```

The first pose is at one interval after the current state. Evaluation adapts a
trajectory to its configured grid; visualization preserves its original grid.
This keeps producer format separate from metric format.

## Data ownership

The scene reader owns coordinate conversion, map field validation, annotation
transforms and sensor decoding. Feature builders own conversion to model input.
Agents own only their architecture, loss and optimizer. The scorer owns metric
sampling and aggregation. No layer should duplicate another layer's constants.

## Vendored code

Several geometry and scoring modules are maintained from validated upstream
implementations. Headers identify a source and commit only when that source is
public; private inputs use a generic generated header. Run:

```bash
uv run python tools/vendor.py status
uv run python tools/vendor.py check
```

Do not hand-edit a vendored module. Put interface changes in a devkit-authored
adapter or wrapper; re-run the vendor tool when the upstream source changes.

## Shared constants

`common/constants.py` defines the source rate, window sizes, tensor dimensions,
camera profiles and default PDM weights. The default trajectory is 8 poses at
0.5 s for compatibility, not a requirement on every agent.

The raw future window (80 frames at 10 Hz), the PDM observation window (50
future frames) and the 4-second scoring horizon are separate contracts. A
reader window can be longer than the trajectory consumed by the scorer.
