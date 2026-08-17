"""Small dependency-free interfaces for T4 closed-loop simulation.

The default runner uses recorded observations, replayed traffic and
``PerfectTracker``.  These protocols make each part replaceable without
changing the sensor-replay contract or requiring a particular planner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

import numpy as np

from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    T4AgentInput,
    T4Scene,
    Trajectory,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static type checking
    from t4_e2e_devkit.planning.simulation.closed_loop import KinematicState, T4ClosedLoopResult


@runtime_checkable
class EgoController(Protocol):
    """Advance the simulated ego state by one control interval."""

    def step(
        self,
        state: "KinematicState",
        reference_world: np.ndarray,
    ) -> "KinematicState":
        ...

    def reset(self) -> None:
        ...


@runtime_checkable
class ObservationProvider(Protocol):
    """Build the agent-visible input for one replay tick."""

    def get_observation(
        self,
        scene: T4Scene,
        history_world: np.ndarray,
        state: "KinematicState",
        dt_s: float,
    ) -> T4AgentInput:
        ...


@runtime_checkable
class TrafficPolicy(Protocol):
    """Choose the traffic geometry presented at one replay tick."""

    def update(
        self,
        scene: T4Scene,
        *,
        state: "KinematicState",
        step: int,
        dt_s: float,
    ) -> T4Scene:
        ...


@dataclass(frozen=True)
class TrafficAgentState:
    """One replayed traffic participant passed to a reactive controller."""

    track_token: str
    label: int
    box: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        box = np.asarray(self.box, dtype=np.float32).reshape(-1)
        velocity = np.asarray(self.velocity, dtype=np.float32).reshape(-1)
        if box.size < 7:
            raise ValueError(f"traffic agent boxes need at least 7 values, got {box.shape}")
        if velocity.size < 2:
            raise ValueError(f"traffic agent velocity needs two values, got {velocity.shape}")
        if box.size < 9:
            padded = np.zeros(9, dtype=np.float32)
            padded[: box.size] = box
            box = padded
        box[7:9] = velocity[:2]
        object.__setattr__(self, "box", np.ascontiguousarray(box))
        object.__setattr__(self, "velocity", np.ascontiguousarray(velocity[:2]))
        object.__setattr__(self, "track_token", str(self.track_token))
        object.__setattr__(self, "label", int(self.label))


@runtime_checkable
class TrafficAgentController(Protocol):
    """Lifecycle for one reactive traffic-agent policy."""

    def reset(self) -> None:
        ...

    def step(
        self,
        agent: TrafficAgentState,
        *,
        ego_state: "KinematicState",
        dt_s: float,
    ) -> TrafficAgentState:
        ...


class CallableTrafficAgentController:
    """Adapt a callable to :class:`TrafficAgentController`."""

    def __init__(
        self,
        function: Callable[[TrafficAgentState, "KinematicState", float], TrafficAgentState],
    ) -> None:
        self.function = function

    def reset(self) -> None:
        return None

    def step(
        self,
        agent: TrafficAgentState,
        *,
        ego_state: "KinematicState",
        dt_s: float,
    ) -> TrafficAgentState:
        result = self.function(agent, ego_state, dt_s)
        if not isinstance(result, TrafficAgentState):
            raise TypeError(
                "traffic-agent controllers must return TrafficAgentState, "
                f"got {type(result).__name__}"
            )
        return result


class ConstantVelocityTrafficAgentController:
    """Small deterministic controller useful for controlled traffic rollouts."""

    def __init__(self, *, max_speed_mps: Optional[float] = None) -> None:
        if max_speed_mps is not None and max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive when provided")
        self.max_speed_mps = max_speed_mps

    def reset(self) -> None:
        return None

    def step(
        self,
        agent: TrafficAgentState,
        *,
        ego_state: "KinematicState",
        dt_s: float,
    ) -> TrafficAgentState:
        del ego_state
        velocity = np.asarray(agent.velocity, dtype=np.float32).copy()
        if self.max_speed_mps is not None:
            speed = float(np.linalg.norm(velocity))
            if speed > self.max_speed_mps:
                velocity *= float(self.max_speed_mps / speed)
        box = np.asarray(agent.box, dtype=np.float32).copy()
        box[:2] += velocity * float(dt_s)
        return TrafficAgentState(
            track_token=agent.track_token,
            label=agent.label,
            box=box,
            velocity=velocity,
        )


@runtime_checkable
class SimulationCallback(Protocol):
    """Optional hooks for a simulation manager or closed-loop runner."""

    def on_start(self, token: str, state: "KinematicState") -> None:
        ...

    def on_step(self, tick: "SimulationTick") -> None:
        ...

    def on_end(self, result: "T4ClosedLoopResult") -> None:
        ...

    def on_error(self, error: BaseException) -> None:
        ...


@dataclass(frozen=True)
class SimulationTick:
    """Ephemeral per-step event delivered to simulation callbacks."""

    step: int
    source_frame: int
    scene: T4Scene
    observation: T4AgentInput
    state: "KinematicState"
    next_state: "KinematicState"
    plan: Optional[Trajectory]


class ReactiveTrafficPolicy:
    """Apply a controller to every annotated participant at each tick.

    This policy updates tracked-object states only. It does not render camera
    or LiDAR observations; those remain the recorded payload selected by the
    observation provider.
    """

    def __init__(self, controller: TrafficAgentController | Callable[..., TrafficAgentState]) -> None:
        self.controller: TrafficAgentController = (
            controller
            if isinstance(controller, TrafficAgentController)
            else CallableTrafficAgentController(controller)
        )
        self._active_tokens: set[str] = set()

    def reset(self) -> None:
        self._active_tokens.clear()
        reset = getattr(self.controller, "reset", None)
        if reset is not None:
            reset()

    def update(
        self,
        scene: T4Scene,
        *,
        state: "KinematicState",
        step: int,
        dt_s: float,
    ) -> T4Scene:
        del step
        annotations = scene.current_frame.annotations
        if annotations is None or len(annotations) == 0:
            return scene
        boxes = np.asarray(annotations.boxes, dtype=np.float32).copy()
        labels = np.asarray(annotations.labels, dtype=np.int64).reshape(-1)
        tokens = (
            [str(token) for token in annotations.track_tokens]
            if annotations.track_tokens is not None
            else [str(index) for index in range(len(boxes))]
        )
        if len(tokens) != len(boxes):
            tokens = [str(index) for index in range(len(boxes))]
        velocities = _annotation_velocities(annotations, boxes)
        updated_tokens: set[str] = set()
        for index, box in enumerate(boxes):
            agent = TrafficAgentState(
                track_token=tokens[index],
                label=int(labels[index]),
                box=box,
                velocity=velocities[index],
            )
            updated = self.controller.step(agent, ego_state=state, dt_s=float(dt_s))
            if not isinstance(updated, TrafficAgentState):
                raise TypeError(
                    "traffic-agent controllers must return TrafficAgentState, "
                    f"got {type(updated).__name__}"
                )
            width = min(int(updated.box.size), int(boxes.shape[1]))
            boxes[index, :width] = updated.box[:width]
            velocities[index] = updated.velocity
            tokens[index] = updated.track_token
            labels[index] = updated.label
            updated_tokens.add(updated.track_token)
        self._active_tokens = updated_tokens
        result = Annotations(
            boxes=boxes,
            labels=np.array(labels, copy=True),
            track_tokens=tokens,
            velocities=np.array(velocities, dtype=np.float32),
        )
        frames = list(scene.frames)
        frames[scene.current_frame_index] = replace(scene.current_frame, annotations=result)
        return replace(scene, frames=frames)


class CallableObservationProvider:
    """Adapt a function to :class:`ObservationProvider`."""

    def __init__(
        self,
        function: Callable[[T4Scene, np.ndarray, "KinematicState", float], T4AgentInput],
    ) -> None:
        self.function = function

    def get_observation(
        self,
        scene: T4Scene,
        history_world: np.ndarray,
        state: "KinematicState",
        dt_s: float,
    ) -> T4AgentInput:
        return self.function(scene, history_world, state, dt_s)


class ReplayObservationProvider:
    """Default observation provider using replayed bytes and rebased vectors."""

    def get_observation(
        self,
        scene: T4Scene,
        history_world: np.ndarray,
        state: "KinematicState",
        dt_s: float,
    ) -> T4AgentInput:
        from t4_e2e_devkit.planning.simulation.closed_loop import _build_live_agent_input

        return _build_live_agent_input(scene, history_world, state, dt_s)


class ReplayTrafficPolicy:
    """Keep recorded traffic unchanged while the ego is simulated."""

    def reset(self) -> None:
        return None

    def update(
        self,
        scene: T4Scene,
        *,
        state: "KinematicState",
        step: int,
        dt_s: float,
    ) -> T4Scene:
        del state, step, dt_s
        return scene


class CallableTrafficPolicy:
    """Adapt a function to :class:`TrafficPolicy`."""

    def __init__(
        self,
        function: Callable[[T4Scene, "KinematicState", int, float], T4Scene],
    ) -> None:
        self.function = function

    def reset(self) -> None:
        return None

    def update(
        self,
        scene: T4Scene,
        *,
        state: "KinematicState",
        step: int,
        dt_s: float,
    ) -> T4Scene:
        return self.function(scene, state, step, dt_s)


class ConstantVelocityTrafficPolicy:
    """Optional simple traffic policy for controlled synthetic rollouts.

    It advances the current annotation boxes with their recorded ``vx, vy``
    values.  This is deliberately modest: it is a policy hook for tests and
    ablations, not a replacement for a traffic simulator.
    """

    def __init__(self, *, max_speed_mps: Optional[float] = None) -> None:
        if max_speed_mps is not None and max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive when provided")
        self.max_speed_mps = max_speed_mps

    def reset(self) -> None:
        return None

    def update(
        self,
        scene: T4Scene,
        *,
        state: "KinematicState",
        step: int,
        dt_s: float,
    ) -> T4Scene:
        del state, step
        annotations = scene.current_frame.annotations
        if annotations is None or len(annotations) == 0:
            return scene
        boxes = np.array(annotations.boxes, copy=True)
        if annotations.velocities is not None:
            velocities = np.asarray(annotations.velocities, dtype=np.float64)
        elif boxes.shape[1] >= 9:
            velocities = np.asarray(boxes[:, 7:9], dtype=np.float64)
        else:
            return scene
        if self.max_speed_mps is not None:
            speeds = np.linalg.norm(velocities, axis=1)
            scale = np.minimum(1.0, self.max_speed_mps / np.maximum(speeds, 1.0e-9))
            velocities = velocities * scale[:, None]
        boxes[:, 0:2] += (velocities * float(dt_s)).astype(boxes.dtype)
        updated = Annotations(
            boxes=boxes,
            labels=np.array(annotations.labels, copy=True),
            track_tokens=None if annotations.track_tokens is None else list(annotations.track_tokens),
            velocities=np.array(velocities, dtype=np.float32),
        )
        frames = list(scene.frames)
        frames[scene.current_frame_index] = replace(
            scene.current_frame,
            annotations=updated,
        )
        return replace(scene, frames=frames)


def _annotation_velocities(annotations: Annotations, boxes: np.ndarray) -> np.ndarray:
    if annotations.velocities is not None:
        values = np.asarray(annotations.velocities, dtype=np.float32)
        if values.ndim == 2 and values.shape[0] == len(boxes) and values.shape[1] >= 2:
            return np.ascontiguousarray(values[:, :2])
    if boxes.shape[1] >= 9:
        return np.ascontiguousarray(boxes[:, 7:9], dtype=np.float32)
    return np.zeros((len(boxes), 2), dtype=np.float32)


__all__ = [
    "CallableTrafficAgentController",
    "CallableObservationProvider",
    "CallableTrafficPolicy",
    "ConstantVelocityTrafficAgentController",
    "ConstantVelocityTrafficPolicy",
    "EgoController",
    "ObservationProvider",
    "ReactiveTrafficPolicy",
    "ReplayObservationProvider",
    "ReplayTrafficPolicy",
    "SimulationCallback",
    "SimulationTick",
    "TrafficAgentController",
    "TrafficAgentState",
    "TrafficPolicy",
]
