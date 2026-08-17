"""Small dependency-free interfaces for T4 closed-loop simulation.

The default runner uses recorded observations, replayed traffic and
``PerfectTracker``.  These protocols make each part replaceable without
changing the sensor-replay contract or requiring a particular planner.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

import numpy as np

from t4_e2e_devkit.common.dataclasses import Annotations, T4AgentInput, T4Scene

if TYPE_CHECKING:  # pragma: no cover - import only for static type checking
    from t4_e2e_devkit.planning.simulation.closed_loop import KinematicState


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


__all__ = [
    "CallableObservationProvider",
    "CallableTrafficPolicy",
    "ConstantVelocityTrafficPolicy",
    "EgoController",
    "ObservationProvider",
    "ReplayObservationProvider",
    "ReplayTrafficPolicy",
    "TrafficPolicy",
]
