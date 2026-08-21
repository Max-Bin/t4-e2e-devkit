"""Stateful multi-agent traffic policies for the T4 closed loop.

The policy keeps tracked objects in the global frame, advances them with a
controller, and converts the resulting annotations back to the current T4
observation frame.  This gives the ego planner a coherent evolving scene while
preserving recorded camera and LiDAR payloads.  It is intentionally separate
from sensor rendering: geometry can be simulated, pixels cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from t4_e2e_devkit.common.dataclasses import Annotations, T4Scene
from t4_e2e_devkit.common.enums import T4BoxIndex
from t4_e2e_devkit.planning.simulation.interfaces import (
    TrafficAgentController,
    TrafficAgentState,
)

if TYPE_CHECKING:
    from t4_e2e_devkit.planning.simulation.closed_loop import KinematicState


class IDMTrafficAgentController:
    """Deterministic IDM-style controller for one tracked participant.

    The controller uses the ego as a lead vehicle when it is ahead in the
    participant's lane.  It is a useful internal traffic policy for closed
    loop stress tests; it does not claim to infer routes that are absent from
    the T4 scene.
    """

    def __init__(
        self,
        *,
        desired_speed_mps: float = 13.9,
        time_headway_s: float = 1.5,
        min_gap_m: float = 2.0,
        max_acceleration_mps2: float = 1.5,
        comfortable_deceleration_mps2: float = 2.0,
        exponent: float = 4.0,
        lane_half_width_m: float = 1.6,
    ) -> None:
        if desired_speed_mps <= 0.0 or time_headway_s <= 0.0 or min_gap_m < 0.0:
            raise ValueError("IDM speed, headway and gap parameters are invalid")
        if max_acceleration_mps2 <= 0.0 or comfortable_deceleration_mps2 <= 0.0:
            raise ValueError("IDM acceleration parameters must be positive")
        if exponent <= 0.0 or lane_half_width_m <= 0.0:
            raise ValueError("IDM exponent and lane width must be positive")
        self.desired_speed_mps = float(desired_speed_mps)
        self.time_headway_s = float(time_headway_s)
        self.min_gap_m = float(min_gap_m)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.comfortable_deceleration_mps2 = float(comfortable_deceleration_mps2)
        self.exponent = float(exponent)
        self.lane_half_width_m = float(lane_half_width_m)

    def reset(self) -> None:
        return None

    def step(
        self,
        agent: TrafficAgentState,
        *,
        ego_state: "KinematicState",
        dt_s: float,
    ) -> TrafficAgentState:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        box = np.asarray(agent.box, dtype=np.float64).copy()
        velocity = np.asarray(agent.velocity, dtype=np.float64).copy()
        heading = float(box[T4BoxIndex.HEADING])
        direction = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
        speed = max(0.0, float(np.dot(velocity, direction)))

        ego_delta = np.array([ego_state.x - box[T4BoxIndex.X], ego_state.y - box[T4BoxIndex.Y]])
        longitudinal = float(np.dot(ego_delta, direction))
        lateral = abs(float(direction[0] * ego_delta[1] - direction[1] * ego_delta[0]))
        lead_gap = None
        if longitudinal > 0.0 and lateral <= self.lane_half_width_m:
            lead_gap = max(
                0.0,
                longitudinal
                - float(agent.box[T4BoxIndex.LENGTH]) * 0.5
                - float(ego_state.speed_mps) * 0.0,
            )
        desired_gap = self.min_gap_m + speed * self.time_headway_s
        if lead_gap is not None:
            closing_speed = speed - max(0.0, float(ego_state.speed_mps))
            desired_gap += (
                speed
                * closing_speed
                / (2.0 * math.sqrt(self.max_acceleration_mps2 * self.comfortable_deceleration_mps2))
            )
            interaction = (desired_gap / max(lead_gap, 1.0e-3)) ** 2
        else:
            interaction = 0.0
        free_road = (speed / self.desired_speed_mps) ** self.exponent
        acceleration = self.max_acceleration_mps2 * (1.0 - free_road - interaction)
        acceleration = float(
            np.clip(
                acceleration,
                -self.comfortable_deceleration_mps2 * 2.0,
                self.max_acceleration_mps2,
            )
        )
        new_speed = max(0.0, speed + acceleration * float(dt_s))
        box[:2] += (direction * new_speed * float(dt_s)).astype(box.dtype)
        velocity = direction * new_speed
        box[T4BoxIndex.HEADING] = heading
        if box.size >= T4BoxIndex.VELOCITY_Y + 1:
            box[T4BoxIndex.VELOCITY_X : T4BoxIndex.VELOCITY_Y + 1] = velocity
        return TrafficAgentState(
            track_token=agent.track_token,
            label=agent.label,
            box=box,
            velocity=velocity,
        )


@dataclass(frozen=True)
class TrafficPolicyConfig:
    """Retention settings for a stateful multi-agent policy."""

    max_track_age_steps: int = 3

    def __post_init__(self) -> None:
        if self.max_track_age_steps < 0:
            raise ValueError("max_track_age_steps must be non-negative")


class MultiAgentTrafficPolicy:
    """Advance every visible track with a persistent controller state.

    ``controller`` may be one controller shared by all tracks or a factory
    taking ``(track_token, label)``.  Track states are kept in global
    coordinates, so a changing recorded ego pose does not introduce a frame
    jump.  Tracks absent for a few frames are predicted and then retired.
    """

    def __init__(
        self,
        controller: TrafficAgentController | Callable[[str, int], TrafficAgentController],
        *,
        config: Optional[TrafficPolicyConfig] = None,
    ) -> None:
        self.controller = controller
        self.config = config or TrafficPolicyConfig()
        self._states: dict[str, TrafficAgentState] = {}
        self._ages: dict[str, int] = {}
        self._controllers: dict[str, TrafficAgentController] = {}

    def reset(self) -> None:
        self._states.clear()
        self._ages.clear()
        self._controllers.clear()
        reset = getattr(self.controller, "reset", None)
        if reset is not None and callable(reset):
            reset()

    def snapshot(self) -> tuple[TrafficAgentState, ...]:
        return tuple(self._states[token] for token in sorted(self._states))

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
        if annotations is None:
            return scene
        recorded_pose = _scene_pose(scene)
        observed: dict[str, TrafficAgentState] = {}
        labels = np.asarray(annotations.labels, dtype=np.int64).reshape(-1)
        tokens = (
            [str(value) for value in annotations.track_tokens]
            if annotations.track_tokens is not None
            else [str(index) for index in range(len(annotations))]
        )
        if len(tokens) != len(annotations):
            tokens = [str(index) for index in range(len(annotations))]
        velocities = _annotation_velocities(annotations)
        for index, local_box in enumerate(np.asarray(annotations.boxes, dtype=np.float64)):
            world_box = _local_box_to_world(local_box, recorded_pose)
            world_velocity = _local_vector_to_world(velocities[index], recorded_pose[2])
            observed[tokens[index]] = TrafficAgentState(
                track_token=tokens[index],
                label=int(labels[index]),
                box=world_box,
                velocity=world_velocity,
            )

        next_states: dict[str, TrafficAgentState] = {}
        next_ages: dict[str, int] = {}
        for token, initial in observed.items():
            if token in self._states:
                active = self._controller_for(token, initial.label).step(
                    self._states[token], ego_state=state, dt_s=float(dt_s)
                )
            else:
                active = initial
            next_states[token] = active
            next_ages[token] = 0

        for token, previous in self._states.items():
            if token in observed:
                continue
            age = self._ages.get(token, 0) + 1
            if age > self.config.max_track_age_steps:
                continue
            active = self._controller_for(token, previous.label).step(
                previous, ego_state=state, dt_s=float(dt_s)
            )
            next_states[token] = active
            next_ages[token] = age

        self._states = next_states
        self._ages = next_ages
        local_states = [
            _state_to_local(state_value, recorded_pose) for state_value in self.snapshot()
        ]
        if not local_states:
            updated_annotations = Annotations.empty()
        else:
            updated_annotations = Annotations(
                boxes=np.stack([item.box for item in local_states], axis=0).astype(np.float32),
                labels=np.asarray([item.label for item in local_states], dtype=np.int64),
                track_tokens=[item.track_token for item in local_states],
                velocities=np.stack([item.velocity for item in local_states], axis=0).astype(
                    np.float32
                ),
            )
        frames = list(scene.frames)
        frames[scene.current_frame_index] = replace(
            scene.current_frame,
            annotations=updated_annotations,
        )
        return replace(scene, frames=frames)

    def _controller_for(self, token: str, label: int) -> TrafficAgentController:
        if token not in self._controllers:
            if isinstance(self.controller, TrafficAgentController):
                controller = self.controller
            else:
                controller = self.controller(token, label)
            if not isinstance(controller, TrafficAgentController):
                raise TypeError("traffic controller factory must return TrafficAgentController")
            self._controllers[token] = controller
        return self._controllers[token]


def _scene_pose(scene: T4Scene) -> np.ndarray:
    values = scene.scene_metadata.global_center_pose
    if values is None:
        raise ValueError(f"scene {scene.scene_metadata.token} has no global_center_pose")
    pose = np.asarray(values, dtype=np.float64).reshape(-1)
    if pose.shape != (4,):
        raise ValueError("global_center_pose must contain x, y, cos, sin")
    return np.array([pose[0], pose[1], math.atan2(pose[3], pose[2])], dtype=np.float64)


def _local_box_to_world(box: np.ndarray, pose: np.ndarray) -> np.ndarray:
    result = np.asarray(box, dtype=np.float64).copy()
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    x, y = float(result[T4BoxIndex.X]), float(result[T4BoxIndex.Y])
    result[T4BoxIndex.X] = pose[0] + c * x - s * y
    result[T4BoxIndex.Y] = pose[1] + s * x + c * y
    result[T4BoxIndex.HEADING] += pose[2]
    if result.size >= T4BoxIndex.VELOCITY_Y + 1:
        result[T4BoxIndex.VELOCITY_X : T4BoxIndex.VELOCITY_Y + 1] = _local_vector_to_world(
            result[T4BoxIndex.VELOCITY_X : T4BoxIndex.VELOCITY_Y + 1], pose[2]
        )
    return result


def _state_to_local(agent: TrafficAgentState, pose: np.ndarray) -> TrafficAgentState:
    box = np.asarray(agent.box, dtype=np.float64).copy()
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    dx, dy = box[T4BoxIndex.X] - pose[0], box[T4BoxIndex.Y] - pose[1]
    box[T4BoxIndex.X] = c * dx + s * dy
    box[T4BoxIndex.Y] = -s * dx + c * dy
    box[T4BoxIndex.HEADING] -= pose[2]
    velocity = _world_vector_to_local(agent.velocity, pose[2])
    if box.size >= T4BoxIndex.VELOCITY_Y + 1:
        box[T4BoxIndex.VELOCITY_X : T4BoxIndex.VELOCITY_Y + 1] = velocity
    return TrafficAgentState(agent.track_token, agent.label, box, velocity)


def _local_vector_to_world(vector: np.ndarray, heading: float) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(2)
    c, s = math.cos(float(heading)), math.sin(float(heading))
    return np.array([c * values[0] - s * values[1], s * values[0] + c * values[1]])


def _world_vector_to_local(vector: np.ndarray, heading: float) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(2)
    c, s = math.cos(float(heading)), math.sin(float(heading))
    return np.array([c * values[0] + s * values[1], -s * values[0] + c * values[1]])


def _annotation_velocities(annotations: Annotations) -> np.ndarray:
    if annotations.velocities is not None:
        return np.asarray(annotations.velocities, dtype=np.float64)
    boxes = np.asarray(annotations.boxes, dtype=np.float64)
    if boxes.shape[1] >= T4BoxIndex.VELOCITY_Y + 1:
        return boxes[:, T4BoxIndex.VELOCITY_X : T4BoxIndex.VELOCITY_Y + 1]
    return np.zeros((len(boxes), 2), dtype=np.float64)


__all__ = ["IDMTrafficAgentController", "MultiAgentTrafficPolicy", "TrafficPolicyConfig"]
