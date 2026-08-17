"""Closed-loop T4 rollout with sensor replay and a perfect tracker.

The rollout owns the ego state only.  At every simulation tick it reads the
recorded T4 window for that timestamp, replaces the recorded ego history with
the simulated ego history, asks the agent for a trajectory, and advances the
ego by one step.  Camera and LiDAR payloads remain replayed bytes from the
recording; they are not rendered again from the simulated pose.

``PerfectTracker`` follows the lightweight tracker used by the reference
closed-loop evaluator: the first future reference point determines a
velocity-limited Euler step and the vehicle heading is set to that point's
heading.  This is intentionally separate from the PDM proposal simulator,
which tracks one fixed proposal through an LQR controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol

import numpy as np

from t4_e2e_devkit.common.constants import (
    PAST_FRAMES,
    T4_INTERVAL_LENGTH,
    X,
    Y,
)
from t4_e2e_devkit.common.dataclasses import (
    EgoStatus,
    MapTensors,
    SceneFilter,
    T4AgentInput,
    T4Scene,
    Trajectory,
)
from t4_e2e_devkit.common.enums import T4BoxIndex
from t4_e2e_devkit.dataset.scene import build_ego_status
from t4_e2e_devkit.dataset.window import T4WindowBuilder
from t4_e2e_devkit.planning.simulation.closed_loop_geometry import (
    ReplayGeometry,
    compute_replay_geometry,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


class ReplaySceneProvider(Protocol):
    """Callable that returns the recorded input window at one source frame."""

    def __call__(self, frame_index: int) -> T4Scene:
        ...


@dataclass(frozen=True)
class T4ClosedLoopConfig:
    """Settings for a sensor-replay, ego-only closed-loop rollout."""

    dt_s: float = T4_INTERVAL_LENGTH
    history_frames: int = PAST_FRAMES
    replan_interval: int = 1
    max_speed_mps: float = 20.0
    goal_radius_m: float = 2.0
    ttc_horizon_s: Optional[float] = 1.0

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {self.dt_s}")
        if not math.isclose(self.dt_s, T4_INTERVAL_LENGTH, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "T4 sensor replay currently advances at the source rate of "
                f"{T4_INTERVAL_LENGTH:g}s; got dt_s={self.dt_s:g}"
            )
        if self.history_frames < 1:
            raise ValueError(f"history_frames must be positive, got {self.history_frames}")
        if self.replan_interval < 1:
            raise ValueError(
                f"replan_interval must be positive, got {self.replan_interval}"
            )
        if self.max_speed_mps <= 0.0:
            raise ValueError(
                f"max_speed_mps must be positive, got {self.max_speed_mps}"
            )
        if self.goal_radius_m <= 0.0:
            raise ValueError(
                f"goal_radius_m must be positive, got {self.goal_radius_m}"
            )
        if self.ttc_horizon_s is not None and self.ttc_horizon_s <= 0.0:
            raise ValueError(
                f"ttc_horizon_s must be positive or None, got {self.ttc_horizon_s}"
            )


@dataclass(frozen=True)
class KinematicState:
    """The simulated rear-axle state in the scene's global frame."""

    x: float
    y: float
    heading: float
    speed_mps: float
    acceleration_mps2: float = 0.0
    yaw_rate_radps: float = 0.0
    steering_rad: float = 0.0

    @property
    def pose(self) -> np.ndarray:
        """:return: ``[x, y, heading]`` as a fresh array."""

        return np.array([self.x, self.y, self.heading], dtype=np.float64)


class PerfectTracker:
    """Velocity-limited Euler follower for one receding-horizon step.

    The reference trajectory is in the global frame and starts one control
    interval after the current state.  The tracker uses the first reference
    displacement to compute speed, integrates position using the current
    heading, and snaps the new heading to the reference heading.  The full
    reference is only used for the resume-from-rest guard.
    """

    def __init__(
        self,
        dt_s: float = T4_INTERVAL_LENGTH,
        max_speed_mps: float = 20.0,
        *,
        dt: Optional[float] = None,
        max_speed: Optional[float] = None,
    ):
        if dt is not None:
            dt_s = dt
        if max_speed is not None:
            max_speed_mps = max_speed
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if max_speed_mps <= 0.0:
            raise ValueError(f"max_speed_mps must be positive, got {max_speed_mps}")
        self.dt_s = float(dt_s)
        self.max_speed_mps = float(max_speed_mps)
        self.last_accel = 0.0
        self.last_yaw_rate = 0.0
        self.last_steering = 0.0

    def track(
        self,
        state: KinematicState | np.ndarray,
        reference_world: np.ndarray,
    ) -> KinematicState | tuple[np.ndarray, float]:
        """Track one step using either a T4 state or reference-array input.

        Passing :class:`KinematicState` returns a state object for the T4
        runner. Passing ``[x, y, heading, speed]`` returns
        ``(new_pose, new_speed)``, matching the reference tracker API.
        """

        if isinstance(state, KinematicState):
            updated = self.track_state(state, reference_world)
            self.last_accel = updated.acceleration_mps2
            self.last_yaw_rate = updated.yaw_rate_radps
            self.last_steering = updated.steering_rad
            return updated
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        if values.shape[0] != 4:
            raise ValueError(f"state array must have four values, got {values.shape}")
        updated = self.track_state(
            KinematicState(
                x=float(values[0]),
                y=float(values[1]),
                heading=float(values[2]),
                speed_mps=float(values[3]),
            ),
            reference_world,
        )
        self.last_accel = updated.acceleration_mps2
        self.last_yaw_rate = updated.yaw_rate_radps
        self.last_steering = updated.steering_rad
        return updated.pose.astype(np.float32), updated.speed_mps

    def track_state(
        self,
        state: KinematicState,
        reference_world: np.ndarray,
    ) -> KinematicState:
        """Advance ``state`` by one ``dt_s`` using ``reference_world``."""

        reference = np.asarray(reference_world, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "reference_world must have shape [num_poses, 3], "
                f"got {reference.shape}"
            )
        if reference.shape[0] == 0:
            return KinematicState(
                state.x,
                state.y,
                state.heading,
                0.0,
                acceleration_mps2=-state.speed_mps / self.dt_s,
            )

        dx = float(reference[0, 0] - state.x)
        dy = float(reference[0, 1] - state.y)
        target_speed = min(math.hypot(dx, dy) / self.dt_s, self.max_speed_mps)

        # A stopped vehicle can receive several near-zero first points from a
        # planner that eases into motion.  The full-horizon average keeps the
        # rollout from remaining parked forever while retaining the tracker's
        # one-step semantics.
        full_horizon_s = reference.shape[0] * self.dt_s
        tail_reach = math.hypot(
            float(reference[-1, 0] - state.x),
            float(reference[-1, 1] - state.y),
        )
        average_plan_speed = tail_reach / full_horizon_s if full_horizon_s > 0.0 else 0.0
        if state.speed_mps < 0.1 and average_plan_speed > 0.5:
            target_speed = max(
                target_speed,
                min(self.max_speed_mps, average_plan_speed),
            )

        x_new = state.x + target_speed * math.cos(state.heading) * self.dt_s
        y_new = state.y + target_speed * math.sin(state.heading) * self.dt_s
        heading_new = float(reference[0, 2])
        heading_delta = _wrapped_angle(heading_new - state.heading)

        return KinematicState(
            x=x_new,
            y=y_new,
            heading=heading_new,
            speed_mps=target_speed,
            acceleration_mps2=(target_speed - state.speed_mps) / self.dt_s,
            yaw_rate_radps=heading_delta / self.dt_s,
        )

    def reset(self) -> None:
        """Reset tracker telemetry; the perfect tracker has no warm start."""

        self.last_accel = 0.0
        self.last_yaw_rate = 0.0
        self.last_steering = 0.0


@dataclass
class T4ClosedLoopResult:
    """Recorded inputs and realized ego states from one rollout."""

    source_frames: np.ndarray
    states: List[KinematicState]
    plans: List[Optional[Trajectory]]
    dt_s: float
    goal_pose_world: Optional[np.ndarray] = None
    collision_steps: Optional[tuple[int, ...]] = None
    timeout: Optional[bool] = None
    termination_reason: str = "completed"
    geometry: Optional[List[Optional[ReplayGeometry]]] = None

    def __post_init__(self) -> None:
        self.source_frames = np.asarray(self.source_frames, dtype=np.int64)
        if len(self.states) != len(self.source_frames) + 1:
            raise ValueError(
                "closed-loop results need one initial state plus one state per source frame"
            )
        if len(self.plans) != len(self.source_frames):
            raise ValueError("closed-loop plans must align with source_frames")
        if self.goal_pose_world is not None:
            goal = np.asarray(self.goal_pose_world, dtype=np.float64).reshape(-1)
            if goal.shape != (4,):
                raise ValueError(
                    "goal_pose_world must have four values (x, y, cos, sin), "
                    f"got {goal.shape}"
                )
            self.goal_pose_world = np.ascontiguousarray(goal)
        if self.collision_steps is not None:
            steps = tuple(sorted({int(step) for step in self.collision_steps}))
            if any(step < 0 or step >= len(self.source_frames) for step in steps):
                raise ValueError(
                    "collision_steps must refer to evaluated source ticks, got "
                    f"{steps} for {len(self.source_frames)} steps"
                )
            self.collision_steps = steps
        if self.geometry is not None and len(self.geometry) != len(self.source_frames):
            raise ValueError(
                "closed-loop geometry must align with source_frames"
            )
        if self.timeout is not None:
            self.timeout = bool(self.timeout)
        if self.termination_reason not in {
            "completed",
            "goal_reached",
            "collision",
            "timeout",
        }:
            raise ValueError(
                "termination_reason must be one of completed, goal_reached, "
                f"collision or timeout; got {self.termination_reason!r}"
            )

    @property
    def realized_poses_world(self) -> np.ndarray:
        """:return: realized poses, including the initial state, in global coordinates."""

        return np.asarray(
            [[state.x, state.y, state.heading] for state in self.states],
            dtype=np.float32,
        )

    def realized_trajectory(self) -> Trajectory:
        """Return the realized future in the initial ego frame."""

        initial = self.states[0]
        local = _world_to_local(self.realized_poses_world[1:], initial.pose)
        return Trajectory(
            poses=local.astype(np.float32),
            trajectory_sampling=TrajectorySampling(
                num_poses=len(local), interval_length=self.dt_s
            ),
        )


class T4ClosedLoopRunner:
    """Run an agent against replayed T4 observations with a simulated ego."""

    def __init__(
        self,
        agent,
        scene_provider: ReplaySceneProvider | Callable[[int], T4Scene],
        config: Optional[T4ClosedLoopConfig] = None,
        close_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        if getattr(agent, "requires_scene", False):
            raise ValueError(
                "closed-loop inference requires an agent that plans from T4AgentInput; "
                "privileged scene oracles are not valid rollout agents"
            )
        self.agent = agent
        self.scene_provider = scene_provider
        self.config = config or T4ClosedLoopConfig()
        self._close_callback = close_callback

    @classmethod
    def from_scene_dir(
        cls,
        agent,
        scene_dir: str | Path,
        root: str | Path,
        *,
        config: Optional[T4ClosedLoopConfig] = None,
        reader_config: Optional[dict] = None,
    ) -> "T4ClosedLoopRunner":
        """Create a runner whose observations come from one T4 scene directory."""

        if getattr(agent, "requires_scene", False):
            raise ValueError(
                "closed-loop inference requires an agent that plans from T4AgentInput; "
                "privileged scene oracles are not valid rollout agents"
            )
        loop_config = config or T4ClosedLoopConfig()
        builder = T4WindowBuilder(
            scene_dir=scene_dir,
            root=root,
            sensor_config=agent.get_sensor_config(),
            scene_filter=SceneFilter(
                num_history_frames=loop_config.history_frames,
                num_future_frames=0,
                frame_interval=1,
            ),
            reader_config=reader_config,
        )
        return cls(agent, builder.build, loop_config, close_callback=builder.close)

    def close(self) -> None:
        """Release resources owned by :meth:`from_scene_dir`."""

        if self._close_callback is not None:
            callback, self._close_callback = self._close_callback, None
            callback()

    def __enter__(self) -> "T4ClosedLoopRunner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def run(self, start_frame: int, num_steps: int) -> T4ClosedLoopResult:
        """Run ``num_steps`` ticks, using source frames ``start_frame + k``."""

        if start_frame < self.config.history_frames - 1:
            raise ValueError(
                f"start_frame={start_frame} needs at least "
                f"{self.config.history_frames - 1} history frames"
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        first_scene = self.scene_provider(int(start_frame))
        if len(first_scene.frames) != self.config.history_frames:
            raise ValueError(
                "replay scene history length does not match the closed-loop config: "
                f"{len(first_scene.frames)} != {self.config.history_frames}"
            )
        initial_pose = _metadata_pose(first_scene)
        initial_status = first_scene.current_frame.ego_status
        initial_velocity = np.asarray(initial_status.ego_velocity, dtype=np.float64).reshape(2)
        initial_speed = max(0.0, float(np.linalg.norm(initial_velocity)))
        state = KinematicState(*initial_pose, speed_mps=initial_speed)
        goal_pose_world = _goal_pose_world(first_scene)

        history_world = _scene_history_world_poses(first_scene)
        source_frames: List[int] = []
        states = [state]
        plans: List[Optional[Trajectory]] = []
        collision_steps: List[int] = []
        collision_events_available = False
        geometry_events: List[Optional[ReplayGeometry]] = []
        geometry_events_available = False
        last_replay_scene: Optional[T4Scene] = None
        cached_world_plan: Optional[np.ndarray] = None
        cached_plan_offset = 0

        for step in range(num_steps):
            source_frame = int(start_frame + step)
            replay_scene = first_scene if step == 0 else self.scene_provider(source_frame)
            last_replay_scene = replay_scene
            if len(replay_scene.frames) != self.config.history_frames:
                raise ValueError(
                    f"replay scene at frame {source_frame} has "
                    f"{len(replay_scene.frames)} history frames, expected "
                    f"{self.config.history_frames}"
                )

            collision_tokens = _replay_collision_tokens(state, replay_scene)
            if collision_tokens is not None:
                collision_events_available = True
                if collision_tokens:
                    collision_steps.append(step)

            agent_input = _build_live_agent_input(replay_scene, history_world, state, self.config.dt_s)
            raw_plan: Optional[Trajectory] = None
            if cached_world_plan is None or step % self.config.replan_interval == 0:
                raw_plan = self.agent.compute_trajectory(agent_input)
                if not isinstance(raw_plan, Trajectory):
                    raise TypeError(
                        "closed-loop agents must return a t4_e2e_devkit Trajectory, "
                        f"got {type(raw_plan).__name__}"
                    )
                dense_plan = _densify_for_simulation(raw_plan, self.config.dt_s)
                cached_world_plan = _local_to_world(dense_plan.poses, state.pose)
                cached_plan_offset = 0

            assert cached_world_plan is not None
            reference = cached_world_plan[cached_plan_offset:]
            next_state = self._tracker.track(state, reference)
            assert isinstance(next_state, KinematicState)
            cached_plan_offset += 1

            geometry_event = compute_replay_geometry(
                next_state,
                replay_scene,
                ttc_horizon_s=self.config.ttc_horizon_s,
                ttc_step_s=self.config.dt_s,
            )
            geometry_events.append(geometry_event if geometry_event.available else None)
            geometry_events_available = geometry_events_available or geometry_event.available

            source_frames.append(source_frame)
            plans.append(raw_plan)
            states.append(next_state)
            state = next_state
            history_world = np.concatenate(
                (history_world[1:], state.pose[None, :]), axis=0
            )

        if last_replay_scene is not None:
            final_collision_tokens = _replay_collision_tokens(states[-1], last_replay_scene)
            if final_collision_tokens is not None:
                collision_events_available = True
                if final_collision_tokens and (num_steps - 1) not in collision_steps:
                    collision_steps.append(num_steps - 1)
                    collision_steps.sort()

            final_geometry = compute_replay_geometry(
                states[-1],
                last_replay_scene,
                ttc_horizon_s=self.config.ttc_horizon_s,
                ttc_step_s=self.config.dt_s,
            )
            if final_geometry.available:
                geometry_events[-1] = final_geometry
                geometry_events_available = True

        timeout: Optional[bool] = None
        termination_reason = "completed"
        if goal_pose_world is not None:
            goal_distance = float(
                np.linalg.norm(states[-1].pose[:2] - goal_pose_world[:2])
            )
            timeout = goal_distance > self.config.goal_radius_m
        if collision_steps:
            termination_reason = "collision"
        elif goal_pose_world is not None:
            termination_reason = "goal_reached" if not timeout else "timeout"

        return T4ClosedLoopResult(
            source_frames=np.asarray(source_frames, dtype=np.int64),
            states=states,
            plans=plans,
            dt_s=self.config.dt_s,
            goal_pose_world=goal_pose_world,
            collision_steps=tuple(collision_steps) if collision_events_available else None,
            geometry=geometry_events if geometry_events_available else None,
            timeout=timeout,
            termination_reason=termination_reason,
        )

    @property
    def _tracker(self) -> PerfectTracker:
        """Lazily create the stateless tracker once per runner."""

        tracker = getattr(self, "_tracker_instance", None)
        if tracker is None:
            tracker = PerfectTracker(
                dt_s=self.config.dt_s,
                max_speed_mps=self.config.max_speed_mps,
            )
            self._tracker_instance = tracker
        return tracker


def run_t4_closed_loop(
    agent,
    scene_dir: str | Path,
    root: str | Path,
    start_frame: int,
    num_steps: int,
    *,
    config: Optional[T4ClosedLoopConfig] = None,
    reader_config: Optional[dict] = None,
) -> T4ClosedLoopResult:
    """Convenience function for one sensor-replay closed-loop rollout."""

    with T4ClosedLoopRunner.from_scene_dir(
        agent,
        scene_dir,
        root,
        config=config,
        reader_config=reader_config,
    ) as runner:
        return runner.run(start_frame=start_frame, num_steps=num_steps)


def _densify_for_simulation(trajectory: Trajectory, dt_s: float) -> Trajectory:
    """Convert any trajectory grid to the simulator's source-rate grid."""

    num_poses = int(math.floor(trajectory.trajectory_sampling.time_horizon / dt_s + 1e-8))
    if num_poses < 1:
        raise ValueError(
            f"trajectory horizon {trajectory.duration:g}s is shorter than the "
            f"simulation step {dt_s:g}s"
        )
    target_sampling = TrajectorySampling(num_poses=num_poses, interval_length=dt_s)
    return trajectory.resample(target_sampling)


def _metadata_pose(scene: T4Scene) -> np.ndarray:
    metadata_pose = scene.scene_metadata.global_center_pose
    if metadata_pose is None:
        raise ValueError(
            f"scene {scene.scene_metadata.token} has no global_center_pose; "
            "closed-loop state propagation needs a global seed pose"
        )
    values = np.asarray(metadata_pose, dtype=np.float64).reshape(-1)
    if values.shape[0] != 4:
        raise ValueError(f"global_center_pose must have four values, got {values.shape}")
    return np.array(
        [values[0], values[1], math.atan2(float(values[3]), float(values[2]))],
        dtype=np.float64,
    )


def _goal_pose_world(scene: T4Scene) -> Optional[np.ndarray]:
    """Convert the scene-local destination into global ``(x, y, cos, sin)``."""

    if scene.goal_pose is None:
        return None
    goal = np.asarray(scene.goal_pose, dtype=np.float64).reshape(-1)
    if goal.shape != (4,):
        raise ValueError(
            f"scene {scene.scene_metadata.token} goal_pose must have four values, "
            f"got {goal.shape}"
        )
    recorded_pose = _metadata_pose(scene)
    local_goal = np.array(
        [goal[0], goal[1], math.atan2(float(goal[3]), float(goal[2]))],
        dtype=np.float64,
    )
    world_goal = _local_to_world(local_goal.reshape(1, 3), recorded_pose)[0]
    return np.array(
        [
            world_goal[0],
            world_goal[1],
            math.cos(world_goal[2]),
            math.sin(world_goal[2]),
        ],
        dtype=np.float64,
    )


def _replay_collision_tokens(
    state: KinematicState,
    scene: T4Scene,
) -> Optional[tuple[str, ...]]:
    """Return replayed tracks intersecting the simulated ego footprint.

    Annotations are local to the recorded current frame.  The current frame's
    global pose moves every object into the rollout frame before the geometry
    test.  ``None`` means the replay source did not provide annotations, while
    an empty tuple is an observed, collision-free frame.
    """

    annotations = scene.current_frame.annotations
    if annotations is None:
        return None

    from shapely.geometry import Polygon

    recorded_pose = _metadata_pose(scene)
    ego_shape = scene.current_frame.ego_status.ego_shape
    ego_center = np.array(
        [
            state.x + ego_shape.rear_axle_to_center * math.cos(state.heading),
            state.y + ego_shape.rear_axle_to_center * math.sin(state.heading),
        ],
        dtype=np.float64,
    )
    ego_polygon = Polygon(
        _box_corners(
            ego_center[0],
            ego_center[1],
            state.heading,
            ego_shape.length,
            ego_shape.width,
        )
    )

    collided: list[str] = []
    boxes = np.asarray(annotations.boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < T4BoxIndex.HEADING + 1:
        return ()
    for index, box in enumerate(boxes):
        width = float(box[T4BoxIndex.WIDTH])
        length = float(box[T4BoxIndex.LENGTH])
        if width <= 0.0 or length <= 0.0:
            continue
        local_box = np.array(
            [box[T4BoxIndex.X], box[T4BoxIndex.Y], box[T4BoxIndex.HEADING]],
            dtype=np.float64,
        )
        world_box = _local_to_world(local_box.reshape(1, 3), recorded_pose)[0]
        if ego_polygon.intersects(
            Polygon(
                _box_corners(
                    world_box[0],
                    world_box[1],
                    world_box[2],
                    length,
                    width,
                )
            )
        ):
            if annotations.track_tokens is not None:
                collided.append(str(annotations.track_tokens[index]))
            else:
                collided.append(str(index))
    return tuple(collided)


def _box_corners(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Return oriented rectangle corners in counter-clockwise order."""

    half_length = length / 2.0
    half_width = width / 2.0
    c, s = math.cos(heading), math.sin(heading)
    local = np.array(
        [
            [half_length, half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
            [half_length, -half_width],
        ],
        dtype=np.float64,
    )
    return np.column_stack(
        (
            x + local[:, 0] * c - local[:, 1] * s,
            y + local[:, 0] * s + local[:, 1] * c,
        )
    )


def _scene_history_world_poses(scene: T4Scene) -> np.ndarray:
    """Recover the recorded history in global coordinates for rollout seeding."""

    center = _metadata_pose(scene)
    local = scene.get_history_poses().astype(np.float64)
    return _local_to_world(local, center)


def _local_to_world(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    c, s = math.cos(float(origin[2])), math.sin(float(origin[2]))
    world = np.empty_like(values)
    world[:, 0] = origin[0] + c * values[:, 0] - s * values[:, 1]
    world[:, 1] = origin[1] + s * values[:, 0] + c * values[:, 1]
    world[:, 2] = values[:, 2] + origin[2]
    return world


def _world_to_local(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    c, s = math.cos(float(origin[2])), math.sin(float(origin[2]))
    dx = values[:, 0] - origin[0]
    dy = values[:, 1] - origin[1]
    local = np.empty_like(values)
    local[:, 0] = c * dx + s * dy
    local[:, 1] = -s * dx + c * dy
    local[:, 2] = values[:, 2] - origin[2]
    return local


def _rebase_map(map_tensors: Optional[MapTensors], recorded_pose: np.ndarray, live_pose: np.ndarray):
    """Re-express vector map geometry around the simulated ego.

    Raw images and point clouds remain replayed in their recorded sensor frame.
    Vector geometry can be rebased exactly, so it is transformed when present.
    """

    if map_tensors is None:
        return None

    relative_heading = float(recorded_pose[2] - live_pose[2])
    c, s = math.cos(relative_heading), math.sin(relative_heading)
    recorded_in_live = _world_to_local(
        np.asarray(recorded_pose, dtype=np.float64).reshape(1, 3), live_pose
    )[0, :2]

    def transform_positions(values: np.ndarray, columns: tuple[int, int]) -> np.ndarray:
        result = np.asarray(values).copy()
        x_values = np.asarray(values[..., columns[0]], dtype=np.float64)
        y_values = np.asarray(values[..., columns[1]], dtype=np.float64)
        result[..., columns[0]] = recorded_in_live[0] + c * x_values - s * y_values
        result[..., columns[1]] = recorded_in_live[1] + s * x_values + c * y_values
        return result

    def transform_vectors(values: np.ndarray, columns: tuple[int, int]) -> np.ndarray:
        result = np.asarray(values).copy()
        x_values = np.asarray(values[..., columns[0]], dtype=np.float64)
        y_values = np.asarray(values[..., columns[1]], dtype=np.float64)
        result[..., columns[0]] = c * x_values - s * y_values
        result[..., columns[1]] = s * x_values + c * y_values
        return result

    def transform_lane_field(values: np.ndarray) -> np.ndarray:
        result = transform_positions(values, (X, Y))
        result = transform_vectors(result, (2, 3))
        result = transform_positions(result, (4, 5))
        return transform_positions(result, (6, 7))

    return MapTensors(
        lanes=transform_lane_field(map_tensors.lanes),
        lanes_speed_limit=np.array(map_tensors.lanes_speed_limit, copy=True),
        lanes_has_speed_limit=np.array(map_tensors.lanes_has_speed_limit, copy=True),
        route_lanes=transform_lane_field(map_tensors.route_lanes),
        route_lanes_speed_limit=np.array(map_tensors.route_lanes_speed_limit, copy=True),
        route_lanes_has_speed_limit=np.array(map_tensors.route_lanes_has_speed_limit, copy=True),
        polygons=transform_positions(map_tensors.polygons, (0, 1)),
        line_strings=transform_positions(map_tensors.line_strings, (0, 1)),
    )


def _rebase_goal(
    goal_pose: Optional[np.ndarray], recorded_pose: np.ndarray, live_pose: np.ndarray
) -> Optional[np.ndarray]:
    if goal_pose is None:
        return None
    goal = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    if goal.shape[0] != 4:
        raise ValueError(f"goal_pose must have four values, got {goal.shape}")
    goal_local = np.array(
        [goal[0], goal[1], math.atan2(float(goal[3]), float(goal[2]))], dtype=np.float64
    )
    goal_world = _local_to_world(goal_local.reshape(1, 3), recorded_pose)
    goal_live = _world_to_local(goal_world, live_pose)[0]
    return np.array(
        [goal_live[0], goal_live[1], math.cos(goal_live[2]), math.sin(goal_live[2])],
        dtype=np.float32,
    )


def _build_live_agent_input(
    scene: T4Scene,
    history_world: np.ndarray,
    state: KinematicState,
    dt_s: float,
) -> T4AgentInput:
    """Replace the logged ego state while retaining replayed sensors."""

    logged = scene.get_agent_input()
    logged_statuses = logged.ego_statuses
    if len(history_world) != len(logged_statuses):
        raise ValueError(
            f"live ego history has {len(history_world)} rows but replay input has "
            f"{len(logged_statuses)}"
        )

    center = _metadata_pose(scene)
    live_rows = build_ego_status(
        np.column_stack(
            (
                history_world[:, 0],
                history_world[:, 1],
                np.cos(history_world[:, 2]),
                np.sin(history_world[:, 2]),
            )
        ),
        np.array([state.x, state.y, math.cos(state.heading), math.sin(state.heading)]),
        dt_s,
    )
    statuses: List[EgoStatus] = []
    for index, (row, logged_status) in enumerate(zip(live_rows, logged_statuses, strict=True)):
        control_state = None
        if index == len(live_rows) - 1:
            control_state = {
                "velocity": np.array([state.speed_mps, 0.0], dtype=np.float32),
                "acceleration": np.array([state.acceleration_mps2, 0.0], dtype=np.float32),
                "steering": state.steering_rad,
                "yaw_rate": state.yaw_rate_radps,
            }
            # Keep the current pose/kinematics tied to the tracker rather than
            # to the finite-difference history if the tracker snapped heading.
            row = row.copy()
            row[0:3] = 0.0
            row[3:5] = [state.speed_mps, 0.0]
            row[5:7] = [state.acceleration_mps2, 0.0]
        statuses.append(
            EgoStatus(
                ego_pose=row[0:3],
                ego_velocity=row[3:5],
                ego_acceleration=row[5:7],
                ego_shape=logged_status.ego_shape,
                driving_command=logged_status.driving_command,
                turn_indicator=logged_status.turn_indicator,
                control_state=control_state,
            )
        )

    return T4AgentInput(
        ego_statuses=statuses,
        cameras=logged.cameras,
        lidars=logged.lidars,
        map_tensors=_rebase_map(scene.current_frame.map_tensors, center, state.pose),
        goal_pose=_rebase_goal(logged.goal_pose, center, state.pose),
        scene_metadata=logged.scene_metadata,
    )


def _wrapped_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
