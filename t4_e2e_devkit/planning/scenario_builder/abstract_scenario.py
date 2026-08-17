"""Scenario access for T4 windows.

The interface follows the useful part of a scenario-database API while keeping
T4's direct ``(scene, center_frame)`` addressing. All poses and boxes returned
by this module use the scenario's current ego frame; iteration ``0`` is the
current frame and later iterations are recorded future frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol, runtime_checkable

import numpy as np

from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.common.dataclasses import EgoStatus, T4Frame, T4Scene
from t4_e2e_devkit.common.t4_map import T4MapAPI
from t4_e2e_devkit.planning.simulation.observation.observation_type import DetectionsTracks


@dataclass(frozen=True)
class T4TrafficLightStatus:
    """One lane-associated traffic-light state from the vector map."""

    lane_id: Optional[str]
    state: str
    row_index: int


@runtime_checkable
class AbstractScenario(Protocol):
    """Scenario surface shared by readers, planners and metric runners."""

    @property
    def token(self) -> str:
        ...

    @property
    def database_interval(self) -> float:
        ...

    def get_number_of_iterations(self) -> int:
        ...

    def get_ego_status_at_iteration(self, iteration: int) -> EgoStatus:
        ...

    def get_ego_future_trajectory(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
    ) -> Iterable[EgoStatus]:
        ...

    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        ...

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
    ) -> Iterable[DetectionsTracks]:
        ...

    def get_map_api(self) -> Optional[T4MapAPI]:
        ...


class T4Scenario:
    """NuPlan-like scenario view over one :class:`T4Scene` window."""

    def __init__(
        self,
        scene: T4Scene,
        interval_length: float,
        *,
        map_api: Optional[T4MapAPI] = None,
    ) -> None:
        if interval_length <= 0.0:
            raise ValueError("interval_length must be positive")
        self._scene = scene
        self._interval_length = float(interval_length)
        self._map_api = map_api
        self._future_statuses: Optional[tuple[EgoStatus, ...]] = None

    @property
    def scene(self) -> T4Scene:
        """:return: the underlying privileged scene window."""
        return self._scene

    @property
    def token(self) -> str:
        """:return: stable scene-window token."""
        return self._scene.scene_metadata.token

    @property
    def log_name(self) -> str:
        """:return: the T4 scene directory relative to its dataset root."""
        return self._scene.scene_metadata.scene_dir

    @property
    def scenario_type(self) -> str:
        """:return: semantic tag summary, or ``"t4"`` when untagged."""
        events = sorted({event for tag in self._scene.scene_metadata.scene_tags for event in tag.events})
        return "+".join(events) if events else "t4"

    @property
    def database_interval(self) -> float:
        """:return: source sampling interval in seconds."""
        return self._interval_length

    @property
    def initial_ego_status(self) -> EgoStatus:
        """:return: the current status at iteration zero."""
        return self.get_ego_status_at_iteration(0)

    @property
    def number_of_iterations(self) -> int:
        """:return: current iteration plus every available future pose."""
        return self.get_number_of_iterations()

    def get_number_of_iterations(self) -> int:
        if self._scene.future_ego_poses is not None:
            return int(len(self._scene.future_ego_poses) + 1)
        if self._scene.future_annotations is not None:
            return int(len(self._scene.future_annotations))
        return 1

    def get_time_point(self, iteration: int) -> int:
        """Return the source timestamp at one scenario iteration in microseconds."""

        if iteration < 0 or iteration >= self.get_number_of_iterations():
            raise IndexError(f"invalid scenario iteration {iteration} for {self.token}")
        return self._timestamp_at_iteration(iteration)

    def get_ego_status_at_iteration(self, iteration: int) -> EgoStatus:
        statuses = self._ego_statuses()
        try:
            return statuses[int(iteration)]
        except (IndexError, TypeError):
            raise IndexError(
                f"scenario {self.token} has no ego status at iteration {iteration}; "
                f"valid range is [0, {len(statuses)})"
            ) from None

    def get_past_ego_statuses(
        self,
        iteration: int = 0,
        time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[EgoStatus]:
        """Return statuses ending at ``iteration`` and including that status."""
        if iteration != 0:
            raise ValueError("T4 scenario past samples are anchored at iteration 0")
        available = self._scene.frames
        indices = self._sample_indices(
            time_horizon,
            num_samples,
            len(available) - 1,
            reverse=True,
        )
        return [available[self._scene.current_frame_index - offset].ego_status for offset in indices]

    def get_future_ego_statuses(
        self,
        iteration: int = 0,
        time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[EgoStatus]:
        """Return statuses starting at ``iteration`` and including it."""
        total = self.get_number_of_iterations() - 1 - int(iteration)
        indices = self._sample_indices(time_horizon, num_samples, total)
        return [self.get_ego_status_at_iteration(int(iteration) + offset) for offset in indices]

    def get_ego_past_trajectory(
        self,
        iteration: int = 0,
        time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[EgoStatus]:
        """NuPlan-style alias for :meth:`get_past_ego_statuses`."""

        return self.get_past_ego_statuses(iteration, time_horizon, num_samples)

    def get_ego_future_trajectory(
        self,
        iteration: int = 0,
        time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[EgoStatus]:
        """NuPlan-style alias for :meth:`get_future_ego_statuses`."""

        return self.get_future_ego_statuses(iteration, time_horizon, num_samples)

    def get_past_timestamps(
        self,
        iteration: int = 0,
        time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[int]:
        """Return history timestamps in chronological order, including now."""

        if iteration != 0:
            raise ValueError("T4 scenario past samples are anchored at iteration 0")
        offsets = self._sample_indices(
            time_horizon,
            num_samples,
            len(self._scene.frames) - 1,
            reverse=True,
        )
        return [
            int(self._scene.frames[self._scene.current_frame_index - offset].timestamp_us)
            for offset in offsets
        ]

    def get_future_timestamps(
        self,
        iteration: int = 0,
        time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[int]:
        """Return future timestamps in increasing order, including now."""

        total = self.get_number_of_iterations() - 1 - int(iteration)
        offsets = self._sample_indices(time_horizon, num_samples, total)
        return [self._timestamp_at_iteration(int(iteration) + offset) for offset in offsets]

    def get_annotations_at_iteration(self, iteration: int) -> Any:
        """Return raw T4 annotations at one scenario iteration."""
        value = int(iteration)
        if value < 0 or value >= self.get_number_of_iterations():
            raise IndexError(f"invalid scenario iteration {iteration} for {self.token}")
        if self._scene.future_annotations is not None and value < len(self._scene.future_annotations):
            return self._scene.future_annotations[value]
        if value == 0 and self._scene.current_frame.annotations is not None:
            return self._scene.current_frame.annotations
        raise ValueError(
            f"scenario {self.token} has no annotations at iteration {iteration}; "
            "build it with t4_include_history_annotations when past tracks are needed"
        )

    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        from t4_e2e_devkit.dataset.tracks import annotations_to_detections_tracks

        annotations = self.get_annotations_at_iteration(iteration)
        timestamp = self._timestamp_at_iteration(iteration)
        return annotations_to_detections_tracks(
            annotations,
            timestamp_us=timestamp,
        )

    def get_past_tracked_objects(
        self,
        iteration: int = 0,
        time_horizon: float = 0.0,
        num_samples: Optional[int] = None,
    ) -> List[DetectionsTracks]:
        if iteration != 0:
            raise ValueError("T4 scenario past samples are anchored at iteration 0")
        from t4_e2e_devkit.dataset.tracks import annotations_to_detections_tracks

        frames = self._scene.frames
        offsets = self._sample_indices(
            time_horizon,
            num_samples,
            len(frames) - 1,
            reverse=True,
        )
        result: List[DetectionsTracks] = []
        for offset in offsets:
            frame = frames[self._scene.current_frame_index - offset]
            if frame.annotations is None:
                raise ValueError(
                    f"scenario {self.token} has no past annotations; enable "
                    "t4_include_history_annotations"
                )
            result.append(
                annotations_to_detections_tracks(frame.annotations, timestamp_us=frame.timestamp_us)
            )
        return result

    def get_future_tracked_objects(
        self,
        iteration: int = 0,
        time_horizon: float = 0.0,
        num_samples: Optional[int] = None,
    ) -> List[DetectionsTracks]:
        total = self.get_number_of_iterations() - 1 - int(iteration)
        offsets = self._sample_indices(time_horizon, num_samples, total)
        return [
            self.get_tracked_objects_at_iteration(int(iteration) + offset)
            for offset in offsets
        ]

    def get_sensor_frame_at_iteration(self, iteration: int = 0) -> T4Frame:
        """Return current/past replay data; future sensors are not recorded here."""
        value = int(iteration)
        if value > 0:
            raise ValueError("T4 window sensor replay does not include future sensor frames")
        index = self._scene.current_frame_index + value
        if index < 0:
            raise IndexError(f"no sensor frame at history offset {iteration}")
        return self._scene.frames[index]

    def get_sensors_at_iteration(self, iteration: int = 0) -> T4Frame:
        """NuPlan-style alias for the recorded T4 sensor frame."""

        return self.get_sensor_frame_at_iteration(iteration)

    def get_map_api(self) -> Optional[T4MapAPI]:
        """:return: optional source-map facade used to assemble this scenario."""
        return self._map_api

    def get_route_lane_ids(self) -> tuple[str, ...]:
        route = self._scene.scene_metadata.route_metadata
        return () if route is None else route.route_lane_ids

    def get_traffic_light_status_at_iteration(
        self, iteration: int = 0
    ) -> tuple[T4TrafficLightStatus, ...]:
        """Read lane-associated light states from the current map tensor."""
        if int(iteration) != 0:
            return ()
        map_tensors = self._scene.current_frame.map_tensors
        if map_tensors is None:
            return ()
        ids = () if map_tensors.object_ids is None else map_tensors.object_ids.lane_ids
        values = np.asarray(map_tensors.lanes)
        if values.ndim != 3 or values.shape[-1] < C.TRAFFIC_LIGHT + C.TRAFFIC_LIGHT_ONE_HOT_DIM:
            return ()
        names = ("green", "yellow", "red", "white", "none")
        result: list[T4TrafficLightStatus] = []
        for row_index, row in enumerate(values):
            if not np.any(np.abs(row[:, :2]) > 1.0e-6):
                continue
            one_hot = np.mean(
                row[:, C.TRAFFIC_LIGHT : C.TRAFFIC_LIGHT + C.TRAFFIC_LIGHT_ONE_HOT_DIM],
                axis=0,
            )
            result.append(
                T4TrafficLightStatus(
                    lane_id=ids[row_index] if row_index < len(ids) else None,
                    state=names[int(np.argmax(one_hot))],
                    row_index=row_index,
                )
            )
        return tuple(result)

    def _ego_statuses(self) -> tuple[EgoStatus, ...]:
        if self._future_statuses is not None:
            return self._future_statuses
        current = self._scene.current_frame.ego_status
        poses = np.zeros((self.get_number_of_iterations(), 3), dtype=np.float32)
        poses[0] = np.asarray(current.ego_pose, dtype=np.float32)
        if self._scene.future_ego_poses is not None:
            future = np.asarray(self._scene.future_ego_poses, dtype=np.float32)
            poses[1 : len(future) + 1] = future
        velocity = np.zeros((len(poses), 2), dtype=np.float32)
        acceleration = np.zeros((len(poses), 2), dtype=np.float32)
        if len(poses) > 1:
            velocity[1:] = np.diff(poses[:, :2], axis=0) / self._interval_length
            velocity[0] = velocity[1]
        if len(poses) > 2:
            acceleration[2:] = np.diff(velocity[1:], axis=0) / self._interval_length
            acceleration[:2] = acceleration[2]
        statuses = [
            EgoStatus(
                ego_pose=poses[index],
                ego_velocity=velocity[index],
                ego_acceleration=acceleration[index],
                ego_shape=current.ego_shape,
                driving_command=current.driving_command,
                turn_indicator=current.turn_indicator,
            )
            for index in range(len(poses))
        ]
        self._future_statuses = tuple(statuses)
        return self._future_statuses

    def _timestamp_at_iteration(self, iteration: int) -> int:
        if int(iteration) == 0:
            return int(self._scene.current_frame.timestamp_us)
        return int(self._scene.current_frame.timestamp_us + int(iteration) * self._interval_length * 1.0e6)

    def _sample_indices(
        self,
        time_horizon: Optional[float],
        num_samples: Optional[int],
        available: int,
        reverse: bool = False,
    ) -> np.ndarray:
        """Return inclusive offsets, uniformly covering the requested horizon.

        ``num_samples`` is the number of intervals in the returned sequence;
        the current sample is always included as offset zero.  Thus a request
        for two samples over a four-second horizon returns offsets ``[0, 20,
        40]`` at a 0.1-second source rate instead of silently shortening the
        horizon to the first two frames.
        """

        if available < 0:
            raise ValueError(f"available must be non-negative, got {available}")
        if time_horizon is None:
            horizon_steps = int(available)
        else:
            if time_horizon < 0.0:
                raise ValueError("time_horizon must be non-negative")
            horizon_steps = int(round(float(time_horizon) / self._interval_length))
        if horizon_steps > available:
            raise ValueError(
                f"requested {horizon_steps} scenario steps but only {available} are available"
            )
        if num_samples is None:
            values = np.arange(horizon_steps + 1, dtype=np.int64)
        else:
            if num_samples < 0:
                raise ValueError("num_samples must be non-negative")
            intervals = min(int(num_samples), horizon_steps)
            values = np.rint(
                np.linspace(0, horizon_steps, intervals + 1, dtype=np.float64)
            ).astype(np.int64)
            # A coarse source grid can round two requested points to the same
            # frame.  Returning a unique, ordered sequence is less surprising
            # than asking a scenario consumer to process duplicate timestamps.
            values = np.unique(values)
        if reverse:
            values = values[::-1]
        return np.clip(values, 0, available)


__all__ = ["AbstractScenario", "T4Scenario", "T4TrafficLightStatus"]
