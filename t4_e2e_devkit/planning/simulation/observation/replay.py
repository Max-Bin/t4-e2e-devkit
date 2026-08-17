"""T4 sensor/track replay observations for the generic simulation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from t4_e2e_devkit.common.constants import T4_SUPPORTED_CAMERA_NAMES
from t4_e2e_devkit.common.dataclasses import T4Scene
from t4_e2e_devkit.dataset.tracks import annotations_to_detections_tracks
from t4_e2e_devkit.planning.simulation.observation.observation_type import (
    CameraChannel,
    DetectionsTracks,
    LidarChannel,
    Observation,
    Sensors,
)
from t4_e2e_devkit.planning.simulation.runtime import SimulationHistoryBuffer
from t4_e2e_devkit.planning.simulation.simulation_iteration import SimulationIteration


@dataclass
class T4ReplayObservation(Observation):
    """Recorded sensors, tracks and scene metadata at one replay iteration."""

    scene: T4Scene
    sensors: Optional[Sensors]
    tracks: Optional[DetectionsTracks]


class T4ReplayObservationSource:
    """Adapt a T4 scene provider to the generic observation protocol."""

    def __init__(
        self,
        scene_provider: Callable[[int], T4Scene],
        *,
        start_frame: int = 0,
        include_sensors: bool = True,
        include_tracks: bool = True,
        include_lidar: bool = False,
        camera_names: Optional[Sequence[str]] = None,
    ) -> None:
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        self.scene_provider = scene_provider
        self.start_frame = int(start_frame)
        self.include_sensors = bool(include_sensors)
        self.include_tracks = bool(include_tracks)
        self.include_lidar = bool(include_lidar)
        self.camera_names = tuple(camera_names or T4_SUPPORTED_CAMERA_NAMES)

    def reset(self) -> None:
        return None

    def get_observation(
        self,
        iteration: SimulationIteration,
        history: SimulationHistoryBuffer,
    ) -> T4ReplayObservation:
        del history
        scene = self.scene_provider(self.start_frame + int(iteration.index))
        frame = scene.current_frame
        sensors = (
            _sensors_from_frame(
                frame,
                camera_names=self.camera_names,
                include_lidar=self.include_lidar,
            )
            if self.include_sensors
            else None
        )
        tracks = None
        if self.include_tracks and frame.annotations is not None:
            tracks = annotations_to_detections_tracks(
                frame.annotations,
                timestamp_us=frame.timestamp_us,
            )
        return T4ReplayObservation(scene=scene, sensors=sensors, tracks=tracks)


def _sensors_from_frame(
    frame,
    *,
    camera_names: Sequence[str] = T4_SUPPORTED_CAMERA_NAMES,
    include_lidar: bool = False,
) -> Sensors:
    images = {}
    if frame.cameras is not None:
        allowed = set(str(name) for name in camera_names)
        for name in frame.cameras.names:
            if name not in allowed:
                continue
            try:
                channel = CameraChannel(name)
            except ValueError:
                continue
            image = frame.cameras[name].image
            if image is not None:
                images[channel] = image
    pointcloud = {}
    if include_lidar and frame.lidar is not None and frame.lidar.lidar_pc is not None:
        pointcloud[LidarChannel.LIDAR_CONCAT] = frame.lidar.lidar_pc
    return Sensors(
        pointcloud=pointcloud or None,
        images=images or None,
    )


__all__ = ["T4ReplayObservation", "T4ReplayObservationSource"]
