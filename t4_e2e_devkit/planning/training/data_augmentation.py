"""Small composable data-augmentation boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Optional

import numpy as np

from t4_e2e_devkit.common.dataclasses import (
    Annotations,
    Cameras,
    EgoStatus,
    Lidar,
    MapTensors,
    T4AgentInput,
    T4Scene,
)


class Augmentor(ABC):
    @abstractmethod
    def __call__(self, agent_input: T4AgentInput, scene: Optional[T4Scene]) -> tuple[T4AgentInput, Optional[T4Scene]]:
        """Return augmented agent input and optional privileged scene."""


class ComposeAugmentor(Augmentor):
    def __init__(self, augmentors: list[Augmentor] | tuple[Augmentor, ...]) -> None:
        self.augmentors = tuple(augmentors)

    def __call__(self, agent_input: T4AgentInput, scene: Optional[T4Scene]) -> tuple[T4AgentInput, Optional[T4Scene]]:
        for augmentor in self.augmentors:
            agent_input, scene = augmentor(agent_input, scene)
        return agent_input, scene


class RandomSE2Augmentor(Augmentor):
    """Apply a random planar transform to vector and track geometry.

    Image pixels are not warped by this class. If a sample contains decoded
    images, the augmentor raises instead of silently making camera and vector
    features inconsistent. It is therefore suitable for map/track/trajectory
    inputs and for camera-free training pipelines.
    """

    def __init__(self, translation_std_m: float = 0.0, rotation_std_rad: float = 0.0, seed: Optional[int] = None) -> None:
        if translation_std_m < 0.0 or rotation_std_rad < 0.0:
            raise ValueError("augmentation standard deviations must be non-negative")
        self.translation_std_m = float(translation_std_m)
        self.rotation_std_rad = float(rotation_std_rad)
        self.rng = np.random.default_rng(seed)
        self.last_transform: Optional[np.ndarray] = None

    def __call__(self, agent_input: T4AgentInput, scene: Optional[T4Scene]) -> tuple[T4AgentInput, Optional[T4Scene]]:
        self.last_transform = np.array(
            [
                self.rng.normal(0.0, self.translation_std_m),
                self.rng.normal(0.0, self.translation_std_m),
                self.rng.normal(0.0, self.rotation_std_rad),
            ],
            dtype=np.float32,
        )
        if _has_images(agent_input) or (scene is not None and _scene_has_images(scene)):
            raise ValueError(
                "RandomSE2Augmentor does not warp camera pixels; use it with camera-free inputs"
            )
        transform = self.last_transform
        augmented_input = _transform_agent_input(agent_input, transform)
        augmented_scene = None if scene is None else _transform_scene(scene, transform)
        return augmented_input, augmented_scene


def _transform_agent_input(agent_input: T4AgentInput, transform: np.ndarray) -> T4AgentInput:
    return T4AgentInput(
        ego_statuses=[_transform_ego_status(status, transform) for status in agent_input.ego_statuses],
        cameras=agent_input.cameras,
        lidars=[_transform_lidar(lidar, transform) for lidar in agent_input.lidars],
        map_tensors=_transform_map(agent_input.map_tensors, transform),
        goal_pose=_transform_goal(agent_input.goal_pose, transform),
        scene_metadata=agent_input.scene_metadata,
    )


def _transform_scene(scene: T4Scene, transform: np.ndarray) -> T4Scene:
    frames = [
        replace(
            frame,
            ego_status=_transform_ego_status(frame.ego_status, transform),
            map_tensors=_transform_map(frame.map_tensors, transform),
            annotations=_transform_annotations(frame.annotations, transform),
            lidar=_transform_lidar(frame.lidar, transform),
            cameras=_transform_cameras(frame.cameras, transform),
        )
        for frame in scene.frames
    ]
    future_poses = None
    if scene.future_ego_poses is not None:
        future_poses = _transform_pose_array(scene.future_ego_poses, transform)
    future_annotations = (
        None
        if scene.future_annotations is None
        else [_transform_annotations(item, transform) for item in scene.future_annotations]
    )
    return replace(
        scene,
        frames=frames,
        future_ego_poses=future_poses,
        future_annotations=future_annotations,
        goal_pose=_transform_goal(scene.goal_pose, transform),
    )


def _transform_ego_status(status: EgoStatus, transform: np.ndarray) -> EgoStatus:
    pose = _transform_pose_array(np.asarray(status.ego_pose, dtype=np.float32).reshape(1, 3), transform)[0]
    velocity = _rotate_vectors(np.asarray(status.ego_velocity, dtype=np.float32).reshape(1, 2), transform)[0]
    acceleration = _rotate_vectors(
        np.asarray(status.ego_acceleration, dtype=np.float32).reshape(1, 2), transform
    )[0]
    return replace(status, ego_pose=pose, ego_velocity=velocity, ego_acceleration=acceleration)


def _transform_annotations(annotations: Optional[Annotations], transform: np.ndarray) -> Optional[Annotations]:
    if annotations is None:
        return None
    boxes = np.array(annotations.boxes, copy=True)
    if len(boxes):
        boxes[:, :2] = _transform_xy(boxes[:, :2], transform)
        boxes[:, 6] += transform[2]
        if boxes.shape[1] >= 9:
            boxes[:, 7:9] = _rotate_vectors(boxes[:, 7:9], transform)
    velocities = None
    if annotations.velocities is not None:
        velocities = _rotate_vectors(annotations.velocities, transform)
    return Annotations(
        boxes=boxes,
        labels=np.array(annotations.labels, copy=True),
        track_tokens=None if annotations.track_tokens is None else list(annotations.track_tokens),
        velocities=velocities,
    )


def _transform_map(map_tensors: Optional[MapTensors], transform: np.ndarray) -> Optional[MapTensors]:
    if map_tensors is None:
        return None
    lanes = _transform_lane_array(map_tensors.lanes, transform)
    route_lanes = _transform_lane_array(map_tensors.route_lanes, transform)
    polygons = _transform_xy(np.asarray(map_tensors.polygons)[..., :2], transform)
    polygon_values = np.array(map_tensors.polygons, copy=True)
    polygon_values[..., :2] = polygons
    lines = np.array(map_tensors.line_strings, copy=True)
    lines[..., :2] = _transform_xy(lines[..., :2], transform)
    return MapTensors(
        lanes=lanes,
        lanes_speed_limit=np.array(map_tensors.lanes_speed_limit, copy=True),
        lanes_has_speed_limit=np.array(map_tensors.lanes_has_speed_limit, copy=True),
        route_lanes=route_lanes,
        route_lanes_speed_limit=np.array(map_tensors.route_lanes_speed_limit, copy=True),
        route_lanes_has_speed_limit=np.array(map_tensors.route_lanes_has_speed_limit, copy=True),
        polygons=polygon_values,
        line_strings=lines,
        object_ids=map_tensors.object_ids,
    )


def _transform_lane_array(values: np.ndarray, transform: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    if result.ndim != 3 or result.shape[-1] < 8:
        return result
    result[..., :2] = _transform_xy(result[..., :2], transform)
    result[..., 2:4] = _rotate_vectors(result[..., 2:4].reshape(-1, 2), transform).reshape(result[..., 2:4].shape)
    result[..., 4:8] = _rotate_vectors(result[..., 4:8].reshape(-1, 2), transform).reshape(result[..., 4:8].shape)
    return result


def _transform_pose_array(values: np.ndarray, transform: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] < 3:
        raise ValueError(f"pose array needs a final dimension of at least 3, got {result.shape}")
    result[..., :2] = _transform_xy(result[..., :2], transform)
    result[..., 2] += transform[2]
    return result


def _transform_goal(goal: Optional[np.ndarray], transform: np.ndarray) -> Optional[np.ndarray]:
    if goal is None:
        return None
    result = np.asarray(goal, dtype=np.float32).copy().reshape(-1)
    if result.shape != (4,):
        raise ValueError(f"goal pose must have shape [4], got {result.shape}")
    result[:2] = _transform_xy(result[:2].reshape(1, 2), transform)[0]
    heading = float(np.arctan2(result[3], result[2]) + transform[2])
    result[2:] = (np.cos(heading), np.sin(heading))
    return result


def _transform_lidar(lidar: Optional[Lidar], transform: np.ndarray) -> Optional[Lidar]:
    if lidar is None or lidar.lidar_pc is None:
        return lidar
    points = np.array(lidar.lidar_pc, copy=True)
    points[:, :2] = _transform_xy(points[:, :2], transform)
    return Lidar(points)


def _transform_cameras(cameras: Optional[Cameras], transform: np.ndarray) -> Optional[Cameras]:
    del transform
    return cameras


def _transform_xy(values: np.ndarray, transform: np.ndarray) -> np.ndarray:
    translation = transform[:2]
    angle = float(transform[2])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return np.asarray(values, dtype=np.float32) @ rotation.T + translation


def _rotate_vectors(values: np.ndarray, transform: np.ndarray) -> np.ndarray:
    angle = float(transform[2])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return np.asarray(values, dtype=np.float32) @ rotation.T


def _has_images(agent_input: T4AgentInput) -> bool:
    return any(camera is not None and any(view.image is not None for view in camera) for camera in agent_input.cameras)


def _scene_has_images(scene: T4Scene) -> bool:
    return any(
        frame.cameras is not None and any(view.image is not None for view in frame.cameras)
        for frame in scene.frames
    )


__all__ = ["Augmentor", "ComposeAugmentor", "RandomSE2Augmentor"]
