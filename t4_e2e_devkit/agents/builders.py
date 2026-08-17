"""Feature and target builders: the boundary between data and model.

A builder turns the devkit's scene dataclasses into the tensors one model
wants.  Splitting this out of the model is what lets a LiDAR agent and a camera
agent share one dataset: the dataset does not know what a model consumes, it
only runs the builders the agent handed it.

The split between the two kinds is a privilege boundary, not a convenience:

* :class:`AbstractFeatureBuilder` sees :class:`T4AgentInput` -- history, sensors,
  map, goal.  No future.
* :class:`AbstractTargetBuilder` sees the whole :class:`T4Scene`, future
  included, because that is what supervision is made of.

Building a target from ``T4AgentInput`` is impossible by construction, and a
feature builder cannot reach the future even by accident.  That is worth more
than the small duplication of passing two objects around.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from t4_e2e_devkit.common.constants import (
    PDM_OBSERVATION_FRAMES,
    SCORER_FUTURE_FRAMES,
    TRAJECTORY_INTERVAL,
    TRAJECTORY_POSES,
)
from t4_e2e_devkit.common.dataclasses import T4AgentInput, T4Scene
from t4_e2e_devkit.dataset.contract import CONTRACT_MAP_FIELDS
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

#: ImageNet statistics, matching the reference camera readers exactly.  The
#: reader hands over resized ``uint8`` bytes and this is where they become
#: model input, so these constants must not drift from the reader's.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AbstractFeatureBuilder(ABC):
    """Turns an agent input into model features."""

    @abstractmethod
    def get_unique_name(self) -> str:
        """:return: a name identifying this builder, used for caching."""

    def get_feature_unique_name(self) -> str:
        """Compatibility alias used by feature-cache orchestration."""

        return self.get_unique_name()

    def get_features_from_scenario(self, scenario: T4Scene) -> Dict[str, torch.Tensor]:
        """Build features from a scenario exposing ``get_agent_input``."""

        return self.compute_features(scenario.get_agent_input())

    @abstractmethod
    def compute_features(self, agent_input: T4AgentInput) -> Dict[str, torch.Tensor]:
        """
        :param agent_input: the non-privileged view of one window.
        :return: named feature tensors, without a batch dimension.
        """


class AbstractTargetBuilder(ABC):
    """Turns a scene into supervision targets."""

    @abstractmethod
    def get_unique_name(self) -> str:
        """:return: a name identifying this builder, used for caching."""

    def get_target_unique_name(self) -> str:
        """Compatibility alias used by target-cache orchestration."""

        return self.get_unique_name()

    def get_targets(self, scenario: T4Scene) -> Dict[str, torch.Tensor]:
        """Compatibility alias for :meth:`compute_targets`."""

        return self.compute_targets(scenario)

    @abstractmethod
    def compute_targets(self, scene: T4Scene) -> Dict[str, torch.Tensor]:
        """
        :param scene: the privileged view of one window, future included.
        :return: named target tensors, without a batch dimension.
        """


class FeatureBuilderRegistry:
    """Stable registry for composing independent feature builders."""

    def __init__(self, builders: Optional[Sequence[AbstractFeatureBuilder]] = None) -> None:
        self._builders: dict[str, AbstractFeatureBuilder] = {}
        for builder in builders or ():
            self.register(builder)

    @property
    def builders(self) -> tuple[AbstractFeatureBuilder, ...]:
        return tuple(self._builders.values())

    def register(self, builder: AbstractFeatureBuilder) -> None:
        name = str(builder.get_unique_name())
        if not name:
            raise ValueError("feature builder name must not be empty")
        if name in self._builders:
            raise ValueError(f"feature builder is already registered: {name}")
        self._builders[name] = builder

    def compute(self, agent_input: T4AgentInput) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        for builder in self._builders.values():
            values = builder.compute_features(agent_input)
            overlap = set(features).intersection(values)
            if overlap:
                raise ValueError(f"feature builders emitted duplicate fields: {sorted(overlap)}")
            features.update(values)
        return features

    def compute_features(self, agent_input: T4AgentInput) -> Dict[str, torch.Tensor]:
        """Explicit alias for callers that distinguish features from targets."""

        return self.compute(agent_input)


class TargetBuilderRegistry:
    """Stable registry for composing independent target builders."""

    def __init__(self, builders: Optional[Sequence[AbstractTargetBuilder]] = None) -> None:
        self._builders: dict[str, AbstractTargetBuilder] = {}
        for builder in builders or ():
            self.register(builder)

    @property
    def builders(self) -> tuple[AbstractTargetBuilder, ...]:
        return tuple(self._builders.values())

    def register(self, builder: AbstractTargetBuilder) -> None:
        name = str(builder.get_unique_name())
        if not name:
            raise ValueError("target builder name must not be empty")
        if name in self._builders:
            raise ValueError(f"target builder is already registered: {name}")
        self._builders[name] = builder

    def compute(self, scene: T4Scene) -> Dict[str, torch.Tensor]:
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._builders.values():
            values = builder.compute_targets(scene)
            overlap = set(targets).intersection(values)
            if overlap:
                raise ValueError(f"target builders emitted duplicate fields: {sorted(overlap)}")
            targets.update(values)
        return targets

    def compute_targets(self, scene: T4Scene) -> Dict[str, torch.Tensor]:
        """Explicit alias for callers that distinguish targets from features."""

        return self.compute(scene)


# --------------------------------------------------------------------------- #
# Built-in feature builders
# --------------------------------------------------------------------------- #


class EgoStatusFeatureBuilder(AbstractFeatureBuilder):
    """The current ego kinematic row, and optionally its history."""

    def __init__(self, include_history: bool = False) -> None:
        """
        :param include_history: emit all history rows rather than the current one.
        """
        self.include_history = include_history

    def get_unique_name(self) -> str:
        """:return: builder name."""
        return "ego_status_history" if self.include_history else "ego_status"

    def compute_features(self, agent_input: T4AgentInput) -> Dict[str, torch.Tensor]:
        """
        :param agent_input: one window's agent input.
        :return: ``ego_status`` ``[7]`` or ``[T, 7]``, plus ``ego_shape`` ``[3]``.
        """
        if self.include_history:
            rows = np.stack([status.as_array() for status in agent_input.ego_statuses])
        else:
            rows = agent_input.ego_status.as_array()
        status = agent_input.ego_status
        control = status.control_state or {}
        velocity = np.asarray(control.get("velocity", status.ego_velocity), dtype=np.float32).reshape(-1)
        acceleration = np.asarray(
            control.get("acceleration", status.ego_acceleration), dtype=np.float32
        ).reshape(-1)
        control_state = np.array(
            [
                velocity[0],
                velocity[1],
                acceleration[0],
                acceleration[1],
                float(control.get("steering", 0.0)),
                float(control.get("yaw_rate", 0.0)),
            ],
            dtype=np.float32,
        )
        return {
            "ego_status": torch.from_numpy(np.ascontiguousarray(rows)),
            "ego_shape": torch.from_numpy(agent_input.ego_status.ego_shape.as_array().astype(np.float32)),
            "control_state": torch.from_numpy(control_state),
        }


class MapFeatureBuilder(AbstractFeatureBuilder):
    """The vector map, route and destination at the current frame."""

    def get_unique_name(self) -> str:
        """:return: builder name."""
        return "vector_map"

    def compute_features(self, agent_input: T4AgentInput) -> Dict[str, torch.Tensor]:
        """
        :param agent_input: one window's agent input.
        :return: the eight map tensors plus ``goal_pose``.
        :raises ValueError: when the map is absent.  Zero-filling here is what
            lets a camera-only run silently discard the route and destination
            and still train, producing a model that cannot follow a route; the
            reference camera path treats a missing map field as fatal for the
            same reason.
        """
        if agent_input.map_tensors is None:
            raise ValueError(
                "MapFeatureBuilder requires map tensors, but this window carries none. "
                "A missing map is never zero-filled: that would train a route-conditioned "
                "model on no route at all."
            )
        features = {
            name: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
            for name, value in agent_input.map_tensors.as_dict().items()
        }
        goal = agent_input.goal_pose
        features["goal_pose"] = torch.from_numpy(
            np.zeros(4, dtype=np.float32) if goal is None else np.asarray(goal, dtype=np.float32)
        )
        return features

    @property
    def field_names(self) -> tuple[str, ...]:
        """:return: the map field names this builder emits."""
        return CONTRACT_MAP_FIELDS + ("goal_pose",)


class CameraFeatureBuilder(AbstractFeatureBuilder):
    """The camera register at the current frame, normalized for a backbone.

    Reproduces the reference readers' chain exactly: the resized image becomes
    ``float32`` in ``[0, 1]``, then ``(x - mean) / std``, then ``CHW``.  A
    missing view is filled with the ImageNet mean, which is precisely zero after
    normalization -- so the model sees "no information" rather than "black", and
    the window stays trainable.
    """

    def __init__(self, camera_names: tuple[str, ...] | None = None) -> None:
        """
        :param camera_names: register order to emit; the input's own order otherwise.
        """
        self.camera_names = camera_names

    def get_unique_name(self) -> str:
        """:return: builder name."""
        return "cameras"

    def compute_features(self, agent_input: T4AgentInput) -> Dict[str, torch.Tensor]:
        """
        :param agent_input: one window's agent input.
        :return: ``camera_images`` ``[N, 3, H, W]`` plus intrinsics and extrinsics.
        :raises ValueError: when no cameras were decoded, which means the
            agent's sensor config and its builders disagree.
        """
        cameras = agent_input.cameras[-1]
        if cameras is None or len(cameras) == 0:
            raise ValueError(
                "CameraFeatureBuilder found no decoded cameras. The agent's "
                "SensorConfig must request the cameras its feature builders read."
            )
        names = list(self.camera_names) if self.camera_names else cameras.names

        images, intrinsics, extrinsics = [], [], []
        shape = self._reference_shape(cameras, names)
        for name in names:
            camera = cameras[name]
            if camera.image is None:
                pixels = np.broadcast_to(IMAGENET_MEAN, shape).astype(np.float32)
            else:
                pixels = camera.image.astype(np.float32) / 255.0
            pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
            images.append(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
            intrinsics.append(
                np.eye(3, dtype=np.float32) if camera.intrinsics is None else camera.intrinsics
            )
            extrinsics.append(self._extrinsic_matrix(camera))

        return {
            "camera_images": torch.from_numpy(np.stack(images, axis=0)),
            "camera_intrinsics": torch.from_numpy(np.stack(intrinsics, axis=0).astype(np.float32)),
            "camera_extrinsics": torch.from_numpy(np.stack(extrinsics, axis=0).astype(np.float32)),
        }

    @staticmethod
    def _reference_shape(cameras, names) -> tuple[int, int, int]:
        for name in names:
            image = cameras[name].image
            if image is not None:
                return image.shape
        raise ValueError(
            f"every requested camera view is missing ({names}); the window has no image "
            "to take a reference resolution from"
        )

    @staticmethod
    def _extrinsic_matrix(camera) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        if camera.camera2ego_rotation is not None:
            matrix[:3, :3] = camera.camera2ego_rotation
        if camera.camera2ego_translation is not None:
            matrix[:3, 3] = camera.camera2ego_translation
        return matrix


class LidarFeatureBuilder(AbstractFeatureBuilder):
    """The current LiDAR sweep, as raw points.

    Voxelization stays in the model: the grid, its range and its resolution are
    architecture choices, and baking one into the loader would force every
    LiDAR agent to share it.
    """

    def get_unique_name(self) -> str:
        """:return: builder name."""
        return "lidar"

    def compute_features(self, agent_input: T4AgentInput) -> Dict[str, torch.Tensor]:
        """
        :param agent_input: one window's agent input.
        :return: ``points`` ``[N, 5]``; ragged across samples, so the collate
            function keeps it a list rather than padding it.
        :raises ValueError: when no sweep was decoded.
        """
        lidar = agent_input.lidars[-1]
        if lidar is None or lidar.lidar_pc is None:
            raise ValueError(
                "LidarFeatureBuilder found no decoded LiDAR sweep. The agent's "
                "SensorConfig must set `lidar` for the steps its builders read."
            )
        return {"points": torch.from_numpy(np.ascontiguousarray(lidar.lidar_pc, dtype=np.float32))}


# --------------------------------------------------------------------------- #
# Built-in target builders
# --------------------------------------------------------------------------- #


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Build a recorded trajectory on a caller-selected time grid."""

    def __init__(
        self,
        num_poses: int | None = None,
        interval_length: float | None = None,
        trajectory_sampling: TrajectorySampling | None = None,
    ) -> None:
        """
        :param num_poses: number of poses to supervise when an explicit
            ``trajectory_sampling`` is not supplied.
        :param interval_length: seconds between target poses.
        :param trajectory_sampling: explicit target sampling.
        """
        if trajectory_sampling is not None:
            if num_poses is not None or interval_length is not None:
                raise ValueError(
                    "trajectory_sampling cannot be combined with num_poses or interval_length"
                )
            self.trajectory_sampling = trajectory_sampling
        else:
            self.trajectory_sampling = TrajectorySampling(
                num_poses=TRAJECTORY_POSES if num_poses is None else num_poses,
                interval_length=(
                    TRAJECTORY_INTERVAL if interval_length is None else interval_length
                ),
            )
        self.num_poses = self.trajectory_sampling.num_poses
        self.interval_length = self.trajectory_sampling.interval_length

    def get_unique_name(self) -> str:
        """:return: builder name."""
        return "trajectory"

    def compute_targets(self, scene: T4Scene) -> Dict[str, torch.Tensor]:
        """
        :param scene: one window.
        :return: ``trajectory`` ``[num_poses, 3]``.
        """
        trajectory = scene.get_future_trajectory(
            trajectory_sampling=self.trajectory_sampling
        )
        return {
            "trajectory": torch.from_numpy(
                np.ascontiguousarray(trajectory.poses, dtype=np.float32)
            )
        }


class OracleTargetBuilder(AbstractTargetBuilder):
    """Everything a scorer-supervised agent needs beyond the trajectory.

    The future agent boxes stay ragged numpy arrays rather than tensors: they
    are consumed by the scorer, not by a network layer, and padding them to a
    fixed object count would either truncate crowded frames or inflate every
    batch to the worst case.

    ``pdm_progress`` is the ego-progress denominator from PDM-Closed. It is
    optional at the dataset boundary: GPU scorer supervision generates it
    online from these arrays, while explicit CPU scoring still requires an
    offline reference.
    """

    def __init__(self, require_pdm_progress: bool = False) -> None:
        """
        :param require_pdm_progress: fail when the PDM-Closed reference is
            missing. Set this to ``True`` for an explicit cache-audit run.
        """
        self.require_pdm_progress = require_pdm_progress

    def get_unique_name(self) -> str:
        """:return: builder name."""
        return "oracle"

    def compute_targets(self, scene: T4Scene) -> Dict[str, object]:
        """
        :param scene: one window.
        :return: future boxes, labels, and the PDM progress denominator.
        :raises ValueError: when the recorded future or reference progress is
            absent.
        """
        if scene.future_annotations is None:
            raise ValueError(
                f"scene {scene.scene_metadata.token} has no future annotations; oracle "
                "supervision requires the recorded future, and its absence is never "
                "treated as an empty traffic scene"
            )
        current = scene.current_frame.annotations
        if current is None:
            raise ValueError(
                f"scene {scene.scene_metadata.token} has no current annotations; "
                "oracle supervision needs the 51-frame PDM observation"
            )
        if len(scene.future_annotations) < PDM_OBSERVATION_FRAMES + 1:
            raise ValueError(
                f"scene {scene.scene_metadata.token} has only {len(scene.future_annotations)} "
                f"annotation frames; expected at least {PDM_OBSERVATION_FRAMES + 1}"
            )
        targets: Dict[str, object] = {
            "oracle_future_trajectory": torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(scene.future_ego_poses[:SCORER_FUTURE_FRAMES], dtype=np.float32)
                )
            )
            if scene.future_ego_poses is not None
            else None,
            "current_agent_boxes": np.asarray(current.boxes, dtype=np.float64),
            "current_agent_labels": np.asarray(current.labels, dtype=np.int64),
            "future_agent_boxes": [
                np.asarray(a.boxes, dtype=np.float64)
                for a in scene.future_annotations[1 : PDM_OBSERVATION_FRAMES + 1]
            ],
            "future_agent_labels": [
                np.asarray(a.labels, dtype=np.int64)
                for a in scene.future_annotations[1 : PDM_OBSERVATION_FRAMES + 1]
            ],
            "agent_gt_available": True,
        }
        if targets["oracle_future_trajectory"] is None:
            raise ValueError(
                f"scene {scene.scene_metadata.token} has no future ego trajectory; "
                "oracle supervision requires the recorded future"
            )
        if scene.pdm_progress is None:
            if self.require_pdm_progress:
                raise ValueError(
                    f"scene {scene.scene_metadata.token} has no pdm_progress. Build "
                    "a PDM-Closed reference cache or online reference provider first; "
                    "substituting the demonstrated "
                    "endpoint silently changes what ego-progress measures."
                )
        else:
            targets["pdm_progress"] = torch.tensor(scene.pdm_progress, dtype=torch.float32)
        return targets
