"""Portable T4 prediction manifests.

The manifest is the boundary between model repositories and the devkit.  It
contains only data-list keys and local-frame trajectories; scene data,
checkpoint objects and experiment-tracking clients stay outside the format.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

PREDICTION_MANIFEST_FORMAT = "t4-e2e.predictions"
PREDICTION_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class PredictionRecord:
    """One local-frame prediction keyed by a T4 data-list row."""

    scene: str
    center: int
    poses: np.ndarray

    @property
    def key(self) -> tuple[str, int]:
        return self.scene, self.center


@dataclass(frozen=True)
class PredictionManifest:
    """A validated prediction manifest and its header metadata."""

    header: Mapping[str, Any]
    records: Mapping[tuple[str, int], PredictionRecord]


def _validate_key(scene: Any, center: Any) -> tuple[str, int]:
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("prediction scene must be a non-empty string")
    normalized = scene.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"prediction scene must be a safe relative path, got {scene!r}")
    if isinstance(center, bool) or not isinstance(center, (int, np.integer)):
        raise ValueError(f"prediction center must be an integer, got {center!r}")
    center = int(center)
    if center < 0:
        raise ValueError(f"prediction center must be non-negative, got {center}")
    return path.as_posix(), center


def _validate_sampling(num_poses: Any, interval_seconds: Any) -> tuple[int, float]:
    if isinstance(num_poses, bool) or not isinstance(num_poses, (int, np.integer)):
        raise ValueError("manifest trajectory.num_poses must be an integer")
    num_poses = int(num_poses)
    if num_poses < 1:
        raise ValueError("manifest trajectory.num_poses must be positive")
    try:
        interval_seconds = float(interval_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest trajectory.interval_seconds must be numeric") from error
    if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
        raise ValueError("manifest trajectory.interval_seconds must be finite and positive")
    return num_poses, interval_seconds


def _header_sampling(header: Mapping[str, Any]) -> tuple[int, float]:
    trajectory = header.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError("manifest header 'trajectory' must be an object")
    if trajectory.get("pose_format") != "x_y_heading":
        raise ValueError("manifest trajectory.pose_format must be 'x_y_heading'")
    return _validate_sampling(
        trajectory.get("num_poses"),
        trajectory.get("interval_seconds"),
    )


def trajectory_to_poses(
    trajectory: Any, *, num_poses: int
) -> np.ndarray:
    """Convert ``[T, 3]`` or ``[T, 4]`` output to ``x/y/heading`` poses."""

    value = getattr(trajectory, "poses", trajectory)
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    values = np.asarray(value)
    if values.shape not in {(num_poses, 3), (num_poses, 4)}:
        raise ValueError(
            f"trajectory must have shape ({num_poses}, 3) or ({num_poses}, 4), "
            f"got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"trajectory must contain numeric values, got {values.dtype}")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("trajectory contains NaN or Inf")
    if values.shape[1] == 3:
        poses = values
    else:
        heading = np.arctan2(values[:, 3], values[:, 2])
        poses = np.column_stack((values[:, :2], heading))
    return np.ascontiguousarray(poses, dtype=np.float32)


def _validate_poses(poses: Any, *, num_poses: int) -> np.ndarray:
    values = np.asarray(poses)
    if values.shape != (num_poses, 3):
        raise ValueError(
            f"manifest poses must have shape ({num_poses}, 3) in x/y/heading form, "
            f"got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"manifest poses must contain numeric values, got {values.dtype}")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("manifest poses contain NaN or Inf")
    return np.ascontiguousarray(values, dtype=np.float32)


def load_prediction_manifest(path: str | Path) -> PredictionManifest:
    """Read and validate a JSONL prediction manifest."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prediction manifest not found: {path}")

    header: dict[str, Any] | None = None
    records: dict[tuple[str, int], PredictionRecord] = {}
    num_poses = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            if header is None:
                if payload.get("format") != PREDICTION_MANIFEST_FORMAT:
                    raise ValueError(
                        f"{path}:{line_number}: expected format "
                        f"{PREDICTION_MANIFEST_FORMAT!r}"
                    )
                if payload.get("version") != PREDICTION_MANIFEST_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: unsupported prediction manifest version "
                        f"{payload.get('version')!r}"
                    )
                num_poses, _ = _header_sampling(payload)
                header = payload
                continue

            if not {"scene", "center", "poses"}.issubset(payload):
                raise ValueError(
                    f"{path}:{line_number}: prediction rows require scene, center and poses"
                )
            scene, center = _validate_key(payload["scene"], payload["center"])
            key = (scene, center)
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate prediction key {key!r}")
            records[key] = PredictionRecord(
                scene=scene,
                center=center,
                poses=_validate_poses(payload["poses"], num_poses=num_poses),
            )

    if header is None:
        raise ValueError(f"prediction manifest is empty: {path}")
    if not records:
        raise ValueError(f"prediction manifest contains no prediction rows: {path}")
    return PredictionManifest(header=header, records=records)


def validate_prediction_keys(
    records: Mapping[tuple[str, int], PredictionRecord],
    expected_rows: Iterable[tuple[str, int]],
) -> None:
    """Require an exact one-to-one match with selected data-list rows."""

    expected = list(expected_rows)
    if len(set(expected)) != len(expected):
        raise ValueError("selected data list contains duplicate scene/center keys")
    actual = set(records)
    wanted = set(expected)
    missing = sorted(wanted - actual)
    extra = sorted(actual - wanted)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)} first={missing[:3]}")
        if extra:
            details.append(f"extra={len(extra)} first={extra[:3]}")
        raise ValueError("prediction/data-list key mismatch: " + ", ".join(details))


def file_sha256(path: str | Path) -> str:
    """Return a content digest without exposing the local file name or path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_list_sha256(path: str | Path) -> str:
    """Return the digest used to pin a prediction manifest to its data list."""

    return file_sha256(path)


class PredictionManifestWriter:
    """Stream one validated prediction manifest to a JSONL file."""

    def __init__(
        self,
        path: str | Path,
        *,
        data_list: str | Path | None = None,
        num_poses: int,
        interval_seconds: float,
    ) -> None:
        num_poses, interval_seconds = _validate_sampling(num_poses, interval_seconds)
        self.path = Path(path)
        self._handle = None
        self._keys: set[tuple[str, int]] = set()
        header: dict[str, Any] = {
            "format": PREDICTION_MANIFEST_FORMAT,
            "version": PREDICTION_MANIFEST_VERSION,
            "trajectory": {
                "num_poses": num_poses,
                "interval_seconds": interval_seconds,
                "pose_format": "x_y_heading",
            },
        }
        if data_list is not None:
            data_list_path = Path(data_list)
            if data_list_path.is_file():
                header["data_list_sha256"] = data_list_sha256(data_list_path)
        self.header = header
        self.num_poses = num_poses
        self.interval_seconds = interval_seconds

    def __enter__(self) -> "PredictionManifestWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._handle.write(json.dumps(self.header, separators=(",", ":")) + "\n")
        return self

    def write(self, scene: str, center: int, trajectory: Any) -> None:
        """Append one model trajectory after converting it to the public form."""

        if self._handle is None:
            raise RuntimeError("PredictionManifestWriter must be used as a context manager")
        scene, center = _validate_key(scene, center)
        key = (scene, center)
        if key in self._keys:
            raise ValueError(f"duplicate prediction key {key!r}")
        poses = trajectory_to_poses(trajectory, num_poses=self.num_poses)
        payload = {"scene": scene, "center": center, "poses": poses.tolist()}
        self._handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._keys.add(key)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "PREDICTION_MANIFEST_FORMAT",
    "PREDICTION_MANIFEST_VERSION",
    "PredictionManifest",
    "PredictionManifestWriter",
    "PredictionRecord",
    "file_sha256",
    "data_list_sha256",
    "load_prediction_manifest",
    "trajectory_to_poses",
    "validate_prediction_keys",
]
