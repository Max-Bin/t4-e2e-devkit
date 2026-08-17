"""Portable trajectory submissions and local validation.

The format is intentionally small and model-agnostic: one trajectory per
scenario token, with its own sampling metadata.  It is suitable for internal
evaluation, rank merging, and archiving without serializing scene data or
requiring an experiment-tracking service.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from t4_e2e_devkit.common.dataclasses import Trajectory
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

SUBMISSION_FORMAT = "t4.trajectory-submission"
SUBMISSION_VERSION = 1
MANIFEST_NAME = "manifest.json"
PREDICTIONS_NAME = "predictions.jsonl"


@dataclass(frozen=True)
class TrajectorySubmission:
    """One validated prediction keyed by a stable data-list token."""

    token: str
    trajectory: Trajectory
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        token = str(self.token)
        if not token:
            raise ValueError("submission token must not be empty")
        if not isinstance(self.trajectory, Trajectory):
            raise TypeError("submission trajectory must be a Trajectory")
        poses = np.asarray(self.trajectory.poses, dtype=np.float64)
        if not np.isfinite(poses).all():
            raise ValueError(f"submission trajectory for {token!r} contains non-finite values")
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def sampling(self) -> TrajectorySampling:
        return self.trajectory.trajectory_sampling

    def as_dict(self) -> dict[str, Any]:
        sampling = self.sampling
        return {
            "token": self.token,
            "poses": np.asarray(self.trajectory.poses, dtype=np.float32).tolist(),
            "sampling": {
                "num_poses": int(sampling.num_poses),
                "time_horizon": float(sampling.time_horizon),
                "interval_length": float(sampling.interval_length),
            },
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectorySubmission":
        try:
            sampling_value = value["sampling"]
            sampling = TrajectorySampling(
                num_poses=int(sampling_value["num_poses"]),
                time_horizon=float(sampling_value["time_horizon"]),
                interval_length=float(sampling_value["interval_length"]),
            )
            trajectory = Trajectory(
                poses=np.asarray(value["poses"], dtype=np.float32),
                trajectory_sampling=sampling,
            )
            return cls(
                token=str(value["token"]),
                trajectory=trajectory,
                metadata=value.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid trajectory submission entry") from error


@dataclass(frozen=True)
class SubmissionValidation:
    """Machine-readable validation result."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checked: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("invalid submission: " + "; ".join(self.errors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked": self.checked,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SubmissionPackage:
    """A set of trajectories and immutable run metadata."""

    entries: tuple[TrajectorySubmission, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(not isinstance(entry, TrajectorySubmission) for entry in entries):
            raise TypeError("submission entries must be TrajectorySubmission objects")
        tokens = [entry.token for entry in entries]
        if len(tokens) != len(set(tokens)):
            duplicates = sorted(token for token in set(tokens) if tokens.count(token) > 1)
            raise ValueError(f"submission contains duplicate tokens: {duplicates}")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda item: item.token)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(entry.token for entry in self.entries)

    def entry(self, token: str) -> TrajectorySubmission:
        wanted = str(token)
        for entry in self.entries:
            if entry.token == wanted:
                return entry
        raise KeyError(wanted)

    def validate(
        self,
        *,
        expected_tokens: Optional[Iterable[str]] = None,
        allow_extra: bool = False,
        min_horizon_s: Optional[float] = None,
        max_horizon_s: Optional[float] = None,
    ) -> SubmissionValidation:
        """Validate token coverage and every trajectory's numerical contract."""

        errors: list[str] = []
        warnings: list[str] = []
        expected = None if expected_tokens is None else {str(token) for token in expected_tokens}
        actual = set(self.tokens)
        if expected is not None:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing:
                errors.append(f"missing tokens: {missing[:8]}" + (" ..." if len(missing) > 8 else ""))
            if extra and not allow_extra:
                errors.append(f"unexpected tokens: {extra[:8]}" + (" ..." if len(extra) > 8 else ""))
        if min_horizon_s is not None and min_horizon_s <= 0.0:
            raise ValueError("min_horizon_s must be positive")
        if max_horizon_s is not None and max_horizon_s <= 0.0:
            raise ValueError("max_horizon_s must be positive")
        if min_horizon_s is not None and max_horizon_s is not None and min_horizon_s > max_horizon_s:
            raise ValueError("min_horizon_s must not exceed max_horizon_s")
        for entry in self.entries:
            sampling = entry.sampling
            horizon = float(sampling.time_horizon)
            if sampling.num_poses is None or sampling.num_poses < 1:
                errors.append(f"{entry.token}: sampling must contain at least one pose")
            if sampling.interval_length is None or sampling.interval_length <= 0.0:
                errors.append(f"{entry.token}: sampling interval must be positive")
            if min_horizon_s is not None and horizon + 1.0e-9 < min_horizon_s:
                errors.append(f"{entry.token}: horizon {horizon:g}s is shorter than {min_horizon_s:g}s")
            if max_horizon_s is not None and horizon - 1.0e-9 > max_horizon_s:
                warnings.append(f"{entry.token}: horizon {horizon:g}s exceeds {max_horizon_s:g}s")
        return SubmissionValidation(tuple(errors), tuple(warnings), len(self.entries))

    def write(self, directory: str | Path, *, overwrite: bool = True) -> Path:
        """Write an atomic manifest and JSONL prediction stream."""

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        if not overwrite and any((output / name).exists() for name in (MANIFEST_NAME, PREDICTIONS_NAME)):
            raise FileExistsError(f"submission directory already contains an artifact: {output}")
        validation = self.validate()
        validation.raise_for_errors()
        prediction_bytes = "".join(
            json.dumps(entry.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            for entry in self.entries
        ).encode("utf-8")
        manifest = {
            "format": SUBMISSION_FORMAT,
            "version": SUBMISSION_VERSION,
            "num_entries": len(self.entries),
            "tokens_sha256": _tokens_digest(self.tokens),
            "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "metadata": _jsonable(self.metadata),
        }
        _atomic_bytes(output / PREDICTIONS_NAME, prediction_bytes)
        _atomic_bytes(
            output / MANIFEST_NAME,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return output

    @classmethod
    def read(cls, directory: str | Path, *, validate: bool = True) -> "SubmissionPackage":
        source = Path(directory)
        try:
            manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read submission manifest: {source}") from error
        if manifest.get("format") != SUBMISSION_FORMAT or manifest.get("version") != SUBMISSION_VERSION:
            raise ValueError(f"unsupported submission format: {source}")
        prediction_path = source / PREDICTIONS_NAME
        try:
            raw = prediction_path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read submission predictions: {prediction_path}") from error
        if hashlib.sha256(raw).hexdigest() != manifest.get("predictions_sha256"):
            raise ValueError(f"submission predictions digest mismatch: {prediction_path}")
        entries = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                entries.append(TrajectorySubmission.from_dict(value))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid submission entry at line {line_number}") from error
        package = cls(tuple(entries), metadata=manifest.get("metadata", {}))
        if int(manifest.get("num_entries", -1)) != len(package.entries):
            raise ValueError("submission manifest entry count does not match predictions")
        if manifest.get("tokens_sha256") != _tokens_digest(package.tokens):
            raise ValueError("submission token digest does not match predictions")
        if validate:
            package.validate().raise_for_errors()
        return package

    @classmethod
    def merge(
        cls,
        directories: Sequence[str | Path],
        output_dir: str | Path,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Merge rank packages and reject duplicate or missing predictions."""

        if not directories:
            raise ValueError("at least one submission directory is required")
        packages = [cls.read(directory) for directory in directories]
        entries = tuple(entry for package in packages for entry in package.entries)
        merged_metadata = dict(packages[0].metadata)
        merged_metadata.pop("rank", None)
        merged_metadata["merged"] = True
        merged_metadata["source_packages"] = len(packages)
        merged = cls(entries, metadata=metadata or merged_metadata)
        return merged.write(output_dir)


def _tokens_digest(tokens: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(token) for token in tokens)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("submission metadata must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"submission metadata is not JSON serializable: {type(value).__name__}")


__all__ = [
    "MANIFEST_NAME",
    "PREDICTIONS_NAME",
    "SUBMISSION_FORMAT",
    "SUBMISSION_VERSION",
    "SubmissionPackage",
    "SubmissionValidation",
    "TrajectorySubmission",
]
