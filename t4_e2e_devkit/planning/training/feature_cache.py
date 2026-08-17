"""Content-addressed cache for derived numeric features.

Only builder outputs are cached.  Raw camera bytes, point-cloud payloads and
other byte strings are rejected at the boundary so a cache cannot silently
become a second sensor-data store.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np


class FeatureCache:
    """Atomic ``.npz`` cache keyed by sample identity and builder signature."""

    FORMAT = "t4-feature-cache-v1"

    def __init__(
        self,
        root: str | Path = "results/cache",
        *,
        version: str = "1",
        namespace: str = "features",
    ) -> None:
        if not str(version):
            raise ValueError("version must not be empty")
        if not str(namespace):
            raise ValueError("namespace must not be empty")
        self.root = Path(root)
        self.version = str(version)
        self.namespace = str(namespace)

    def key(
        self,
        sample_key: str,
        *,
        signature: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Return a stable digest for one sample and feature definition."""

        payload = json.dumps(
            {
                "namespace": self.namespace,
                "version": self.version if version is None else str(version),
                "sample": str(sample_key),
                "signature": None if signature is None else str(signature),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def path_for(self, key: str) -> Path:
        value = str(key)
        if len(value) < 16 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("feature cache keys must be hexadecimal digests")
        return self.root / self.namespace / value[:2] / f"{value}.npz"

    def load(self, key: str) -> Optional[dict[str, Any]]:
        """Load a mapping, returning ``None`` for a miss or incomplete file."""

        path = self.path_for(key)
        try:
            with np.load(path, allow_pickle=False) as archive:
                manifest_value = archive["__manifest__"]
                manifest = json.loads(str(np.asarray(manifest_value).item()))
                if (
                    manifest.get("format") != self.FORMAT
                    or str(manifest.get("version")) != self.version
                    or str(manifest.get("namespace")) != self.namespace
                ):
                    return None
                feature_specs = manifest.get("features")
                if not isinstance(feature_specs, Mapping):
                    return None
                result: dict[str, Any] = {}
                for name, spec in feature_specs.items():
                    if not isinstance(spec, Mapping):
                        return None
                    if name not in archive:
                        return None
                    value = np.array(archive[name], copy=True)
                    expected_shape = tuple(spec.get("shape", ()))
                    if tuple(value.shape) != expected_shape:
                        return None
                    if str(value.dtype) != str(spec.get("dtype")):
                        return None
                    if spec.get("backend") == "torch":
                        try:
                            import torch
                        except ImportError:
                            return value
                        result[name] = torch.from_numpy(value)
                    else:
                        result[name] = value
                return result
        except (EOFError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(self, key: str, features: Mapping[str, Any]) -> Path:
        """Atomically save a numeric feature mapping and return its path."""

        arrays: dict[str, np.ndarray] = {}
        manifest_features: dict[str, dict[str, Any]] = {}
        for name, value in features.items():
            feature_name = str(name)
            if not feature_name or feature_name.startswith("__"):
                raise ValueError(f"invalid feature name: {feature_name!r}")
            array, backend = _as_numeric_array(value)
            arrays[feature_name] = array
            manifest_features[feature_name] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "backend": backend,
            }
        manifest = {
            "format": self.FORMAT,
            "version": self.version,
            "namespace": self.namespace,
            "features": manifest_features,
        }
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".npz",
            dir=str(path.parent),
        )
        os.close(fd)
        try:
            np.savez_compressed(
                temporary_name,
                __manifest__=np.asarray(json.dumps(manifest, sort_keys=True)),
                **arrays,
            )
            os.replace(temporary_name, path)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        return path

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return a cached mapping or compute and atomically store it."""

        cached = self.load(key)
        if cached is not None:
            return cached
        features = dict(compute())
        self.save(key, features)
        return features


def _as_numeric_array(value: Any) -> tuple[np.ndarray, str]:
    """Convert NumPy/Torch-like tensors while rejecting raw sensor payloads."""

    if isinstance(value, (bytes, bytearray, memoryview, str)):
        raise TypeError("feature cache accepts numeric arrays, not raw sensor bytes or strings")
    backend = "numpy"
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
        backend = "torch"
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"feature value is not array-like: {type(value).__name__}") from error
    if array.dtype.kind in {"O", "S", "U", "V"}:
        raise TypeError(
            "feature cache only stores numeric/bool arrays; object/string/byte arrays "
            "may contain raw sensor payloads"
        )
    return np.ascontiguousarray(array), backend


__all__ = ["FeatureCache"]
