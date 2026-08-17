"""Small file-backed cache for deterministic metric records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional


class MetricCache:
    """Content-addressed JSON cache with atomic writes.

    The cache stores metric outputs only. Inputs such as scenes and sensor
    payloads remain outside it; callers provide a stable token and optional
    configuration signature to invalidate stale records.
    """

    def __init__(self, root: str | Path, *, namespace: str = "metrics") -> None:
        self.root = Path(root)
        self.namespace = str(namespace)

    def key(
        self,
        token: str,
        metric_name: str,
        signature: Optional[str] = None,
    ) -> str:
        payload = json.dumps(
            {
                "namespace": self.namespace,
                "token": str(token),
                "metric": str(metric_name),
                "signature": signature,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def load(self, key: str) -> Optional[Mapping[str, Any]]:
        path = self.path_for(key)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, Mapping) else None

    def save(self, key: str, value: Mapping[str, Any]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        return path

    def path_for(self, key: str) -> Path:
        value = str(key)
        if len(value) < 16 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("metric cache keys must be hexadecimal digests")
        return self.root / self.namespace / value[:2] / f"{value}.json"


__all__ = ["MetricCache"]
