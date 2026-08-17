"""Explicit CPU/offline PDM-Closed reference labels.

The GPU scorer owns cache-free online reference generation. This provider is
kept for explicit CPU/offline runs: a valid cache is read first; missing or
incomplete centers are computed from the T4 scene on demand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from t4_e2e_devkit.evaluation.reference.pdm_closed import (
    T4PDMReferenceCache,
    T4PDMReferenceConfig,
    T4PDMReferenceResult,
    compute_t4_pdm_reference,
)

LOGGER = logging.getLogger(__name__)


class T4PDMReferenceProvider:
    """Serve explicit CPU labels from cache or compute them lazily.

    The online path intentionally keeps results in the owning scene reader's
    memory. Repeated requests for one center therefore do not rerun the
    closed-loop reference, while no disk cache is created as a side effect of
    evaluation.
    """

    def __init__(
        self,
        reader: Any,
        root: str | Path,
        *,
        cache_root: str | Path | None = None,
        config: T4PDMReferenceConfig | None = None,
        verify_source: bool = True,
    ) -> None:
        self.reader = reader
        self.root = Path(root)
        self.config = config or T4PDMReferenceConfig()
        self.config.validate()
        self.cache_root = None if cache_root in (None, "", "null", "None") else Path(cache_root)
        self._cache: T4PDMReferenceCache | None = None
        if self.cache_root is not None:
            self._cache = T4PDMReferenceCache.open(
                Path(reader.scene_dir),
                self.root,
                self.cache_root,
                config=self.config,
                require=False,
                verify_source=verify_source,
            )
        self._online: dict[int, T4PDMReferenceResult] = {}
        self._warned_online = False
        self.cache_hits = 0
        self.online_computations = 0

    def frame(self, center: int) -> dict[str, Any]:
        """Return one complete reference label for ``center``.

        A cache may be partial, so an invalid or absent cached center follows
        the same online path as a completely absent cache.
        """

        index = int(center)
        if self._cache is not None:
            try:
                cached = self._cache.frame(index)
            except (IndexError, KeyError, ValueError):
                pass
            else:
                self.cache_hits += 1
                return cached

        result = self._online.get(index)
        if result is None:
            if not self._warned_online:
                cache_state = (
                    "no cache was configured"
                    if self.cache_root is None
                    else "the cache is missing, incomplete, or incompatible"
                )
                LOGGER.warning(
                    "Computing T4 PDM-Closed references online for %s because %s; "
                    "uncached scoring is substantially slower",
                    self.reader.scene_dir,
                    cache_state,
                )
                self._warned_online = True
            result = compute_t4_pdm_reference(self.reader, index, self.config)
            self._online[index] = result
            self.online_computations += 1
        return _result_to_frame(result)

    def close(self) -> None:
        """Release cache mappings and online labels."""

        if self._cache is not None:
            self._cache.close()
            self._cache = None
        self._online.clear()


def _result_to_frame(result: T4PDMReferenceResult) -> dict[str, Any]:
    """Convert an online result to the cache reader's public payload shape."""

    return {
        "pdm_progress": float(result.pdm_progress),
        "reference_trajectory": np.asarray(result.reference_trajectory, dtype=np.float32).copy(),
        "selected_proposal": int(result.selected_proposal),
        "proposal_scores": np.asarray(result.proposal_scores, dtype=np.float32).copy(),
        "reference_nc": float(result.reference_nc),
        "reference_dac": float(result.reference_dac),
        "reference_raw_progress": float(result.reference_raw_progress),
    }


__all__ = ["T4PDMReferenceProvider"]
