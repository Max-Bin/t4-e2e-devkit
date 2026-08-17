# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : t4devkit:t4_devkit/typing/trajectory.py
# Commit : 4496b3e
# Tool   : tools/vendor.py
#
# Re-run ``python tools/vendor.py sync`` to update this file, and
# ``python tools/vendor.py check`` to detect drift against its source.
#
# Only ``import`` statements were rewritten; every numeric expression is
# byte-identical to the source. Edits belong upstream, or in a devkit module
# that wraps this one.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Trajectory"]


class Trajectory(np.ndarray):
    def __new__(cls, input_array: ArrayLike) -> Trajectory:
        # convert input to numpy array
        obj = np.array(input_array).view(cls)

        # validate the shape of the array
        if obj.ndim != 3 or obj.shape[2] != 3:
            raise ValueError(f"Trajectory must be the shape of (M, T, 3), got: {obj.shape}")

        return obj

    def __array_finalize__(self, obj: Trajectory | None) -> None:
        if obj is None:
            return

        if hasattr(obj, "shape") and (obj.ndim != 3 or obj.shape[2] != 3):
            # This can happen during slicing or other array operations
            # We'll allow it for now to maintain numpy compatibility
            pass
