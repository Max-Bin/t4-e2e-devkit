# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : t4devkit:t4_devkit/common/converter.py
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

from typing import TYPE_CHECKING

import numpy as np

from t4_e2e_devkit.common.tier4.typing import Quaternion

if TYPE_CHECKING:
    from t4_e2e_devkit.common.tier4.typing import RotationLike

__all__ = ["to_quaternion"]


def to_quaternion(x: RotationLike) -> Quaternion:
    """Convert input rotation like array to `Quaternion`.

    Args:
        x (RotationLike): Rotation matrix or quaternion.

    Returns:
        Quaternion: Converted instance.
    """
    return Quaternion(matrix=x) if isinstance(x, np.ndarray) and x.ndim == 2 else Quaternion(x)
