# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : t4devkit:t4_devkit/typing/quaternion.py
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

from pyquaternion import Quaternion as PyQuaternion

__all__ = ["Quaternion"]


class Quaternion(PyQuaternion):
    """A quaternion class that wraps the PyQuaternion class.

    This wrapper exists to provide a consistent and explicit quaternion representation.

    Examples:
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q
        Quaternion(1.000000, 2.000000, 3.000000, 4.000000)
        >>> q.conjugate()
        Quaternion(1.000000, -2.000000, -3.000000, -4.000000)
        >>> q.norm()
        5.477226
        >>> q.inverse()
        Quaternion(0.181818, -0.363636, -0.545455, -0.727273)
        >>> q * q.inverse()
        Quaternion(1.000000, 0.000000, 0.000000, 0.000000)
    """

    pass
