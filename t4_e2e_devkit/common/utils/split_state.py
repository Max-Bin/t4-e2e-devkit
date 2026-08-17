# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/common/utils/split_state.py
# Commit : e924167
# Tool   : tools/vendor.py
#
# Re-run ``python tools/vendor.py sync`` to update this file, and
# ``python tools/vendor.py check`` to detect drift against its source.
#
# Only ``import`` statements were rewritten; every numeric expression is
# byte-identical to the source. Edits belong upstream, or in a devkit module
# that wraps this one.
# =============================================================================

from dataclasses import dataclass
from typing import Any, List


@dataclass
class SplitState:
    """Dataclass representing a state split between fixed states, linear states and angular states."""

    linear_states: List[Any]  # Variable states
    angular_states: List[float]  # Variable states, representing angles, with 2pi period
    fixed_states: List[Any]  # Constant states

    def __len__(self) -> int:
        """Returns the number of states"""
        return len(self.linear_states) + len(self.angular_states) + len(self.fixed_states)
