# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/planning/simulation/simulation_time_controller/simulation_iteration.py
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

from t4_e2e_devkit.common.actor_state.state_representation import TimePoint


@dataclass
class SimulationIteration:
    """
    Simulation step time and index.
    """

    time_point: TimePoint  # A time point along simulation

    # Iteration in the simulation, starting from 0.
    # In open loop this represents the n-th sample in a scenario from the log.
    # In closed loop this represents the n-th sample of the simulation.
    index: int

    def __post_init__(self) -> None:
        """Post-init index sanity check."""
        assert self.index >= 0, f"Iteration must be >= 0, but it is {self.index}!"

    @property
    def time_us(self) -> int:
        """
        :return: time in micro seconds.
        """
        return int(self.time_point.time_us)

    @property
    def time_s(self) -> float:
        """
        :return: Time in seconds.
        """
        return float(self.time_point.time_s)
