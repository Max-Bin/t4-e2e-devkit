"""The batch PDM simulator, as the vendored scoring code addresses it.

``simulator.py``, ``batch_lqr*.py`` and ``batch_kinematic_bicycle.py`` in this
package are vendored -- ``VENDORED - do not edit by hand`` -- and the vendored
scorer imports ``simulate_proposals`` from the package, not from the submodule.
This file is the one part of the package the devkit owns, so the re-export lives
here: the alternative is a hand edit inside a vendored file whose source is
non-public, which ``tools/vendor.py check`` cannot detect drifting.
"""

from __future__ import annotations

from t4_e2e_devkit.planning.simulation.pdm_sim.simulator import simulate_proposals

__all__ = ["simulate_proposals"]
