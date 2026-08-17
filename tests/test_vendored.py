"""Guards on the vendored trees.

Several modules here are mechanical ports whose numbers were validated against a
reference -- the GPU PDM oracle against the CPU judge, ``tier4_metrics`` against
the source metric implementation. Two things can quietly undo that:

1. someone edits a vendored file in place, so it no longer matches its source;
2. an upstream repository changes and the devkit's copy silently goes stale.

``tools/vendor.py check`` catches both by re-deriving the transform and diffing.
This test runs it, so a drift is a red test rather than a discovery months later.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_TOOL = REPO_ROOT / "tools" / "vendor.py"


@pytest.mark.skipif(not VENDOR_TOOL.is_file(), reason="vendor tool absent (installed package)")
def test_vendored_files_match_their_sources():
    """Every vendored file is exactly its source plus an import rewrite."""
    result = subprocess.run(
        [sys.executable, str(VENDOR_TOOL), "check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if "source root missing" in result.stdout:
        pytest.skip("upstream sources are not reachable from this checkout")
    assert result.returncode == 0, f"vendored files drifted:\n{result.stdout}\n{result.stderr}"


def test_no_nuplan_dependency():
    """The devkit carries its own geometry vocabulary and must not import nuPlan.

    A nuPlan import reappearing means a vendored file was hand-edited or a new
    module reached for the upstream package -- which would drag a maps database
    and a Python version floor back in.
    """
    offenders = []
    for path in (REPO_ROOT / "t4_e2e_devkit").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "import nuplan" in stripped or stripped.startswith("from nuplan"):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {stripped}")
    assert not offenders, "nuplan imports found:\n" + "\n".join(offenders)


def test_no_upstream_repo_imports():
    """The devkit owns its code; external repositories are not runtime dependencies.

    An import of either would invert the dependency and make the devkit
    unusable without an external checkout.
    """
    offenders = []
    for path in (REPO_ROOT / "t4_e2e_devkit").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for package in ("navsim",):
                if stripped.startswith(f"from {package}") or stripped.startswith(f"import {package}"):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {stripped}")
    assert not offenders, "upstream repo imports found:\n" + "\n".join(offenders)


#: Directories whose contents are mechanically vendored.
VENDORED_ROOTS = (
    "common/actor_state",
    "common/geometry",
    "common/maps",
    "common/utils",
    "planning/simulation/trajectory",
    "planning/simulation/occupancy_map",
    "planning/simulation/path",
    "planning/simulation/pdm_sim",
    "planning/simulation/planner/pdm_planner",
    "evaluation/gpu",
    "evaluation/reference",
    "evaluation/tier4_metrics",
)

#: Devkit-authored modules that happen to live inside a vendored directory.
#: Listed by name rather than inferred, so adding one is a deliberate act and
#: an accidentally hand-edited vendored file still fails the test.
DEVKIT_AUTHORED_IN_VENDORED_TREES = frozenset(
    {
        # Copies a single function out of a nuPlan module whose remaining
        # contents pull in SimulationHistory; see the module docstring.
        "common/geometry/derivatives.py",
        # The adapter that feeds a T4Scene to TIER IV's metric functions.
        "evaluation/tier4_metrics/__init__.py",
        # GPU online PDM-Closed reference generation is devkit-owned; it lives
        # beside the mechanically vendored GPU kernels for one import surface.
        "evaluation/gpu/geometry_ops.py",
        "evaluation/gpu/reference.py",
    }
)


def test_vendored_files_carry_provenance():
    """Every vendored file has a header explaining how it is maintained."""
    missing = []
    for relative in VENDORED_ROOTS:
        directory = REPO_ROOT / "t4_e2e_devkit" / relative
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            key = str(path.relative_to(REPO_ROOT / "t4_e2e_devkit"))
            if key in DEVKIT_AUTHORED_IN_VENDORED_TREES:
                continue
            if path.name == "__init__.py" and path.stat().st_size == 0:
                continue  # empty package markers
            head = path.read_text(encoding="utf-8")[:600]
            if "VENDORED" not in head and "Source :" not in head:
                missing.append(key)
    assert not missing, "vendored files without a provenance header:\n" + "\n".join(missing)


def test_private_vendored_headers_withhold_source_identity():
    """Private inputs retain regeneration metadata without leaking provenance."""

    private_files = (
        "dataset/scene.py",
        "evaluation/ego_progress.py",
        "evaluation/oracle_evaluator.py",
        "evaluation/reference/pdm_closed.py",
    )
    for relative in private_files:
        head = (REPO_ROOT / "t4_e2e_devkit" / relative).read_text(encoding="utf-8")[:600]
        assert "Generated by : tools/vendor.py" in head
        assert "Source :" not in head
        assert "Commit :" not in head
