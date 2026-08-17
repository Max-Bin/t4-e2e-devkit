#!/usr/bin/env python3
"""Vendor validated numeric source files into the devkit.

The devkit owns its runtime surface and has no model-repository dependency.
Several modules here are ports whose *numbers* have already been validated
against an external reference — the GPU PDM oracle was checked window by
window against the CPU judge, and the TIER IV metrics were migrated verbatim
so the figures stay comparable. Re-typing those by hand would silently throw
that validation away.

So they are vendored mechanically instead: copied byte-for-byte, with only
``import`` statements rewritten.  Public sources receive a provenance header;
private or user-provided sources receive only a generic generated header.
``--check`` re-runs the transform and diffs it against what is on disk, so a
drift between the devkit and its source is a test failure rather than a
discovery made months later.

Usage::

    python tools/vendor.py sync           # (re)generate every vendored file
    python tools/vendor.py sync --only common
    python tools/vendor.py check          # fail if any vendored file drifted
    python tools/vendor.py status         # show what is vendored from where
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "t4_e2e_devkit"

# --------------------------------------------------------------------------- #
# External roots. Overridable so a checkout on another machine can vendor
# without reproducing this exact directory layout. Only sources in
# ``PUBLIC_SOURCES`` may appear in generated provenance headers.
# --------------------------------------------------------------------------- #

SOURCES: dict[str, Path] = {
    "nuplan": Path(
        os.environ.get("T4E2E_NUPLAN_SRC", REPO_ROOT / "references" / "nuplan-devkit")
    ),
    # TIER IV's public devkit. Its transform helpers are vendored below.
    "t4devkit": Path(
        os.environ.get("T4E2E_T4DEVKIT_SRC", REPO_ROOT / "references" / "t4-devkit")
    ),
}

# --------------------------------------------------------------------------- #
# Import rewrites.
#
# Order matters: the first pattern that matches a line wins, so the most
# specific module paths are listed first.  Every rewrite is a pure module-path
# substitution — no numeric code is touched.
# --------------------------------------------------------------------------- #

_REFERENCE_AGENT_PACKAGE = r"[A-Za-z][A-Za-z0-9_]*"

REWRITES: list[tuple[str, str]] = [
    # -- nuplan -> devkit common ------------------------------------------- #
    (r"\bnuplan\.common\.actor_state\b", "t4_e2e_devkit.common.actor_state"),
    (r"\bnuplan\.common\.geometry\b", "t4_e2e_devkit.common.geometry"),
    (r"\bnuplan\.common\.maps\b", "t4_e2e_devkit.common.maps"),
    (r"\bnuplan\.common\.utils\b", "t4_e2e_devkit.common.utils"),
    # ``approximate_derivatives`` is the only symbol the devkit needs out of
    # nuPlan's metric helpers, and the module it lives in drags in nuPlan's
    # SimulationHistory.  It is re-homed next to the geometry it belongs with.
    (
        r"\bnuplan\.planning\.metrics\.utils\.state_extractors\b",
        "t4_e2e_devkit.common.geometry.derivatives",
    ),
    (r"\bnuplan\.planning\.simulation\.path\b", "t4_e2e_devkit.planning.simulation.path"),
    (
        r"\bnuplan\.planning\.simulation\.trajectory\b",
        "t4_e2e_devkit.planning.simulation.trajectory",
    ),
    (
        r"\bnuplan\.planning\.simulation\.observation\.observation_type\b",
        "t4_e2e_devkit.planning.simulation.observation.observation_type",
    ),
    (
        r"\bnuplan\.planning\.simulation\.observation\.idm\.utils\b",
        "t4_e2e_devkit.planning.simulation.observation.idm_utils",
    ),
    (
        r"\bnuplan\.planning\.simulation\.occupancy_map\b",
        "t4_e2e_devkit.planning.simulation.occupancy_map",
    ),
    (
        r"\bnuplan\.planning\.simulation\.simulation_time_controller\.simulation_iteration\b",
        "t4_e2e_devkit.planning.simulation.simulation_iteration",
    ),
    (
        r"\bnuplan\.planning\.simulation\.planner\.abstract_planner\b",
        "t4_e2e_devkit.planning.simulation.planner.abstract_planner",
    ),
    (
        r"\bnuplan\.planning\.simulation\.planner\.ml_planner\.transform_utils\b",
        "t4_e2e_devkit.planning.simulation.planner.transform_utils",
    ),
    (
        r"\bnuplan\.planning\.metrics\.utils\.collision_utils\b",
        "t4_e2e_devkit.planning.simulation.observation.collision_utils",
    ),
    (
        r"\bnuplan\.planning\.scenario_builder\.abstract_scenario\b",
        "t4_e2e_devkit.planning.scenario_builder.abstract_scenario",
    ),
    # -- external navigation reference -> devkit ---------------------------- #
    (
        r"\bnavsim\.planning\.simulation\.planner\.pdm_planner\b",
        "t4_e2e_devkit.planning.simulation.planner.pdm_planner",
    ),
    (r"\bnavsim\.common\.dataclasses\b", "t4_e2e_devkit.common.dataclasses"),
    (r"\bnavsim\.common\.enums\b", "t4_e2e_devkit.common.enums"),
    (r"\bnavsim\.evaluate\.pdm_score\b", "t4_e2e_devkit.evaluation.trajectory_transform"),
    (
        rf"\bnavsim\.agents\.{_REFERENCE_AGENT_PACKAGE}\.t4_gpu_oracle\b",
        "t4_e2e_devkit.evaluation.gpu",
    ),
    (
        rf"\bnavsim\.agents\.{_REFERENCE_AGENT_PACKAGE}\.t4_pdm_closed\b",
        "t4_e2e_devkit.evaluation.reference.pdm_closed",
    ),
    (
        rf"\bnavsim\.agents\.{_REFERENCE_AGENT_PACKAGE}\.t4_oracle_evaluator\b",
        "t4_e2e_devkit.evaluation.oracle_evaluator",
    ),
    (
        rf"\bnavsim\.agents\.{_REFERENCE_AGENT_PACKAGE}\.score_module\.train_pdm_scorer\b",
        "t4_e2e_devkit.evaluation.reference.train_pdm_scorer",
    ),
    (rf"\bnavsim\.agents\.{_REFERENCE_AGENT_PACKAGE}\.t4_data\b", "t4_e2e_devkit.dataset.scene"),
    (rf"\bnavsim\.agents\.{_REFERENCE_AGENT_PACKAGE}\.t4_ep\b", "t4_e2e_devkit.evaluation.ego_progress"),
    # The external package's siblings were relative imports; vendoring splits
    # that one directory across ``evaluation/`` and ``evaluation/reference/``,
    # so the relative form no longer resolves and is made absolute here.
    (r"\bfrom \.t4_pdm_closed\b", "from t4_e2e_devkit.evaluation.reference.pdm_closed"),
    (r"\bfrom \.t4_data\b", "from t4_e2e_devkit.dataset.scene"),
    (r"\bfrom \.t4_ep\b", "from t4_e2e_devkit.evaluation.ego_progress"),
    (r"\bfrom \.t4_gpu_oracle\b", "from t4_e2e_devkit.evaluation.gpu"),
    # -- TIER IV's own devkit ------------------------------------------------ #
    #
    # Its transform machinery is the reference for T4 sensor geometry, so the
    # devkit uses it rather than re-deriving the conventions. Vendored instead of
    # depended on: `t4-devkit` pulls rerun-sdk, pycocotools, pyarrow and pypcd4,
    # and `import t4_devkit.dataclass.transform` triggers the whole chain -- the
    # same weight problem that kept nuplan out. The closure taken here needs only
    # numpy, pyquaternion and attrs.
    (r"\bt4_devkit\.dataclass\.transform\b", "t4_e2e_devkit.common.tier4.transform"),
    (r"\bt4_devkit\.common\.converter\b", "t4_e2e_devkit.common.tier4.converter"),
    (r"\bt4_devkit\.common\.geometry\b", "t4_e2e_devkit.common.tier4.geometry"),
    (r"\bt4_devkit\.typing\b", "t4_e2e_devkit.common.tier4.typing"),
]

# A comment block, not a docstring: many vendored modules open with
# ``from __future__ import annotations``, which must be the first statement
# after the module docstring.  Prepending a second string literal would demote
# that import to a non-leading position and the file would stop parsing.
# --------------------------------------------------------------------------- #
# Semantic patches.
#
# Kept strictly separate from REWRITES, which are import-path substitutions only.
# A patch changes what a vendored file *does*, so each one needs a reason, is
# applied to exactly one file, and fails loudly if its anchor text disappears
# upstream -- silently not applying is how a fix gets lost in a re-sync.
#
# Keep this list short. A patch is a statement that the upstream constant is
# wrong for the devkit's data, not a convenient place to edit vendored code.
# --------------------------------------------------------------------------- #

PATCHES: dict[str, list[tuple[str, str, str]]] = {
    # (reason, anchor, replacement)
    "dataset/scene.py": [
        (
            "The reader's hardcoded camera vocabulary predates x2_dev and lacks "
            "CAM_BACK, the two roof views and CAM_TRAFFIC_LIGHT_NEAR, so it "
            "rejects every x2_dev scene. The devkit's constant is the fleet-wide "
            "vocabulary and the register itself is validated against a scene's "
            "readable cameras in dataset/rigs.py.",
            '''T4_ALL_CAMERA_NAMES = (
    "CAM_BACK_LEFT",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT",
    "CAM_BACK_RIGHT_WIDE",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_FRONT_WIDE",
    "CAM_TRAFFIC_LIGHT_FAR",
)''',
            "# PATCHED by tools/vendor.py: fleet-wide vocabulary, see PATCHES.\n"
            "from t4_e2e_devkit.common.constants import (  # noqa: E402\n"
            "    T4_ALL_CAMERA_NAMES,\n"
            "    T4_DEFAULT_CAMERA_NAMES as _T4_DEFAULT_CAMERA_NAMES_UNUSED,\n"
            ")",
        ),
    ],
}


def apply_patches(text: str, dst: str) -> str:
    """Apply the semantic patches registered for one destination file."""
    for reason, anchor, replacement in PATCHES.get(dst, []):
        if anchor not in text:
            raise SystemExit(
                f"vendor patch no longer applies to {dst}: its anchor text is gone "
                f"upstream. Re-check whether the patch is still needed.\n  reason: {reason}"
            )
        text = text.replace(anchor, replacement, 1)
    return text


HEADER = """# =============================================================================
# VENDORED - do not edit by hand.
#
# Generated by : tools/vendor.py
# Source details are intentionally omitted for non-public inputs.
# Tool   : tools/vendor.py
#
# Re-run ``python tools/vendor.py sync`` to update this file, and
# ``python tools/vendor.py check`` to detect drift against its source.
#
# Only ``import`` statements were rewritten; every numeric expression is
# byte-identical to the source.  Edits belong upstream, or in a devkit module
# that wraps this one.
# =============================================================================

"""

PUBLIC_HEADER = """# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : {origin}
# Commit : {commit}
# Tool   : tools/vendor.py
#
# Re-run ``python tools/vendor.py sync`` to update this file, and
# ``python tools/vendor.py check`` to detect drift against its source.
#
# Only ``import`` statements were rewritten; every numeric expression is
# byte-identical to the source. Edits belong upstream, or in a devkit module
# that wraps this one.
# =============================================================================

"""


@dataclass
class Item:
    """One vendored file or directory."""

    group: str
    source: str  # key into SOURCES
    src: str  # path relative to the source root
    dst: str  # path relative to the devkit package
    recursive: bool = False
    exclude: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# The manifest.
# --------------------------------------------------------------------------- #

MANIFEST: list[Item] = [
    # ---- common: the geometry/state vocabulary, lifted out of nuplan ------- #
    Item("common", "nuplan", "nuplan/common/actor_state", "common/actor_state", recursive=True,
         exclude=("test", "BUILD")),
    Item("common", "nuplan", "nuplan/common/geometry", "common/geometry", recursive=True,
         exclude=("test", "BUILD")),
    Item("common", "nuplan", "nuplan/common/maps/maps_datatypes.py", "common/maps/maps_datatypes.py"),
    Item("common", "nuplan", "nuplan/common/maps/abstract_map.py", "common/maps/abstract_map.py"),
    Item("common", "nuplan", "nuplan/common/maps/abstract_map_objects.py",
         "common/maps/abstract_map_objects.py"),
    Item("common", "nuplan", "nuplan/common/utils/interpolatable_state.py",
         "common/utils/interpolatable_state.py"),
    Item("common", "nuplan", "nuplan/common/utils/split_state.py", "common/utils/split_state.py"),
    # ---- planning: trajectory + observation vocabulary --------------------- #
    Item("planning", "nuplan", "nuplan/planning/simulation/trajectory",
         "planning/simulation/trajectory", recursive=True, exclude=("test", "BUILD")),
    Item("planning", "nuplan", "nuplan/planning/simulation/observation/idm/utils.py",
         "planning/simulation/observation/idm_utils.py"),
    Item("planning", "nuplan", "nuplan/planning/simulation/path",
         "planning/simulation/path", recursive=True, exclude=("test", "BUILD")),
    Item("planning", "nuplan", "nuplan/planning/simulation/occupancy_map",
         "planning/simulation/occupancy_map", recursive=True, exclude=("test", "BUILD")),
    Item("planning", "nuplan",
         "nuplan/planning/simulation/simulation_time_controller/simulation_iteration.py",
         "planning/simulation/simulation_iteration.py"),
    Item("planning", "nuplan", "nuplan/planning/simulation/planner/ml_planner/transform_utils.py",
         "planning/simulation/planner/transform_utils.py"),
    # NOTE: ``observation_type``, ``collision_utils`` and ``abstract_planner``
    # are deliberately NOT vendored.  Their nuPlan originals import the metric
    # statistics stack, the sensor database and the simulation history buffer
    # for functionality the devkit does not have; the devkit ships small native
    # equivalents instead (same class and field names, T4 sensor channels).
    # ---- PDM planner: maintained in the devkit ----------------------------- #
    # ---- TIER IV's transform machinery ------------------------------------ #
    #
    # The validated T4 sensor geometry: HomogeneousMatrix and
    # TransformBuffer, plus the typing and quaternion helpers they need. Used for
    # the camera/LiDAR timestamp correction, where getting a frame convention
    # subtly wrong would move objects on the image rather than raise.
    Item("tier4", "t4devkit", "t4_devkit/typing", "common/tier4/typing", recursive=True),
    Item("tier4", "t4devkit", "t4_devkit/common/converter.py", "common/tier4/converter.py"),
    Item("tier4", "t4devkit", "t4_devkit/dataclass/transform.py", "common/tier4/transform.py"),
    # ---- dataset: the T4 container readers -------------------------------- #
    #
    # The CPU reference judge binds to this reader by name -- it calls
    # ``_gt_frame``, ``trajectory``, ``_bridge_stationary_boxes`` and
    # ``_transform_agent_boxes_to_center`` directly -- so the module is vendored
    # rather than restructured, and the judge's contract holds byte-for-byte.
    # The devkit's own unified surface (``T4Scene``, ``SensorConfig``-driven
    # decoding) is built on top of it in ``dataset/window.py``.
    # The T4 reader and scoring kernels are maintained in the devkit. Their
    # private research sources are intentionally not part of this manifest.
    # ---- evaluation: the three scoring families --------------------------- #
]

# Public provenance is opt-in. Private or user-provided reference checkouts are
# still checked, but their identity and revision never enter generated files or
# status output.
PUBLIC_SOURCES = frozenset({"nuplan", "t4devkit"})


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def rewrite_imports(text: str) -> str:
    """Apply every module-path rewrite to ``text``."""
    for pattern, replacement in REWRITES:
        text = re.sub(pattern, replacement, text)
    return text


def transform(
    text: str,
    origin: str,
    commit: str,
    dst: str = "",
    include_provenance: bool = False,
) -> str:
    """Header + rewritten imports + registered patches.

    Pure function of its inputs, so ``check`` re-derives exactly what ``sync``
    wrote and a hand edit shows up as a diff.
    """
    body = apply_patches(rewrite_imports(text), dst)
    header = PUBLIC_HEADER if include_provenance else HEADER
    if not body:
        # Keep generated package markers free of an otherwise meaningless blank
        # line at EOF when the upstream __init__.py is empty.
        return header.format(origin=origin, commit=commit).rstrip("\n") + "\n"
    return header.format(origin=origin, commit=commit) + body


def _iter_files(item: Item, src_root: Path) -> list[tuple[Path, Path]]:
    """Resolve one manifest item into (absolute source, package-relative dest)."""
    if any(char in item.src for char in "*?["):
        sources = sorted(src_root.glob(item.src))
    else:
        sources = [src_root / item.src]
    pairs: list[tuple[Path, Path]] = []
    for src in sources:
        if not item.recursive:
            pairs.append((src, Path(item.dst)))
            continue
        for path in sorted(src.rglob("*.py")):
            rel = path.relative_to(src)
            if any(part in item.exclude for part in rel.parts) or rel.name in item.exclude:
                continue
            pairs.append((path, Path(item.dst) / rel))
    return pairs


def resolve(only: str | None) -> list[tuple[Item, Path, Path, str]]:
    """Every (item, src, dst, origin) tuple the manifest expands to."""
    resolved = []
    for item in MANIFEST:
        if only and item.group != only:
            continue
        src_root = SOURCES[item.source]
        if not src_root.exists():
            print(f"  ! source root missing, skipping {item.group}/{item.src}: {src_root}")
            continue
        for src, dst in _iter_files(item, src_root):
            if not src.exists():
                print(f"  ! missing {src}")
                continue
            origin_path = (
                item.src
                if any(char in item.src for char in "*?[")
                else str(src.relative_to(src_root))
            )
            origin = f"{item.source}:{origin_path}"
            resolved.append((item, src, dst, origin))
    return resolved


def cmd_sync(args: argparse.Namespace) -> int:
    commits = {key: _git_commit(root) for key, root in SOURCES.items() if root.exists()}
    written = 0
    for item, src, dst, origin in resolve(args.only):
        out = PACKAGE / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        out.write_text(
            transform(
                text,
                origin,
                commits.get(item.source, "unknown"),
                str(dst),
                include_provenance=item.source in PUBLIC_SOURCES,
            ),
            encoding="utf-8",
        )
        written += 1
    # every vendored package needs an __init__ to be importable
    for pkg_dir in {(PACKAGE / dst).parent for _, _, dst, _ in resolve(args.only)}:
        init = pkg_dir / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    print(f"vendored {written} file(s)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    commits = {key: _git_commit(root) for key, root in SOURCES.items() if root.exists()}
    drifted: list[str] = []
    for item, src, dst, origin in resolve(args.only):
        out = PACKAGE / dst
        if not out.exists():
            drifted.append(f"{dst}: not vendored yet")
            continue
        expected = transform(
            src.read_text(encoding="utf-8"),
            origin,
            commits.get(item.source, "unknown"),
            str(dst),
            include_provenance=item.source in PUBLIC_SOURCES,
        )
        actual = out.read_text(encoding="utf-8")
        if expected != actual:
            drifted.append(
                f"{dst}: differs from {origin if item.source in PUBLIC_SOURCES else 'private source'}"
            )
            if args.diff:
                for line in list(
                    difflib.unified_diff(
                        actual.splitlines(), expected.splitlines(),
                        fromfile=f"devkit/{dst}",
                        tofile=origin if item.source in PUBLIC_SOURCES else "private source",
                        lineterm="",
                        n=2,
                    )
                )[:40]:
                    print(line)
    if drifted:
        print("VENDOR DRIFT:")
        for line in drifted:
            print(f"  {line}")
        return 1
    print("vendored files are in sync with their sources")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    for key, root in SOURCES.items():
        mark = "ok " if root.exists() else "MISSING"
        if key in PUBLIC_SOURCES:
            detail = f"{root}  ({_git_commit(root) if root.exists() else '-'})"
        else:
            detail = "private source (identity withheld)"
        print(f"[{mark}] {key:12s} {detail}")
    print()
    by_group: dict[str, int] = {}
    for item, _, _, _ in resolve(None):
        by_group[item.group] = by_group.get(item.group, 0) + 1
    for group, count in sorted(by_group.items()):
        print(f"  {group:10s} {count:3d} file(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("sync", cmd_sync), ("check", cmd_check), ("status", cmd_status)):
        p = sub.add_parser(name)
        p.add_argument("--only", help="restrict to one manifest group")
        if name == "check":
            p.add_argument("--diff", action="store_true", help="print the first lines of each diff")
        p.set_defaults(handler=handler)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
