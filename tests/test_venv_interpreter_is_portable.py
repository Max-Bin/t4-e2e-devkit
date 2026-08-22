"""A venv outside ``$HOME`` must not borrow an interpreter from inside it.

This repo is normally checked out on shared storage and run from several machines
-- a Slurm cluster where ``/home`` is per-node is the case that motivated this.
In that layout a ``.venv`` whose ``bin/python`` symlinks into ``$HOME`` is broken
everywhere except the node that created it, and it fails in a way that reads like
a missing file rather than a misconfigured environment::

    slurmstepd: error: execve(): /.../e2e-devkit/.venv/bin/python:
                No such file or directory

Every package in ``site-packages`` is reachable, ``sys.path`` is fine, the venv
looks intact -- there simply is no interpreter to start. The usual workaround is
to drive a different interpreter with ``PYTHONPATH`` pointed at the venv's
``site-packages``, which works and quietly gives up the venv.

The invariant asserted here is deliberately *not* "the interpreter must live at
some specific path". It is the portable half of that:

    if the venv is outside ``$HOME``, its interpreter must be too.

So a laptop checkout under ``~/code`` with a ``~/.local`` interpreter passes --
both sides move together, and nothing is shared. Only the mixed case fails, and
the mixed case is broken for any machine that does not share that ``$HOME``.

``uv`` recreates ``.venv`` from whichever interpreter it discovers, so the fix is
to give it a non-``$HOME`` install directory (``UV_PYTHON_INSTALL_DIR``) rather
than to repair the symlink by hand and wait for the next ``uv sync``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _venv_interpreters() -> list[Path]:
    """The ``.venv`` interpreter entry points that exist, resolved."""
    bin_dir = REPO_ROOT / ".venv" / "bin"
    return [path for path in (bin_dir / "python", bin_dir / "python3") if path.is_symlink()]


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def test_venv_interpreter_is_not_inside_home_when_the_venv_is_not():
    """The mixed case -- venv on shared storage, interpreter node-local -- fails."""
    interpreters = _venv_interpreters()
    if not interpreters:
        pytest.skip("no .venv in this checkout; nothing to constrain")

    home = os.environ.get("HOME")
    if not home:
        pytest.skip("HOME is unset, so there is no 'inside $HOME' to test against")
    home_path = Path(home).resolve()

    if _is_relative_to(REPO_ROOT, home_path):
        pytest.skip(
            f"checkout {REPO_ROOT} is itself under {home_path}, so an interpreter "
            "there moves with it"
        )

    offenders = []
    for interpreter in interpreters:
        # The raw link target matters as much as the resolved path: a dangling
        # symlink into $HOME cannot be resolved on the node where it is broken,
        # which is exactly the node where this needs to fail.
        target = Path(os.readlink(interpreter))
        if not target.is_absolute():
            target = (interpreter.parent / target).resolve()
        if _is_relative_to(target, home_path):
            offenders.append(f"{interpreter} -> {target}")

    assert not offenders, (
        "the venv lives outside $HOME but its interpreter is inside it, so this "
        "venv only works on the machine that built it:\n  "
        + "\n  ".join(offenders)
        + f"\n\n$HOME is {home_path}, the checkout is {REPO_ROOT}.\n"
        "Point uv at an interpreter that travels with the checkout, e.g.\n"
        "  export UV_PYTHON_INSTALL_DIR=<dir beside the checkout>\n"
        "  uv python install 3.12\n"
        "  uv sync\n"
        "Repairing .venv/bin/python by hand works until the next `uv sync`."
    )


def test_venv_interpreter_exists():
    """A symlink that resolves nowhere is the failure this guards against."""
    interpreters = _venv_interpreters()
    if not interpreters:
        pytest.skip("no .venv in this checkout")

    missing = [
        f"{interpreter} -> {os.readlink(interpreter)}"
        for interpreter in interpreters
        if not interpreter.exists()
    ]
    assert not missing, (
        "the venv interpreter symlink does not resolve on this machine:\n  "
        + "\n  ".join(missing)
    )
