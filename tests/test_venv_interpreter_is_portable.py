"""A venv outside ``$HOME`` must not borrow an interpreter from inside it.

On a cluster where ``/home`` is per-node, a ``.venv`` on shared storage whose
``bin/python`` symlinks into ``$HOME`` starts only on the machine that built it.
Elsewhere it fails as ``execve(): .../.venv/bin/python: No such file or
directory`` -- which reads as a missing file, not a misconfigured environment,
because ``site-packages`` is intact and only the interpreter is gone.

``uv`` chooses an interpreter when it *creates* the venv and keeps whatever is
already there afterwards (verified), so this is a one-time setup mistake rather
than something that recurs. That is why a guard is the whole mechanism and no
wrapper is needed -- see the README for the one-time command.

The invariant is deliberately not "the interpreter must live at path X", which
would only be true for one site. It is the portable half:

    if the venv is outside $HOME, its interpreter must be too.

A laptop checkout under ``~/code`` on a ``~/.local`` interpreter passes: both
sides move together and nothing is shared. Only the mixed case fails, and the
mixed case is already broken for any machine not sharing that ``$HOME``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _interpreters() -> list[Path]:
    """The ``.venv`` interpreter symlinks that exist."""
    bin_dir = REPO_ROOT / ".venv" / "bin"
    return [p for p in (bin_dir / "python", bin_dir / "python3") if p.is_symlink()]


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _home_or_skip() -> Path:
    if not (home := os.environ.get("HOME")):
        pytest.skip("HOME is unset, so there is no 'inside $HOME' to test against")
    home_path = Path(home).resolve()
    if _inside(REPO_ROOT, home_path):
        pytest.skip(f"{REPO_ROOT} is itself under {home_path}, so both move together")
    return home_path


def test_interpreter_is_not_inside_home_when_the_venv_is_not():
    if not (interpreters := _interpreters()):
        pytest.skip("no .venv in this checkout")
    home = _home_or_skip()

    offenders = []
    for interpreter in interpreters:
        # The raw link target matters as much as the resolved path: a dangling
        # link into $HOME cannot be resolved on the node where it is broken,
        # which is exactly the node where this has to fail.
        target = Path(os.readlink(interpreter))
        if not target.is_absolute():
            target = (interpreter.parent / target).resolve()
        if _inside(target, home):
            offenders.append(f"{interpreter} -> {target}")

    assert not offenders, (
        f"this venv only runs on the machine that built it ($HOME is {home}):\n  "
        + "\n  ".join(offenders)
        + "\nRecreate it against an interpreter that travels with the checkout --"
        " see 'On a shared filesystem' in README.md."
    )


def test_interpreter_symlink_resolves():
    """A link that resolves nowhere is the failure this guards against."""
    if not (interpreters := _interpreters()):
        pytest.skip("no .venv in this checkout")

    missing = [
        f"{p} -> {os.readlink(p)}" for p in interpreters if not p.exists()
    ]
    assert not missing, "the venv interpreter does not resolve on this machine:\n  " + (
        "\n  ".join(missing)
    )
