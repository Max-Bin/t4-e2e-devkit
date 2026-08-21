"""The type-check gate, as a slow test.

Marked ``slow`` because it shells out to mypy over 191 files and costs tens of
seconds; it is opt-in with ``-m slow`` and is meant for CI. What it protects is
narrow and worth protecting: a file with no type errors must not acquire one.
The 239 known errors stay known -- see tools/mypy_baseline.json -- and the tool
reports any file that got better so the baseline can be tightened.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


def test_no_file_type_checks_worse_than_the_baseline():
    if shutil.which(sys.executable) is None:  # pragma: no cover - defensive
        pytest.skip("no interpreter to run mypy with")
    try:
        import mypy  # noqa: F401
    except ImportError:
        pytest.skip("mypy is not installed; it is in the dev dependency group")

    process = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "typecheck.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
