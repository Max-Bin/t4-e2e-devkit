#!/usr/bin/env python3
"""Type-check the first-party tree against a recorded baseline.

The package has type debt: 239 mypy errors across 60 of its 191 first-party
files, most of them numpy's ``ArrayLike`` unions rather than defects. Demanding
zero before the checker may run at all would mean either a flag day or no
checker, so this records what is broken today and fails only on movement in the
wrong direction:

* a file that had no errors and now has some -- the case that matters, since it
  means new code, or a change to old code, introduced a type error;
* a file whose error count grew.

A file that got better is reported too, with the command to shrink the baseline,
because a baseline that is never tightened stops meaning anything.

The exercise is not academic. Running this for the first time found that
``T4MapAPI.match_local_geometries_detailed`` read ``.geometry`` off candidates
that can be lanelets, which carry ``polygon`` -- an AttributeError on every
match against lanelets, reproduced on a real map.

Usage::

    uv run python tools/typecheck.py            # check against the baseline
    uv run python tools/typecheck.py --update    # record the current state
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tools" / "mypy_baseline.json"
TARGET = "t4_e2e_devkit"
ERROR_LINE = re.compile(r"^(?P<path>[^:]+):\d+: error: .*\[(?P<code>[a-z-]+)\]\s*$")


def run_mypy() -> tuple[dict[str, int], dict[str, int]]:
    """
    :return: ``(errors per file, errors per code)`` for the first-party tree.
    """
    process = subprocess.run(
        [sys.executable, "-m", "mypy", TARGET],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if "error:" not in process.stdout and process.returncode not in (0, 1):
        raise SystemExit(
            f"mypy did not run (exit {process.returncode}):\n{process.stderr or process.stdout}"
        )
    per_file: dict[str, int] = collections.Counter()
    per_code: dict[str, int] = collections.Counter()
    for line in process.stdout.splitlines():
        match = ERROR_LINE.match(line.strip())
        if match:
            per_file[match.group("path")] += 1
            per_code[match.group("code")] += 1
    return dict(per_file), dict(per_code)


def load_baseline() -> dict[str, int]:
    """:return: the recorded errors per file, empty when there is no baseline."""
    if not BASELINE_PATH.is_file():
        return {}
    return dict(json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["files"])


def write_baseline(per_file: dict[str, int], per_code: dict[str, int]) -> None:
    """Record the current state as the baseline."""
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "note": "Recorded mypy debt. Run tools/typecheck.py --update to refresh.",
                "total": sum(per_file.values()),
                "files": dict(sorted(per_file.items())),
                "codes": dict(sorted(per_code.items(), key=lambda item: (-item[1], item[0]))),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """
    :param argv: argument vector.
    :return: process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="record the current state")
    args = parser.parse_args(argv)

    per_file, per_code = run_mypy()
    total = sum(per_file.values())

    if args.update:
        write_baseline(per_file, per_code)
        print(f"recorded {total} errors across {len(per_file)} files in {BASELINE_PATH.name}")
        return 0

    baseline = load_baseline()
    if not baseline:
        print(f"no baseline at {BASELINE_PATH}; run with --update first")
        return 1

    regressions = sorted(
        (path, count, baseline.get(path, 0))
        for path, count in per_file.items()
        if count > baseline.get(path, 0)
    )
    improvements = sorted(
        (path, per_file.get(path, 0), count)
        for path, count in baseline.items()
        if per_file.get(path, 0) < count
    )

    for path, now, before in improvements:
        print(f"improved: {path}  {before} -> {now}")
    if improvements:
        print("run tools/typecheck.py --update to record the improvement\n")

    if regressions:
        print(f"{len(regressions)} file(s) type-check worse than the baseline:\n")
        for path, now, before in regressions:
            print(f"  {path}  {before} -> {now}")
        print(f"\ntotal: {total} (baseline {sum(baseline.values())})")
        return 1

    print(f"no new type errors; {total} known (baseline {sum(baseline.values())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
