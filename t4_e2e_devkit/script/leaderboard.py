"""Build a local leaderboard from ignored result directories."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from t4_e2e_devkit.evaluation.leaderboard import build_leaderboard


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e leaderboard")
    parser.add_argument("result_dir", nargs="+")
    parser.add_argument("--family", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--higher-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = build_leaderboard(
        args.result_dir,
        family=args.family,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
    )
    report.write(args.output_dir)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
