"""Merge and validate rank-local trajectory submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from t4_e2e_devkit.evaluation.submission import SubmissionPackage


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="t4e2e merge-submission")
    parser.add_argument("--input-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = SubmissionPackage.merge(args.input_dir, args.output_dir)
    package = SubmissionPackage.read(output)
    summary = {
        "status": "completed",
        "num_predictions": len(package.entries),
        "output_dir": str(Path(output).resolve()),
        "validation": package.validate().as_dict(),
    }
    (Path(output) / "status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
