"""Local leaderboard tables built from validated result directories."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class LeaderboardRow:
    """One completed run ranked by one independent metric."""

    name: str
    directory: str
    family: str
    metric: str
    value: float
    rank: int = 0
    status: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "directory": self.directory,
            "family": self.family,
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
        }


@dataclass(frozen=True)
class LeaderboardReport:
    """Sorted local comparison with no cross-family composite score."""

    family: str
    metric: str
    higher_is_better: bool
    rows: tuple[LeaderboardRow, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "t4.leaderboard",
            "version": 1,
            "family": self.family,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "rows": [row.as_dict() for row in self.rows],
        }

    def write(self, output_dir: str | Path) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "leaderboard.json").write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (output / "leaderboard.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["rank", "name", "directory", "family", "metric", "value", "status"],
            )
            writer.writeheader()
            writer.writerows(row.as_dict() for row in self.rows)
        return output


def build_leaderboard(
    result_dirs: Iterable[str | Path],
    *,
    family: str,
    metric: str,
    higher_is_better: bool = True,
    names: Optional[Mapping[str | Path, str]] = None,
) -> LeaderboardReport:
    """Read completed aggregate files and rank the requested metric."""

    directories = tuple(Path(directory).resolve() for directory in result_dirs)
    if not directories:
        raise ValueError("at least one result directory is required")
    rows: list[LeaderboardRow] = []
    for directory in directories:
        run = _read_mapping(directory / "run.json")
        if run.get("status") != "completed":
            raise ValueError(f"leaderboard input is not completed: {directory}")
        aggregate = _read_mapping(directory / "aggregate.json")
        family_values = aggregate.get(str(family))
        if not isinstance(family_values, Mapping) or metric not in family_values:
            raise ValueError(f"{directory} has no metric {family}/{metric}")
        value = float(family_values[metric])
        if not math.isfinite(value):
            raise ValueError(f"{directory} has a non-finite metric {family}/{metric}")
        name = str(
            (names or {}).get(str(directory), (names or {}).get(directory, ""))
            or run.get("experiment_name")
            or run.get("agent")
            or directory.name
        )
        rows.append(LeaderboardRow(name, str(directory), str(family), str(metric), value))
    rows.sort(key=lambda row: (-row.value if higher_is_better else row.value, row.name))
    ranked = tuple(
        LeaderboardRow(
            name=row.name,
            directory=row.directory,
            family=row.family,
            metric=row.metric,
            value=row.value,
            rank=index,
            status=row.status,
        )
        for index, row in enumerate(rows, start=1)
    )
    return LeaderboardReport(str(family), str(metric), bool(higher_is_better), ranked)


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read leaderboard input: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"leaderboard input must be a mapping: {path}")
    return value


__all__ = ["LeaderboardReport", "LeaderboardRow", "build_leaderboard"]
