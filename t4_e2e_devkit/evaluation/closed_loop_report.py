"""Local, dependency-free reports for closed-loop evaluation artifacts."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from t4_e2e_devkit.evaluation.closed_loop import ClosedLoopMetrics

TRACE_COLUMNS = (
    "token",
    "step",
    "source_frame",
    "time_s",
    "x",
    "y",
    "heading",
    "speed_mps",
    "acceleration_mps2",
    "yaw_rate_radps",
    "steering_rad",
    "step_distance_m",
    "path_length_m",
    "goal_distance_m",
    "collision",
    "plan_available",
    "plan_num_poses",
    "plan_interval_s",
    "agent_count",
    "min_agent_clearance_m",
    "ttc_s",
    "ttc_violation",
    "drivable_violation",
    "road_border_violation",
    "road_border_distance_m",
)


def write_closed_loop_csv(path: str | Path, results: Iterable[ClosedLoopMetrics]) -> Path:
    """Write one aggregate row per rollout."""

    destination = Path(path)
    rows = list(results)
    names = sorted({name for result in rows for name in result.values})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "termination_reason", *names])
        for result in rows:
            values = result.values
            writer.writerow(
                [
                    result.token or "",
                    result.termination_reason or "",
                    *[f"{values[name]:.6f}" if name in values else "" for name in names],
                ]
            )
    return destination


def write_closed_loop_ticks(path: str | Path, results: Iterable[ClosedLoopMetrics]) -> Path:
    """Write the per-step trace for all supplied rollouts."""

    destination = Path(path)
    rows: list[dict[str, object]] = []
    for result in results:
        if result.trace is not None:
            rows.extend(result.trace.rows(result.token))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in TRACE_COLUMNS})
    return destination


def write_static_html_report(
    report_dir: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Render a self-contained HTML report from local CSV/JSON files.

    The report intentionally uses only the Python standard library.  CSV files
    remain the complete machine-readable output; the HTML rollout table is
    capped to keep a large evaluation directory easy to open in a browser.
    """

    directory = Path(report_dir)
    destination = Path(output_path) if output_path is not None else directory / "report.html"
    aggregate = _read_json(directory / "aggregate.json", default={})
    run = _read_json(directory / "run.json", default={})
    failures = _read_csv(directory / "failures.csv")
    rollouts = _read_csv(directory / "closed_loop.csv")
    shown_rollouts = rollouts[:500]

    sections = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Closed-loop report</title>{_STYLE}</head><body>",
        "<main>",
        "<h1>Closed-loop evaluation</h1>",
        _run_summary(run),
        _aggregate_section(aggregate),
        _table_section(
            "Rollouts",
            shown_rollouts,
            empty="No completed rollouts.",
            note=(
                f"Showing {len(shown_rollouts)} of {len(rollouts)} rows; "
                "the complete table is in closed_loop.csv."
                if len(shown_rollouts) < len(rollouts)
                else None
            ),
        ),
        _table_section("Failures", failures, empty="No failed rollouts."),
        '<p class="muted">Generated locally from the evaluation directory. No external service is required.</p>',
        "</main></body></html>",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return destination


def _read_json(path: Path, *, default: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, Mapping) else default


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _run_summary(run: Mapping[str, Any]) -> str:
    keys = (
        "status",
        "agent",
        "num_completed",
        "num_failed",
        "num_rows",
        "num_attempts",
        "num_shards",
        "shard_index",
        "merged",
    )
    cells = []
    for key in keys:
        if key in run:
            cells.append(
                f'<div class="card"><div class="label">{html.escape(key)}</div>'
                f'<div class="value">{html.escape(_format_value(run[key]))}</div></div>'
            )
    return '<section class="cards">' + "".join(cells) + "</section>"


def _aggregate_section(aggregate: Mapping[str, Any]) -> str:
    sections = []
    for family, values in aggregate.items():
        if not isinstance(values, Mapping):
            continue
        rows = [{"metric": str(name), "value": _format_value(value)} for name, value in values.items()]
        sections.append(
            f"<h3>{html.escape(str(family))}</h3>"
            + _table(rows, empty="No values.")
        )
    return "<section><h2>Aggregate metrics</h2>" + ("".join(sections) or "<p>No aggregate.json.</p>") + "</section>"


def _table_section(
    title: str,
    rows: list[dict[str, str]],
    *,
    empty: str,
    note: Optional[str] = None,
) -> str:
    content = f"<h2>{html.escape(title)}</h2>"
    if note:
        content += f'<p class="muted">{html.escape(note)}</p>'
    content += _table(rows, empty=empty)
    return f"<section>{content}</section>"


def _table(rows: list[dict[str, Any]], *, empty: str) -> str:
    if not rows:
        return f'<p class="muted">{html.escape(empty)}</p>'
    columns = list(rows[0])
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f'<td>{html.escape(_format_value(row.get(column, "")))}</td>'
                for column in columns
            )
            + "</tr>"
        )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


_STYLE = """
<style>
:root { color-scheme: light; font-family: system-ui, sans-serif; }
body { margin: 0; background: #f5f7fa; color: #172033; }
main { max-width: 1400px; margin: 0 auto; padding: 28px; }
h1 { margin-top: 0; } h2 { margin: 28px 0 12px; } h3 { margin-bottom: 6px; }
section { background: white; border: 1px solid #dfe4ec; border-radius: 10px; padding: 16px; margin: 16px 0; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; background: transparent; border: 0; padding: 0; }
.card { min-width: 135px; background: white; border: 1px solid #dfe4ec; border-radius: 8px; padding: 12px; }
.label, .muted { color: #647089; font-size: .88rem; } .value { font-size: 1.15rem; font-weight: 650; margin-top: 5px; }
.table-wrap { overflow-x: auto; } table { border-collapse: collapse; width: 100%; font-size: .88rem; }
th, td { text-align: left; border-bottom: 1px solid #edf0f4; padding: 7px 9px; white-space: nowrap; }
th { background: #f7f9fb; position: sticky; top: 0; } td { max-width: 420px; overflow: hidden; text-overflow: ellipsis; }
</style>
"""


__all__ = [
    "TRACE_COLUMNS",
    "write_closed_loop_csv",
    "write_closed_loop_ticks",
    "write_static_html_report",
]
