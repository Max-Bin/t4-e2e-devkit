"""Self-contained local dashboard for ignored evaluation artifacts."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from t4_e2e_devkit.common.artifact_io import read_mapping


class ResultsDashboard:
    """Build a dependency-free HTML view of aggregate and per-row results."""

    def __init__(self, results_dir: str | Path, *, title: str = "T4 evaluation results") -> None:
        self.results_dir = Path(results_dir)
        self.title = str(title)

    def build(self, output_path: Optional[str | Path] = None) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output = Path(output_path) if output_path is not None else self.results_dir / "index.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        aggregate = read_mapping(self.results_dir / "aggregate.json")
        run = read_mapping(self.results_dir / "run.json")
        tables = _read_tables(self.results_dir)
        cards = _cards(aggregate)
        table_data = json.dumps(tables, ensure_ascii=True).replace("<", "\\u003c")
        file_rows = _file_rows(self.results_dir, output)
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title>
<style>
:root {{ color-scheme: light; font: 14px system-ui, sans-serif; color: #17202a; background: #f5f7fa; }}
body {{ margin: 0; }} main {{ max-width: 1280px; margin: auto; padding: 24px; }}
h1 {{ margin: 0 0 4px; }} .muted {{ color: #667085; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }}
.card {{ min-width: 150px; padding: 12px 14px; background: white; border: 1px solid #e4e7ec; border-radius: 10px; }}
.card strong {{ display: block; font-size: 20px; margin-top: 4px; }}
.toolbar {{ display: flex; gap: 10px; margin: 16px 0; }}
input, select {{ border: 1px solid #d0d5dd; border-radius: 7px; padding: 8px; background: white; }}
.table-wrap {{ overflow: auto; background: white; border: 1px solid #e4e7ec; border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; min-width: 720px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eaecf0; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #f9fafb; }}
a {{ color: #175cd3; }} code {{ font-size: 12px; }}
</style></head><body><main>
<h1>{html.escape(self.title)}</h1>
<div class="muted">{html.escape(_run_summary(run))}</div>
<section class="cards">{cards}</section>
<div class="toolbar"><input id="filter" type="search" placeholder="Filter token or value">
<select id="family"><option value="">All families</option></select></div>
<div class="table-wrap"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<h2>Files</h2><ul>{file_rows or '<li class="muted">No result files found.</li>'}</ul>
</main><script>
const tables = {table_data};
const family = document.getElementById('family');
const filter = document.getElementById('filter');
const head = document.getElementById('head');
const body = document.getElementById('body');
Object.keys(tables).sort().forEach(name => {{ const option = document.createElement('option'); option.value = name; option.textContent = name; family.appendChild(option); }});
function render() {{
  const selected = family.value; const query = filter.value.toLowerCase();
  const groups = selected ? [selected] : Object.keys(tables).sort();
  const rows = groups.flatMap(name => (tables[name] || []).map(row => ({{...row, family: name}})));
  const columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const escapeHtml = value => String(value ?? '').replace(/[&<>\"']/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[char]));
  head.innerHTML = '<tr>' + columns.map(name => '<th>' + escapeHtml(name) + '</th>').join('') + '</tr>';
  body.innerHTML = rows.filter(row => JSON.stringify(row).toLowerCase().includes(query)).map(row => '<tr>' + columns.map(name => '<td>' + escapeHtml(row[name]) + '</td>').join('') + '</tr>').join('');
}}
family.addEventListener('change', render); filter.addEventListener('input', render); render();
</script></body></html>"""
        output.write_text(document, encoding="utf-8")
        return output

    def _rows(self) -> list[dict[str, str]]:
        """List files for callers that used the original dashboard helper."""

        rows = []
        for path in sorted(self.results_dir.rglob("*")):
            if path.is_file() and path.name != "index.html":
                kind, summary = _summarize(path)
                rows.append(
                    {
                        "path": str(path.relative_to(self.results_dir)),
                        "kind": kind,
                        "summary": summary,
                    }
                )
        return rows


def write_results_dashboard(
    results_dir: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    title: str = "T4 evaluation results",
) -> Path:
    return ResultsDashboard(results_dir, title=title).build(output_path)


def _read_tables(directory: Path) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                tables[path.stem] = list(csv.DictReader(stream))
        except OSError:
            tables[path.stem] = []
    return tables


def _cards(aggregate: Mapping[str, Any]) -> str:
    cards: list[str] = []
    for family, values in aggregate.items():
        if not isinstance(values, Mapping) or family == "run":
            continue
        for name, value in values.items():
            if name == "num_scenes":
                continue
            cards.append(
                f'<div class="card"><span>{html.escape(str(family))}/{html.escape(str(name))}</span>'
                f"<strong>{html.escape(_format(value))}</strong></div>"
            )
    return "".join(cards)


def _run_summary(run: Mapping[str, Any]) -> str:
    status = str(run.get("status", "unknown"))
    completed = run.get("num_completed", "?")
    failed = run.get("num_failed", "?")
    world_size = run.get("world_size")
    suffix = f" · {world_size} ranks" if world_size and int(world_size) > 1 else ""
    return f"{status} · {completed} completed · {failed} failed{suffix}"


def _file_rows(directory: Path, output: Path) -> str:
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.resolve() == output.resolve():
            continue
        href = Path(os.path.relpath(path, output.parent))
        label = path.relative_to(directory)
        rows.append(
            f'<li><a href="{html.escape(href.as_posix(), quote=True)}">'
            f"<code>{html.escape(str(label))}</code></a></li>"
        )
    return "".join(rows)


def _format(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _summarize(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".json":
        value = read_mapping(path)
        return "json", ", ".join(str(key) for key in list(value)[:4]) or "empty"
    if path.suffix.lower() == ".csv":
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return "csv", "unreadable"
        return "csv", f"{max(0, len(lines) - 1)} rows"
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return path.suffix.lstrip(".") or "file", f"{size} bytes"


__all__ = ["ResultsDashboard", "write_results_dashboard"]
