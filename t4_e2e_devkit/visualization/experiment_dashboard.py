"""Dependency-free multi-run analysis dashboard.

This is the local analysis layer for T4 runs: compare experiments, inspect
per-token metrics, view a metric distribution, and plot a closed-loop trace.
All data is embedded in one HTML file, so reports remain usable offline and
do not require a tracking service or a browser extension.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class ExperimentDashboard:
    """Build one interactive report from one or more result directories."""

    def __init__(
        self, result_dirs: Iterable[str | Path], *, title: str = "T4 experiment analysis"
    ) -> None:
        values = tuple(Path(directory).resolve() for directory in result_dirs)
        if not values:
            raise ValueError("at least one result directory is required")
        self.result_dirs = values
        self.title = str(title)

    def build(self, output_path: Optional[str | Path] = None) -> Path:
        output = (
            Path(output_path) if output_path is not None else self.result_dirs[0] / "analysis.html"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        experiments = [
            _load_experiment(directory, index) for index, directory in enumerate(self.result_dirs)
        ]
        payload = json.dumps(experiments, ensure_ascii=True, separators=(",", ":")).replace(
            "<", "\\u003c"
        )
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title>
<style>
:root {{ color-scheme: light; font: 14px system-ui, sans-serif; color: #17202a; background: #f5f7fa; }}
body {{ margin: 0; }} main {{ max-width: 1500px; margin: auto; padding: 24px; }}
h1 {{ margin: 0 0 4px; }} .muted {{ color: #667085; }}
section {{ background: #fff; border: 1px solid #e4e7ec; border-radius: 10px; padding: 16px; margin-top: 14px; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }}
label {{ display: grid; gap: 4px; color: #475467; font-size: 12px; }}
select, input {{ min-width: 190px; border: 1px solid #d0d5dd; border-radius: 7px; padding: 8px; background: #fff; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
.card {{ min-width: 150px; padding: 12px 14px; background: #fff; border: 1px solid #e4e7ec; border-radius: 10px; }}
.card strong {{ display: block; font-size: 20px; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }}
canvas {{ width: 100%; height: 300px; border: 1px solid #eaecf0; border-radius: 8px; background: #fcfcfd; }}
.table-wrap {{ overflow: auto; max-height: 520px; }} table {{ border-collapse: collapse; width: 100%; min-width: 720px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eaecf0; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #f9fafb; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head><body><main>
<h1>{html.escape(self.title)}</h1>
<div class="muted">Offline comparison of local T4 reports. No external service is used.</div>
<section class="toolbar">
<label>Experiment<select id="experiment"></select></label>
<label>Metric<select id="metric"></select></label>
<label>Scenario token<select id="token"></select></label>
<label>Search<input id="search" type="search" placeholder="token or value"></label>
</section>
<section><div id="cards" class="cards"></div></section>
<div class="grid"><section><h2>Metric comparison</h2><canvas id="comparison" width="900" height="320"></canvas></section>
<section><h2>Metric distribution</h2><canvas id="histogram" width="900" height="320"></canvas></section></div>
<section><h2>Closed-loop trace</h2><canvas id="trace" width="1400" height="420"></canvas></section>
<section><h2>Per-scenario results</h2><div class="table-wrap"><table><thead id="head"></thead><tbody id="body"></tbody></table></div></section>
<script>
const experiments = {payload};
const experiment = document.getElementById('experiment');
const metric = document.getElementById('metric');
const token = document.getElementById('token');
const search = document.getElementById('search');
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));
const number = value => {{ const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; }};
function selected() {{ return experiments[Number(experiment.value) || 0] || {{}}; }}
function allRows() {{ return experiments.flatMap((item, index) => (item.rows || []).map(row => ({{...row, experiment: item.name, experiment_index: index}}))); }}
function metricNames() {{ const names = new Set(); allRows().forEach(row => Object.keys(row).forEach(name => {{ if (name !== 'token' && name !== 'experiment' && name !== 'experiment_index' && number(row[name]) !== null) names.add(name); }})); return [...names].sort(); }}
function fillControls() {{
  experiment.innerHTML = experiments.map((item, index) => `<option value="${{index}}">${{escapeHtml(item.name)}}</option>`).join('');
  metric.innerHTML = metricNames().map(name => `<option>${{escapeHtml(name)}}</option>`).join('');
  refreshTokens();
}}
function refreshTokens() {{
  const current = token.value; const rows = selected().rows || [];
  const tokens = [...new Set(rows.map(row => row.token).filter(Boolean))].sort();
  token.innerHTML = '<option value="">All scenarios</option>' + tokens.map(value => `<option>${{escapeHtml(value)}}</option>`).join('');
  if (tokens.includes(current)) token.value = current;
}}
function drawAxes(ctx, width, height, maxValue, minValue = 0) {{
  ctx.clearRect(0, 0, width, height); ctx.strokeStyle = '#d0d5dd'; ctx.fillStyle = '#667085'; ctx.font = '12px system-ui';
  ctx.beginPath(); ctx.moveTo(48, 12); ctx.lineTo(48, height - 34); ctx.lineTo(width - 12, height - 34); ctx.stroke();
  ctx.fillText(maxValue.toPrecision(4), 4, 20); ctx.fillText(minValue.toPrecision(4), 4, height - 38);
}}
function drawComparison() {{
  const canvas = document.getElementById('comparison'), ctx = canvas.getContext('2d'), values = experiments.map(item => average(item.rows || [], metric.value));
  const finite = values.filter(value => value !== null); const max = Math.max(...finite, 1), min = Math.min(...finite, 0); drawAxes(ctx, canvas.width, canvas.height, max, min);
  const width = (canvas.width - 72) / Math.max(values.length, 1); values.forEach((value, index) => {{ if (value === null) return; const x = 58 + index * width + width * .15, h = (value - min) / Math.max(max - min, 1e-9) * (canvas.height - 54); ctx.fillStyle = '#175cd3'; ctx.fillRect(x, canvas.height - 34 - h, width * .7, h); ctx.fillStyle = '#344054'; ctx.save(); ctx.translate(x + width * .35, canvas.height - 16); ctx.rotate(-.35); ctx.fillText(experiments[index].name, 0, 0); ctx.restore(); ctx.fillText(value.toPrecision(4), x, canvas.height - 40 - h); }});
}}
function drawHistogram() {{
  const canvas = document.getElementById('histogram'), ctx = canvas.getContext('2d'), values = (selected().rows || []).map(row => number(row[metric.value])).filter(value => value !== null);
  const max = Math.max(...values, 1), min = Math.min(...values, 0), bins = 12, counts = Array(bins).fill(0);
  values.forEach(value => {{ const index = Math.min(bins - 1, Math.max(0, Math.floor((value - min) / Math.max(max - min, 1e-9) * bins))); counts[index]++; }}); const peak = Math.max(...counts, 1); drawAxes(ctx, canvas.width, canvas.height, peak, 0);
  const width = (canvas.width - 72) / bins; counts.forEach((count, index) => {{ const h = count / peak * (canvas.height - 54); ctx.fillStyle = '#12b76a'; ctx.fillRect(58 + index * width, canvas.height - 34 - h, width - 2, h); }});
}}
function drawTrace() {{
  const canvas = document.getElementById('trace'), ctx = canvas.getContext('2d'), wanted = token.value; const rows = (selected().trace || []).filter(row => !wanted || row.token === wanted); ctx.clearRect(0, 0, canvas.width, canvas.height); if (!rows.length) {{ ctx.fillStyle = '#667085'; ctx.fillText('No closed-loop trace for the selected scenario.', 18, 30); return; }}
  const xs = rows.map(row => number(row.x)).filter(value => value !== null), ys = rows.map(row => number(row.y)).filter(value => value !== null); const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys); const sx = value => 40 + (value - minX) / Math.max(maxX - minX, 1e-9) * (canvas.width - 70), sy = value => canvas.height - 30 - (value - minY) / Math.max(maxY - minY, 1e-9) * (canvas.height - 60); ctx.strokeStyle = '#d0d5dd'; ctx.beginPath(); ctx.moveTo(40, 10); ctx.lineTo(40, canvas.height - 30); ctx.lineTo(canvas.width - 20, canvas.height - 30); ctx.stroke(); ctx.strokeStyle = '#f04438'; ctx.lineWidth = 3; ctx.beginPath(); rows.forEach((row, index) => {{ const x = sx(number(row.x)), y = sy(number(row.y)); if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); }}); ctx.stroke(); ctx.fillStyle = '#667085'; ctx.font = '12px system-ui'; ctx.fillText('x / y world trajectory', 48, 24);
}}
function average(rows, name) {{ const values = rows.map(row => number(row[name])).filter(value => value !== null); return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null; }}
function render() {{
  refreshTokens(); const item = selected(); const query = search.value.toLowerCase(); const rows = (item.rows || []).filter(row => (!token.value || row.token === token.value) && JSON.stringify(row).toLowerCase().includes(query)).map(row => ({{...row, experiment: item.name}})); const columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
  document.getElementById('head').innerHTML = '<tr>' + columns.map(name => '<th>' + escapeHtml(name) + '</th>').join('') + '</tr>'; document.getElementById('body').innerHTML = rows.map(row => '<tr>' + columns.map(name => '<td>' + escapeHtml(row[name]) + '</td>').join('') + '</tr>').join('');
  const aggregate = item.aggregate || {{}}; document.getElementById('cards').innerHTML = Object.entries(aggregate).flatMap(([family, values]) => Object.entries(values || {{}}).filter(([name]) => name !== 'num_scenes' && name !== 'num_rollouts').map(([name, value]) => `<div class="card"><span>${{escapeHtml(family)}}/${{escapeHtml(name)}}</span><strong>${{escapeHtml(value)}}</strong></div>`)).join(''); drawComparison(); drawHistogram(); drawTrace();
}}
experiment.addEventListener('change', () => {{ refreshTokens(); render(); }}); metric.addEventListener('change', render); token.addEventListener('change', render); search.addEventListener('input', render); fillControls(); render();
</script></main></body></html>"""
        output.write_text(document, encoding="utf-8")
        return output


def write_experiment_dashboard(
    result_dirs: Iterable[str | Path],
    output_path: Optional[str | Path] = None,
    *,
    title: str = "T4 experiment analysis",
) -> Path:
    return ExperimentDashboard(result_dirs, title=title).build(output_path)


def _load_experiment(directory: Path, index: int) -> dict[str, Any]:
    run = _read_json(directory / "run.json")
    aggregate = _read_json(directory / "aggregate.json")
    name = str(run.get("experiment_name") or run.get("agent") or directory.name or f"run-{index}")
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")):
        if path.name.endswith("_ticks.csv") or path.name == "failures.csv":
            continue
        rows.extend({"source": path.stem, **row} for row in _read_csv(path))
    trace = _read_csv(directory / "closed_loop_ticks.csv")
    return {"name": name, "run": run, "aggregate": aggregate, "rows": rows, "trace": trace}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


__all__ = ["ExperimentDashboard", "write_experiment_dashboard"]
