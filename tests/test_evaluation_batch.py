from __future__ import annotations

import csv
import json

import pytest
import torch

from t4_e2e_devkit.common.constants import PDM_COMPONENT_ORDER
from t4_e2e_devkit.common.dataclasses import aggregate_pdm_score
from t4_e2e_devkit.evaluation.batch import (
    RUN_FORMAT,
    RUN_VERSION,
    aggregate_records,
    config_fingerprint,
    record_path,
    write_family_csv,
    write_json,
)
from t4_e2e_devkit.evaluation.distributed import WorkerManifest
from t4_e2e_devkit.evaluation.worker_pool import WorkerResult
from t4_e2e_devkit.script.evaluate import _resolve_backend
from t4_e2e_devkit.script.merge_evaluation import merge_evaluation_reports
from t4_e2e_devkit.visualization.dashboard import write_results_dashboard


def _pdm_values(*, score: float | None = None) -> dict[str, float]:
    values = dict.fromkeys(PDM_COMPONENT_ORDER, 0.8)
    if score is not None:
        values["score"] = score
    return values


def test_auto_backend_uses_cuda_when_available_and_cpu_explicitly():
    expected = "gpu" if torch.cuda.is_available() else "cpu"
    assert _resolve_backend("auto") == expected
    assert _resolve_backend("cpu") == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="needs CUDA"):
            _resolve_backend("gpu")


def test_batch_aggregation_recomputes_missing_score_and_removes_stale_csv(tmp_path):
    record = {"token": "scene@10", "families": {"pdm": _pdm_values()}}

    report = aggregate_records([record])

    assert report["pdm"]["score"] == pytest.approx(aggregate_pdm_score([0.8] * 6))

    (tmp_path / "open_loop.csv").write_text("stale\n", encoding="utf-8")
    write_family_csv(tmp_path, [record])
    assert (tmp_path / "pdm.csv").is_file()
    assert not (tmp_path / "open_loop.csv").exists()
    with (tmp_path / "pdm.csv").open(newline="", encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["token"] == "scene@10"


def _write_rank(tmp_path, rank: int, token: str) -> None:
    config = {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "agent": "agent",
        "families": ["pdm"],
        "backend": "cpu",
        "device": None,
        "history_frames": 1,
        "future_frames": 1,
        "frame_interval": 1,
        "agent_params": {},
        "checkpoint_digest": None,
        "data_digest": "data",
        "reader_config_digest": "reader",
        "selected_rows": 2,
        "rank": rank,
        "world_size": 2,
        "workers": 1,
        "worker_backend": "serial",
        "max_retries": 0,
        "run_id": "run",
    }
    fingerprint = config_fingerprint(config)
    rank_dir = tmp_path / f"rank-{rank}"
    record = {
        "format": "t4.evaluation.record",
        "version": 1,
        "status": "ok",
        "token": token,
        "families": {"pdm": _pdm_values()},
        "attempts": 1,
        "config_fingerprint": fingerprint,
    }
    write_json(record_path(rank_dir / "records", token), record)
    WorkerManifest(
        run_id="run",
        rank=rank,
        world_size=2,
        task_ids=(token,),
        results=(WorkerResult(task_id=token, value={"status": "ok"}, rank=rank),),
    ).write(rank_dir / f"worker-manifest-rank-{rank}.json")
    write_json(
        rank_dir / "run.json",
        {
            **config,
            "status": "completed",
            "config_fingerprint": fingerprint,
            "rank_rows": 1,
            "num_completed": 1,
            "num_failed": 0,
            "manifest": f"worker-manifest-rank-{rank}.json",
        },
    )


def test_rank_merge_validates_manifests_and_keeps_configuration_fingerprint(tmp_path):
    _write_rank(tmp_path, 0, "scene@10")
    _write_rank(tmp_path, 1, "scene@20")

    report = merge_evaluation_reports(
        [tmp_path / "rank-0", tmp_path / "rank-1"],
        tmp_path / "merged",
    )

    assert report["pdm"]["num_scenes"] == pytest.approx(2.0)
    merged_run = json.loads((tmp_path / "merged" / "run.json").read_text(encoding="utf-8"))
    merged_record = json.loads(
        record_path(tmp_path / "merged" / "records", "scene@10").read_text(encoding="utf-8")
    )
    assert merged_record["config_fingerprint"] == merged_run["config_fingerprint"]
    assert "tmp" not in json.dumps(merged_run)


def test_dashboard_links_files_when_written_outside_results_dir(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "aggregate.json").write_text(
        json.dumps({"pdm": {"score": 0.5}}), encoding="utf-8"
    )
    (results / "run.json").write_text(
        json.dumps({"status": "completed", "num_completed": 1, "num_failed": 0}),
        encoding="utf-8",
    )
    (results / "pdm.csv").write_text("token,score\n<scene>,0.5\n", encoding="utf-8")

    output = tmp_path / "html" / "dashboard.html"
    write_results_dashboard(results, output)
    document = output.read_text(encoding="utf-8")

    assert "../results/pdm.csv" in document
    assert "const escapeHtml" in document
    assert str(tmp_path) not in document
