"""The helpers every entry point shares.

These were a private copy per script -- four of the checkpoint loader alone --
and the copies had already drifted: one run reader reported "not finished"
separately, another folded it into one message. Tested here once so the next
entry point inherits the behaviour instead of a fifth copy.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from t4_e2e_devkit.script.utils import (
    file_digest,
    load_agent_checkpoint,
    manifest_tokens,
    read_run,
    value_fingerprint,
    write_json,
)

RUN_FORMAT = "t4.evaluation.run"
RUN_VERSION = 1


class _Agent(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(width, 2)


def _worker_manifest(run_id: str, rank: int, task_ids: tuple[str, ...]):
    """A manifest the reader accepts: it requires one result per declared task."""
    from t4_e2e_devkit.evaluation.distributed import WorkerManifest
    from t4_e2e_devkit.evaluation.worker_pool import WorkerResult

    return WorkerManifest(
        run_id=run_id,
        rank=rank,
        world_size=rank + 1,
        task_ids=task_ids,
        results=tuple(WorkerResult(task_id=task, rank=rank) for task in task_ids),
    )


def _write_run(directory, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "status": "completed",
        "run_id": "run-1",
        "rank": 0,
    }
    record.update(overrides)
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return record


class TestWriteJson:
    def test_writes_sorted_json_and_creates_parents(self, tmp_path):
        path = write_json(tmp_path / "nested" / "out.json", {"b": 1, "a": 2})
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert text.index('"a"') < text.index('"b"')
        assert json.loads(text) == {"b": 1, "a": 2}

    def test_no_temporary_file_survives(self, tmp_path):
        write_json(tmp_path / "out.json", {"a": 1})
        # The write goes through a temp file and os.replace; a leftover dotfile
        # would mean an interrupted write left the directory dirty.
        assert [p.name for p in tmp_path.iterdir()] == ["out.json"]

    def test_replaces_an_existing_file_in_one_step(self, tmp_path):
        path = tmp_path / "out.json"
        write_json(path, {"a": 1})
        write_json(path, {"a": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}


class TestDigests:
    def test_value_fingerprint_ignores_key_order(self):
        assert value_fingerprint({"a": 1, "b": 2}) == value_fingerprint({"b": 2, "a": 1})

    def test_value_fingerprint_separates_different_values(self):
        assert value_fingerprint({"a": 1}) != value_fingerprint({"a": 2})

    def test_file_digest_of_nothing_is_none(self):
        assert file_digest(None) is None

    def test_file_digest_matches_the_manifest_layer(self, tmp_path):
        from t4_e2e_devkit.evaluation.prediction_manifest import file_sha256

        path = tmp_path / "blob.bin"
        path.write_bytes(b"t4")
        # One digest for one file: a run record and a prediction manifest must
        # agree, or provenance cannot be checked across them.
        assert file_digest(path) == file_sha256(path)


class TestReadRun:
    def test_reads_a_completed_run(self, tmp_path):
        _write_run(tmp_path / "rank-0")
        record = read_run(
            tmp_path / "rank-0",
            kind="evaluation",
            run_format=RUN_FORMAT,
            run_version=RUN_VERSION,
        )
        assert record["run_id"] == "run-1"

    def test_a_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(ValueError, match="cannot read"):
            read_run(tmp_path, kind="evaluation", run_format=RUN_FORMAT, run_version=RUN_VERSION)

    def test_another_kind_of_run_is_refused(self, tmp_path):
        _write_run(tmp_path / "rank-0", format="t4.submission-score.run")
        with pytest.raises(ValueError, match="not a evaluation run directory"):
            read_run(
                tmp_path / "rank-0",
                kind="evaluation",
                run_format=RUN_FORMAT,
                run_version=RUN_VERSION,
            )

    def test_an_unfinished_run_says_so_separately(self, tmp_path):
        # "wrong directory" and "still running" send a caller to different
        # places, so they stay two messages.
        _write_run(tmp_path / "rank-0", status="running")
        with pytest.raises(ValueError, match="is not finished"):
            read_run(
                tmp_path / "rank-0",
                kind="evaluation",
                run_format=RUN_FORMAT,
                run_version=RUN_VERSION,
            )


class TestManifestTokens:
    def test_no_manifest_declared_is_none(self, tmp_path):
        record = _write_run(tmp_path / "rank-0")
        assert manifest_tokens(tmp_path / "rank-0", record, kind="evaluation") is None

    def test_a_declared_manifest_must_exist(self, tmp_path):
        record = _write_run(tmp_path / "rank-0", manifest="worker.json")
        with pytest.raises(ValueError, match="missing its worker manifest"):
            manifest_tokens(tmp_path / "rank-0", record, kind="evaluation")

    def test_task_ids_come_back(self, tmp_path):
        directory = tmp_path / "rank-0"
        record = _write_run(directory, manifest="worker.json")
        _worker_manifest("run-1", 0, ("a", "b")).write(directory / "worker.json")
        assert manifest_tokens(directory, record, kind="evaluation") == {"a", "b"}

    def test_a_manifest_from_another_run_is_refused(self, tmp_path):
        directory = tmp_path / "rank-0"
        record = _write_run(directory, manifest="worker.json")
        _worker_manifest("run-2", 0, ("a",)).write(directory / "worker.json")
        with pytest.raises(ValueError, match="different run"):
            manifest_tokens(directory, record, kind="evaluation")

    def test_a_manifest_from_another_rank_is_refused(self, tmp_path):
        directory = tmp_path / "rank-0"
        record = _write_run(directory, manifest="worker.json", rank=0)
        _worker_manifest("run-1", 3, ("a",)).write(directory / "worker.json")
        with pytest.raises(ValueError, match="rank does not match"):
            manifest_tokens(directory, record, kind="evaluation")


class TestLoadAgentCheckpoint:
    def test_a_lightning_prefixed_checkpoint_loads(self, tmp_path):
        agent, source = _Agent(), _Agent()
        path = tmp_path / "last.ckpt"
        torch.save({"state_dict": {f"agent.{k}": v for k, v in source.state_dict().items()}}, path)
        load_agent_checkpoint(agent, path)
        assert torch.equal(agent.head.weight, source.head.weight)

    def test_a_bare_state_dict_loads(self, tmp_path):
        agent, source = _Agent(), _Agent()
        path = tmp_path / "bare.ckpt"
        torch.save(source.state_dict(), path)
        load_agent_checkpoint(agent, path)
        assert torch.equal(agent.head.bias, source.head.bias)

    def test_a_checkpoint_for_another_model_is_refused(self, tmp_path):
        """The failure the four private copies all had.

        ``strict=False`` is needed for optimizer state and loss buffers, and it
        also accepts a checkpoint that fits nothing: the run then evaluates
        randomly initialized weights, which reads as a bad model rather than as
        a wrong path.
        """
        path = tmp_path / "other.ckpt"
        torch.save({"state_dict": {"encoder.weight": torch.zeros(2, 2)}}, path)
        with pytest.raises(ValueError, match="checkpoint for a different model"):
            load_agent_checkpoint(_Agent(), path)

    def test_a_partial_checkpoint_loads_and_warns(self, tmp_path, caplog):
        agent, source = _Agent(), _Agent()
        state = {f"agent.{k}": v for k, v in source.state_dict().items()}
        state.pop("agent.head.bias")
        state["agent.optimizer.step"] = torch.zeros(1)
        path = tmp_path / "partial.ckpt"
        torch.save({"state_dict": state}, path)
        with caplog.at_level("WARNING"):
            load_agent_checkpoint(agent, path)
        assert torch.equal(agent.head.weight, source.head.weight)
        assert "1 missing" in caplog.text and "1 unexpected" in caplog.text
