"""The manifest-level rank executor: the public path nothing in-tree exercised.

``DistributedExecutor`` is exported from the package root and had no caller and
no test.  Its logic is not trivial -- it validates that a resume manifest belongs
to this rank of this run, skips the tasks that already succeeded, restores their
results in task order, and writes the manifest a merger later reads -- so the
absence of coverage was the problem, not the absence of callers.

It is deliberately not the resume the entry points use.  ``t4e2e evaluate``
resumes from per-scenario record files keyed by a config fingerprint, which
survives a lost manifest and works per window; this resumes from one manifest per
rank, which is what a caller wants when the work is not a scenario sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from t4_e2e_devkit.evaluation.distributed import (
    DistributedExecutor,
    DistributedRunConfig,
    WorkerManifest,
)
from t4_e2e_devkit.evaluation.worker_pool import WorkerResult, WorkerTask


def _config(tmp_path: Path, **overrides) -> DistributedRunConfig:
    values = {"run_id": "run-1", "rank": 0, "world_size": 1, "workers": 1, "backend": "serial"}
    values.update(overrides)
    return DistributedRunConfig(**values)


def _touch(marker: str) -> str:
    """A task whose side effect is visible to the test, so a skip is provable."""
    Path(marker).write_text("ran", encoding="utf-8")
    return marker


def _tasks(tmp_path: Path, *names: str) -> list[WorkerTask]:
    return [
        WorkerTask(task_id=name, function=_touch, args=(str(tmp_path / f"{name}.done"),))
        for name in names
    ]


class TestRun:
    def test_runs_every_task_and_writes_the_manifest(self, tmp_path):
        manifest_path = tmp_path / "worker.json"
        results = DistributedExecutor(_config(tmp_path)).run(
            _tasks(tmp_path, "a", "b"), manifest_path=manifest_path
        )
        assert [result.task_id for result in results] == ["a", "b"]
        assert all(result.succeeded for result in results)

        manifest = WorkerManifest.read(manifest_path)
        assert manifest.run_id == "run-1" and manifest.rank == 0
        assert manifest.task_ids == ("a", "b")

    def test_without_a_manifest_path_nothing_is_written(self, tmp_path):
        results = DistributedExecutor(_config(tmp_path)).run(_tasks(tmp_path, "a"))
        assert [result.task_id for result in results] == ["a"]
        assert not list(tmp_path.glob("*.json"))

    def test_a_rank_runs_only_its_own_partition(self, tmp_path):
        first = DistributedExecutor(_config(tmp_path, rank=0, world_size=2)).run(
            _tasks(tmp_path, "a", "b")
        )
        second = DistributedExecutor(_config(tmp_path, rank=1, world_size=2)).run(
            _tasks(tmp_path, "a", "b")
        )
        assert {result.task_id for result in first} | {result.task_id for result in second} == {
            "a",
            "b",
        }
        assert not ({r.task_id for r in first} & {r.task_id for r in second})


class TestResume:
    def test_a_completed_task_is_not_run_again(self, tmp_path):
        manifest_path = tmp_path / "worker.json"
        executor = DistributedExecutor(_config(tmp_path))
        executor.run(_tasks(tmp_path, "a", "b"), manifest_path=manifest_path)

        # Remove the side effect of "a"; if the resume re-ran it, the file
        # would come back.
        (tmp_path / "a.done").unlink()
        results = executor.run(_tasks(tmp_path, "a", "b"), manifest_path=manifest_path)

        assert not (tmp_path / "a.done").exists(), "a completed task was run again"
        assert [result.task_id for result in results] == ["a", "b"], "resumed results lost order"

    def test_resume_can_be_turned_off(self, tmp_path):
        manifest_path = tmp_path / "worker.json"
        executor = DistributedExecutor(_config(tmp_path))
        executor.run(_tasks(tmp_path, "a"), manifest_path=manifest_path)
        (tmp_path / "a.done").unlink()

        executor.run(_tasks(tmp_path, "a"), manifest_path=manifest_path, resume=False)
        assert (tmp_path / "a.done").exists()

    def test_a_manifest_from_another_rank_is_refused(self, tmp_path):
        manifest_path = tmp_path / "worker.json"
        WorkerManifest(
            run_id="run-1",
            rank=1,
            world_size=2,
            task_ids=("a",),
            results=(WorkerResult(task_id="a", rank=1),),
        ).write(manifest_path)
        with pytest.raises(ValueError, match="does not belong to this distributed rank"):
            DistributedExecutor(_config(tmp_path)).run(
                _tasks(tmp_path, "a"), manifest_path=manifest_path
            )

    def test_a_manifest_from_another_run_is_refused(self, tmp_path):
        manifest_path = tmp_path / "worker.json"
        WorkerManifest(
            run_id="run-2",
            rank=0,
            world_size=1,
            task_ids=("a",),
            results=(WorkerResult(task_id="a", rank=0),),
        ).write(manifest_path)
        with pytest.raises(ValueError, match="does not belong to this distributed rank"):
            DistributedExecutor(_config(tmp_path)).run(
                _tasks(tmp_path, "a"), manifest_path=manifest_path
            )

    def test_a_failed_task_is_retried(self, tmp_path):
        manifest_path = tmp_path / "worker.json"
        WorkerManifest(
            run_id="run-1",
            rank=0,
            world_size=1,
            task_ids=("a",),
            results=(WorkerResult(task_id="a", rank=0, error="boom"),),
        ).write(manifest_path)
        DistributedExecutor(_config(tmp_path)).run(
            _tasks(tmp_path, "a"), manifest_path=manifest_path
        )
        # Only successes are resumable: a rank that died mid-task must redo it.
        assert (tmp_path / "a.done").exists()


class TestMergeManifests:
    def _write(self, path: Path, rank: int, world_size: int, task_ids: tuple[str, ...]) -> Path:
        WorkerManifest(
            run_id="run-1",
            rank=rank,
            world_size=world_size,
            task_ids=task_ids,
            results=tuple(WorkerResult(task_id=task, rank=rank) for task in task_ids),
        ).write(path)
        return path

    def test_a_complete_world_merges_in_task_order(self, tmp_path):
        paths = [
            self._write(tmp_path / "rank-0.json", 0, 2, ("a", "c")),
            self._write(tmp_path / "rank-1.json", 1, 2, ("b",)),
        ]
        merged = DistributedExecutor.merge_manifests(paths)
        assert [result.task_id for result in merged] == ["a", "b", "c"]

    def test_an_incomplete_world_is_refused(self, tmp_path):
        paths = [self._write(tmp_path / "rank-0.json", 0, 2, ("a",))]
        with pytest.raises(ValueError):
            DistributedExecutor.merge_manifests(paths)

    def test_an_incomplete_world_can_be_accepted_explicitly(self, tmp_path):
        paths = [self._write(tmp_path / "rank-0.json", 0, 2, ("a",))]
        merged = DistributedExecutor.merge_manifests(paths, require_complete_world=False)
        assert [result.task_id for result in merged] == ["a"]

    def test_no_manifests_is_an_error(self):
        with pytest.raises(ValueError, match="at least one worker manifest"):
            DistributedExecutor.merge_manifests([])
