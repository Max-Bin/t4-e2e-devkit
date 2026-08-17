"""Contract tests for the independent evaluation families."""

from __future__ import annotations

import numpy as np
import pytest

from t4_e2e_devkit.common.dataclasses import PDMResults, Trajectory, aggregate_results
from t4_e2e_devkit.dataset.datalist import DataList
from t4_e2e_devkit.evaluation.closed_loop import compute_closed_loop_metrics
from t4_e2e_devkit.evaluation.open_loop import (
    OpenLoopMetricConfig,
    compute_open_loop_metrics,
)
from t4_e2e_devkit.evaluation.report import aggregate_evaluation
from t4_e2e_devkit.planning.simulation.closed_loop import (
    KinematicState,
    T4ClosedLoopResult,
)
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


def _trajectory(num_poses: int, interval: float) -> Trajectory:
    times = np.arange(1, num_poses + 1, dtype=np.float32) * interval
    return Trajectory(
        poses=np.column_stack((times, np.zeros_like(times), np.zeros_like(times))),
        trajectory_sampling=TrajectorySampling(
            num_poses=num_poses,
            interval_length=interval,
        ),
    )


def test_open_loop_uses_declared_sampling_and_common_horizon():
    result = compute_open_loop_metrics(_trajectory(8, 0.5), _trajectory(80, 0.1))

    assert result.horizon_s == pytest.approx(4.0)
    assert result.num_poses == 40
    assert result.ade_m == pytest.approx(0.0, abs=1e-6)
    assert result.fde_m == pytest.approx(0.0, abs=1e-6)


def test_open_loop_can_require_a_fixed_horizon():
    with pytest.raises(ValueError, match="exceeds prediction horizon"):
        compute_open_loop_metrics(
            _trajectory(8, 0.5),
            _trajectory(80, 0.1),
            config=OpenLoopMetricConfig(
                target_sampling=TrajectorySampling(num_poses=80, interval_length=0.1)
            ),
        )


def test_families_are_not_folded_into_one_aggregate():
    pdm = PDMResults.from_components([1.0] * 6, tier4_metrics={"red_light": 0.0})

    assert "tier4/red_light" not in aggregate_results([pdm])
    report = aggregate_evaluation(
        pdm=[pdm],
        tier4=[{"red_light": 0.0}],
    )
    assert report["pdm"]["score"] == pytest.approx(1.0)
    assert report["tier4"]["red_light"] == pytest.approx(0.0)
    assert report["pdm"] != report["tier4"]


def test_closed_loop_metrics_keep_unknown_events_unreported():
    states = [
        KinematicState(0.0, 0.0, 0.0, 0.0),
        KinematicState(1.0, 0.0, 0.0, 10.0, acceleration_mps2=10.0),
        KinematicState(2.0, 0.0, 0.0, 10.0),
    ]
    result = T4ClosedLoopResult(
        source_frames=np.array([10, 11]),
        states=states,
        plans=[None, None],
        dt_s=0.1,
    )

    metrics = compute_closed_loop_metrics(
        result,
        goal_pose_world=np.array([2.0, 0.0, 1.0]),
        collision_steps=[1],
        timeout=False,
    )

    assert metrics.path_length_m == pytest.approx(2.0)
    assert metrics.goal_reached == 1.0
    assert metrics.collision == 1.0
    assert metrics.first_collision_step == 1.0
    assert metrics.timeout == 0.0
    assert "collision" in metrics.values
    assert metrics.trace is not None
    assert len(metrics.trace.step) == 2
    np.testing.assert_allclose(metrics.trace.path_length_m, [1.0, 2.0])
    assert metrics.trace.collision.tolist() == [False, True]
    assert metrics.trace.rows("scene@10")[1]["source_frame"] == 11


def test_closed_loop_command_writes_family_reports(tmp_path, monkeypatch):
    from t4_e2e_devkit.planning.simulation.closed_loop import T4ClosedLoopResult
    from t4_e2e_devkit.script import evaluate_closed_loop as command

    class _Agent:
        def initialize(self):
            pass

    result = T4ClosedLoopResult(
        source_frames=np.array([10]),
        states=[
            KinematicState(0.0, 0.0, 0.0, 1.0),
            KinematicState(0.1, 0.0, 0.0, 1.0),
        ],
        plans=[None],
        dt_s=0.1,
    )
    monkeypatch.setattr(command, "build_agent", lambda _: _Agent())
    monkeypatch.setattr(command, "run_t4_closed_loop", lambda *args, **kwargs: result)

    report = command.evaluate_closed_loop(
        DataList(root=tmp_path, rows=[("scene", 10)]),
        agent_name="agent",
        output_dir=tmp_path / "report",
        num_steps=1,
    )

    assert report["closed_loop"]["num_rollouts"] == pytest.approx(1.0)
    assert (tmp_path / "report" / "closed_loop.csv").is_file()
    assert (tmp_path / "report" / "aggregate.json").is_file()
    assert (tmp_path / "report" / "aggregate.yaml").is_file()
    assert (tmp_path / "report" / "closed_loop_ticks.csv").is_file()
    assert (tmp_path / "report" / "failures.csv").is_file()
    assert (tmp_path / "report" / "report.html").is_file()
    artifacts = list((tmp_path / "report" / "rollouts").glob("*.json"))
    assert len(artifacts) == 1
    assert (tmp_path / "report" / "run.json").is_file()
    assert "termination/completed" in report["closed_loop"]

    monkeypatch.setattr(
        command,
        "build_agent",
        lambda _: pytest.fail("resume should not build the agent"),
    )
    monkeypatch.setattr(
        command,
        "run_t4_closed_loop",
        lambda *args, **kwargs: pytest.fail("resume should not rerun a completed row"),
    )
    resumed = command.evaluate_closed_loop(
        DataList(root=tmp_path, rows=[("scene", 10)]),
        agent_name="agent",
        output_dir=tmp_path / "report",
        num_steps=1,
        resume=True,
    )
    assert resumed["run"]["num_resumed"] == pytest.approx(1.0)


def test_closed_loop_command_shards_and_retries_rows(tmp_path, monkeypatch):
    from t4_e2e_devkit.planning.simulation.closed_loop import T4ClosedLoopResult
    from t4_e2e_devkit.script import evaluate_closed_loop as command

    class _Agent:
        def initialize(self):
            pass

    result = T4ClosedLoopResult(
        source_frames=np.array([10]),
        states=[
            KinematicState(0.0, 0.0, 0.0, 1.0),
            KinematicState(0.1, 0.0, 0.0, 1.0),
        ],
        plans=[None],
        dt_s=0.1,
    )
    calls = {"runs": 0}

    def _run(*args, **kwargs):
        calls["runs"] += 1
        if calls["runs"] == 1:
            raise RuntimeError("transient")
        return result

    monkeypatch.setattr(command, "build_agent", lambda _: _Agent())
    monkeypatch.setattr(command, "run_t4_closed_loop", _run)

    report = command.evaluate_closed_loop(
        DataList(
            root=tmp_path,
            rows=[("scene0", 10), ("scene1", 11), ("scene2", 12)],
        ),
        agent_name="agent",
        output_dir=tmp_path / "retry",
        num_steps=1,
        shard_index=1,
        num_shards=2,
        max_retries=1,
    )

    assert calls["runs"] == 2
    assert report["run"]["num_rows"] == pytest.approx(1.0)
    assert report["run"]["num_attempts"] == pytest.approx(2.0)
    assert report["run"]["num_failed"] == pytest.approx(0.0)


def test_closed_loop_shards_merge_and_recompute_local_report(tmp_path, monkeypatch):
    from t4_e2e_devkit.planning.simulation.closed_loop import T4ClosedLoopResult
    from t4_e2e_devkit.script import evaluate_closed_loop as command
    from t4_e2e_devkit.script.merge_closed_loop import merge_closed_loop_reports

    class _Agent:
        def initialize(self):
            pass

    result = T4ClosedLoopResult(
        source_frames=np.array([10]),
        states=[
            KinematicState(0.0, 0.0, 0.0, 1.0),
            KinematicState(0.1, 0.0, 0.0, 1.0),
        ],
        plans=[None],
        dt_s=0.1,
    )
    monkeypatch.setattr(command, "build_agent", lambda _: _Agent())
    monkeypatch.setattr(command, "run_t4_closed_loop", lambda *args, **kwargs: result)
    data_list = DataList(root=tmp_path, rows=[("scene0", 10), ("scene1", 11)])

    command.evaluate_closed_loop(
        data_list,
        agent_name="agent",
        output_dir=tmp_path / "shard0",
        num_steps=1,
        shard_index=0,
        num_shards=2,
    )
    command.evaluate_closed_loop(
        data_list,
        agent_name="agent",
        output_dir=tmp_path / "shard1",
        num_steps=1,
        shard_index=1,
        num_shards=2,
    )

    merged = merge_closed_loop_reports(
        [tmp_path / "shard0", tmp_path / "shard1"],
        tmp_path / "merged",
    )
    assert merged["closed_loop"]["num_rollouts"] == pytest.approx(2.0)
    assert merged["run"]["num_completed"] == pytest.approx(2.0)
    assert (tmp_path / "merged" / "report.html").is_file()
    assert len(list((tmp_path / "merged" / "rollouts").glob("*.json"))) == 2
