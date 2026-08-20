from __future__ import annotations

import numpy as np
import pytest
import torch

from t4_e2e_devkit.common.dataclasses import SceneMetadata
from t4_e2e_devkit.evaluation.gpu.navsim import _states_to_global
from t4_e2e_devkit.evaluation.navsim_score import (
    NAVSIM_V1_METRICS,
    NAVSIM_V2_METRICS,
    NavSimScoringError,
    T4NavSimResult,
    T4NavSimScorer,
    T4NavSimScorerConfig,
    required_navsim_metric_names,
    resolve_navsim_metric_names,
)
from t4_e2e_devkit.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


def _scene(center: tuple[float, float, float, float] | None) -> object:
    return type(
        "SceneStub",
        (),
        {
            "scene_metadata": SceneMetadata(
                scene_dir="scene",
                scene_id="scene",
                center_frame=0,
                num_history_frames=1,
                num_future_frames=1,
                global_center_pose=None if center is None else np.array(center),
            )
        },
    )()


def test_pdm_versions_are_the_only_metric_versions():
    assert T4NavSimScorerConfig(version="v1").version == "v1"
    assert T4NavSimScorerConfig(version="v2").version == "v2"
    with pytest.raises(ValueError, match="version"):
        T4NavSimScorerConfig(version="t4")


def test_metric_selection_preserves_order_and_adds_only_internal_dependencies():
    assert resolve_navsim_metric_names("v1") == NAVSIM_V1_METRICS
    assert resolve_navsim_metric_names("v2", ("score", "ego_progress")) == (
        "score",
        "ego_progress",
    )
    assert required_navsim_metric_names("v2", ("score",)) == frozenset(NAVSIM_V2_METRICS)

    config = T4NavSimScorerConfig(
        version="v2",
        metric_names=("score", "ego_progress"),
        backend="cpu",
    )
    assert config.metric_names == ("score", "ego_progress")
    assert config.selected_metric_names == ("score", "ego_progress")

    result = T4NavSimResult("v2", {"score": 0.4, "ego_progress": 0.8})
    assert result.metric_names == ("score", "ego_progress")
    assert result.values == {"score": 0.4, "ego_progress": 0.8}
    with pytest.raises(NavSimScoringError, match="was not requested"):
        _ = result.value("history_comfort")


def test_metric_selection_rejects_unknown_duplicate_and_version_specific_names():
    with pytest.raises(ValueError, match="unknown metrics"):
        resolve_navsim_metric_names("v2", ("not_a_metric",))
    with pytest.raises(ValueError, match="duplicates"):
        resolve_navsim_metric_names("v2", ("score", "score"))
    with pytest.raises(ValueError, match="unknown metrics"):
        resolve_navsim_metric_names("v1", ("extended_comfort",))


def test_score_proposals_returns_selected_columns_with_explicit_order(monkeypatch):
    scorer = T4NavSimScorer(
        T4NavSimScorerConfig(
            version="v2",
            backend="cpu",
            metric_names=("score", "ego_progress"),
        )
    )

    def fake_score_batch(trajectories, scenes, **kwargs):
        assert kwargs["metric_names"] == ("score", "ego_progress")
        return tuple(
            T4NavSimResult(
                "v2",
                {"score": float(index), "ego_progress": float(index + 1)},
            )
            for index in range(len(trajectories))
        )

    monkeypatch.setattr(scorer, "score_batch", fake_score_batch)
    output = scorer.score_proposals(
        torch.zeros((1, 2, 3, 3)),
        [object()],
    )

    assert output.metric_names == ("score", "ego_progress")
    assert tuple(output.shape) == (1, 2, 2)
    assert output.values[0, 1].tolist() == [1.0, 2.0]


def test_gpu_extended_comfort_transform_moves_pose_but_keeps_body_kinematics():
    states = torch.zeros((1, 11), dtype=torch.float64)
    states[0, StateIndex.X] = 2.0
    states[0, StateIndex.Y] = 1.0
    states[0, StateIndex.HEADING] = 0.25
    states[0, StateIndex.VELOCITY_X] = 4.0
    states[0, StateIndex.ACCELERATION_X] = 0.5

    transformed = _states_to_global(states, _scene((10.0, -3.0, 1.0, 0.0)))

    assert transformed[0, StateIndex.X] == pytest.approx(12.0)
    assert transformed[0, StateIndex.Y] == pytest.approx(-2.0)
    assert transformed[0, StateIndex.HEADING] == pytest.approx(0.25)
    assert transformed[0, StateIndex.VELOCITY_X] == pytest.approx(4.0)
    assert transformed[0, StateIndex.ACCELERATION_X] == pytest.approx(0.5)


def test_gpu_extended_comfort_transform_rotates_scene_heading():
    states = torch.zeros((1, 11), dtype=torch.float32)
    states[0, StateIndex.X] = 1.0
    states[0, StateIndex.HEADING] = 0.0
    angle = np.pi / 2.0

    transformed = _states_to_global(
        states,
        _scene((2.0, 3.0, float(np.cos(angle)), float(np.sin(angle)))),
    )

    assert transformed[0, StateIndex.X] == pytest.approx(2.0)
    assert transformed[0, StateIndex.Y] == pytest.approx(4.0)
    assert transformed[0, StateIndex.HEADING] == pytest.approx(angle)


def test_scorer_rejects_gpu_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("this guard is for hosts without CUDA")
    with pytest.raises(ValueError, match="CUDA"):
        T4NavSimScorer(T4NavSimScorerConfig(backend="gpu"))
