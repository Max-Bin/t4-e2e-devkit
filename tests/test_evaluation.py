"""Evaluation tests.

The important one is :class:`TestBackendAgreement`.  The GPU scorer exists so it
can run inside a training step; the CPU judge is what defines the right answer.
Nothing else in the devkit keeps the fast path honest -- so if that test is
skipped or deleted, a regression in the GPU kernels becomes a training run that
optimises a slightly wrong objective and reports numbers that look ordinary.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from t4_e2e_devkit.agents import build_agent
from t4_e2e_devkit.common import constants as C
from t4_e2e_devkit.common.dataclasses import SceneFilter, Trajectory
from t4_e2e_devkit.dataset.datalist import DataList
from t4_e2e_devkit.dataset.dataset import T4Dataset
from t4_e2e_devkit.evaluation.pdm_score import ScoringError, T4PDMScorer, T4PDMScorerConfig
from t4_e2e_devkit.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

#: A PDM-Closed reference cache and the drivable-area buffer it was built with.
#: Both are needed: ``pdm_progress`` is the ego-progress denominator, and the
#: buffer is part of the cache signature.
PDM_CACHE = os.environ.get("T4E2E_TEST_PDM_CACHE")
PDM_BUFFER = float(os.environ.get("T4E2E_TEST_PDM_BUFFER", "0.0"))

requires_cache = pytest.mark.skipif(
    not PDM_CACHE or not Path(PDM_CACHE).is_dir(),
    reason="set T4E2E_TEST_PDM_CACHE to a PDM-Closed reference cache",
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_raw_proposals_are_resampled_from_their_declared_grid():
    scorer = T4PDMScorer(
        backend="cpu",
        config=T4PDMScorerConfig(num_poses=80, trajectory_interval=0.1),
    )
    sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
    times = torch.arange(1, 81, dtype=torch.float32) * 0.1
    proposals = torch.zeros((1, 1, 80, 3), dtype=torch.float32)
    proposals[0, 0, :, 0] = times

    adapted = scorer._resample_proposals(proposals, sampling)

    assert adapted.shape == (1, 1, C.TRAJECTORY_POSES, 3)
    torch.testing.assert_close(
        adapted[0, 0, :, 0], torch.arange(1, 9, dtype=torch.float32) * 0.5
    )


def test_score_proposals_rejects_a_batch_scene_mismatch():
    scorer = T4PDMScorer(backend="cpu")
    proposals = torch.zeros((2, 1, C.TRAJECTORY_POSES, 3), dtype=torch.float32)

    with pytest.raises(ScoringError, match="2 proposal batches for 0 scenes"):
        scorer.score_proposals(proposals, [])


class TestBackendSelection:
    def test_unknown_backend_is_refused(self):
        with pytest.raises(ValueError, match="unknown scoring backend"):
            T4PDMScorer(backend="magic")

    @pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
    def test_gpu_request_without_cuda_does_not_fall_back(self):
        # A silent fall back to the reference judge turns a training step into a
        # minutes-long stall, so it is an error rather than a courtesy.
        with pytest.raises(ValueError, match="never substituted silently"):
            T4PDMScorer(backend="gpu")


@pytest.fixture(scope="module")
def scored_scenes(request):
    """A handful of windows with valid PDM-Closed labels, and their scenes."""
    root = Path(os.environ.get("T4E2E_TEST_ROOT", "data/t4_dataset"))
    if not root.is_dir():
        pytest.skip(f"T4 dataset not found at {root}")
    if not PDM_CACHE:
        pytest.skip("set T4E2E_TEST_PDM_CACHE")

    scene_dirs = sorted(root.glob("prd_jt/*/*/*"))
    if not scene_dirs:
        pytest.skip("no T4 scenes")
    relative = str(scene_dirs[0].relative_to(root))

    data_list = DataList(root=root, rows=[(relative, center) for center in (100, 150, 200, 250)])
    dataset = T4Dataset(
        data_list,
        scene_filter=SceneFilter(num_future_frames=C.PDM_OBSERVATION_FRAMES + 1),
        reader_config={
            "t4_load_oracle_targets": True,
            "t4_pdm_reference_cache_dir": PDM_CACHE,
            "t4_drivable_area_buffer_m": PDM_BUFFER,
        },
    )
    scenes = []
    for index in range(len(dataset)):
        try:
            scene = dataset[index]
        except Exception:  # noqa: BLE001
            continue
        if scene.pdm_progress is not None:
            scenes.append(scene)
    dataset.close()
    if not scenes:
        pytest.skip("no window in the fixture scene has a valid PDM-Closed label")
    return scenes


@pytest.mark.data
@requires_cache
class TestScoringGuards:
    """What the scorer refuses to do, and why."""

    def test_length_mismatch_is_refused(self, scored_scenes):
        scorer = T4PDMScorer(backend="cpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        trajectory = Trajectory(poses=np.zeros((C.TRAJECTORY_POSES, 3), dtype=np.float32))
        with pytest.raises(ScoringError, match="trajector"):
            scorer.score_batch([trajectory], scored_scenes[:2])

    def test_missing_pdm_progress_is_refused(self, scored_scenes):
        # Falling back to the demonstrated endpoint changes what EP measures
        # rather than approximating it, so a missing denominator must stop the
        # run instead of being filled in.
        import copy

        scene = copy.copy(scored_scenes[0])
        scene.pdm_progress = None
        scorer = T4PDMScorer(backend="cpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        trajectory = Trajectory(poses=np.zeros((C.TRAJECTORY_POSES, 3), dtype=np.float32))
        with pytest.raises(ScoringError, match="not a substitute"):
            scorer.score_batch([trajectory], [scene])

    def test_missing_future_is_refused(self, scored_scenes):
        import copy

        scene = copy.copy(scored_scenes[0])
        scene.future_annotations = None
        scorer = T4PDMScorer(backend="cpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        trajectory = Trajectory(poses=np.zeros((C.TRAJECTORY_POSES, 3), dtype=np.float32))
        with pytest.raises(ScoringError, match="empty traffic scene"):
            scorer.score_batch([trajectory], [scene])


@pytest.mark.data
@requires_cache
class TestScoreProperties:
    """Properties the score must have regardless of the data."""

    def test_components_are_in_unit_range(self, scored_scenes):
        scorer = T4PDMScorer(backend="cpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        human = build_agent("human")
        results = scorer.score_batch(
            [human.compute_trajectory_from_scene(scene) for scene in scored_scenes],
            scored_scenes,
        )
        for result in results:
            for name, value in result.components.items():
                assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"
            assert 0.0 <= result.score <= 1.0

    def test_aggregate_matches_the_formula(self, scored_scenes):
        from t4_e2e_devkit.common.dataclasses import aggregate_pdm_score

        scorer = T4PDMScorer(backend="cpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        human = build_agent("human")
        results = scorer.score_batch(
            [human.compute_trajectory_from_scene(scene) for scene in scored_scenes],
            scored_scenes,
        )
        for result in results:
            recomputed = aggregate_pdm_score(list(result.components.values()))
            assert result.score == pytest.approx(recomputed, abs=1e-6)

    def test_human_outscores_constant_velocity(self, scored_scenes):
        # The human replays the recorded future; extrapolating the current
        # velocity ignores the road.  If this inverts, the scorer is measuring
        # something other than driving quality.
        scorer = T4PDMScorer(backend="cpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        human, cv = build_agent("human"), build_agent("constant_velocity")
        human_score = np.mean(
            [
                result.score
                for result in scorer.score_batch(
                    [human.compute_trajectory_from_scene(s) for s in scored_scenes], scored_scenes
                )
            ]
        )
        cv_score = np.mean(
            [
                result.score
                for result in scorer.score_batch(
                    [cv.compute_trajectory(s.get_agent_input()) for s in scored_scenes],
                    scored_scenes,
                )
            ]
        )
        assert human_score >= cv_score


@pytest.mark.data
@pytest.mark.gpu
@requires_cache
@requires_cuda
class TestBackendAgreement:
    """The GPU scorer must agree with the reference judge.

    Tolerance is 1e-3 -- far above the ~2e-7 float32 agreement measured on real
    windows, and far below any deviation that would change a decision.  A
    failure here means the fast path drifted, not that the tolerance is tight.
    """

    def test_components_agree(self, scored_scenes):
        human = build_agent("human")
        trajectories = [human.compute_trajectory_from_scene(scene) for scene in scored_scenes]

        cpu = T4PDMScorer(backend="cpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        gpu = T4PDMScorer(backend="gpu", config=T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER))
        cpu_results = cpu.score_batch(trajectories, scored_scenes)
        gpu_results = gpu.score_batch(trajectories, scored_scenes)

        for name in C.PDM_COMPONENT_ORDER:
            deltas = [
                abs(a.components[name] - b.components[name])
                for a, b in zip(cpu_results, gpu_results, strict=True)
            ]
            assert max(deltas) < 1e-3, f"{name} disagrees by {max(deltas):.2e}"

    def test_aggregate_agrees(self, scored_scenes):
        cv = build_agent("constant_velocity")
        trajectories = [cv.compute_trajectory(scene.get_agent_input()) for scene in scored_scenes]
        config = T4PDMScorerConfig(t4_drivable_area_buffer_m=PDM_BUFFER)
        cpu = T4PDMScorer(backend="cpu", config=config).score_batch(trajectories, scored_scenes)
        gpu = T4PDMScorer(backend="gpu", config=config).score_batch(trajectories, scored_scenes)
        for a, b in zip(cpu, gpu, strict=True):
            assert a.score == pytest.approx(b.score, abs=1e-3)


@pytest.mark.data
class TestTier4Metrics:
    """The extra family is reported alongside, never folded into the score."""

    def test_metrics_are_computed(self, scene):
        from t4_e2e_devkit.evaluation.tier4_metrics import compute_tier4_metrics

        human = build_agent("human")
        metrics = compute_tier4_metrics(human.compute_trajectory_from_scene(scene), scene)
        assert {"kinematic_gate", "feasibility", "red_light"} <= set(metrics)
        assert all(np.isfinite(value) for value in metrics.values())

    def test_densify_matches_the_metric_grid(self, scene):
        from t4_e2e_devkit.evaluation.tier4_metrics import densify_trajectory

        human = build_agent("human")
        dense = densify_trajectory(human.compute_trajectory_from_scene(scene).poses)
        # The thresholds are calibrated at dt = 0.1; feeding 0.5 s poses would
        # divide by the wrong dt and read every acceleration five times small.
        assert dense.shape == (1, C.SCORER_FUTURE_FRAMES, 4)
        norms = dense[0, :, 2:].norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)

    def test_tier4_metrics_do_not_enter_the_score(self):
        from t4_e2e_devkit.common.dataclasses import PDMResults

        base = PDMResults.from_components([1.0] * 6)
        with_extras = PDMResults.from_components(
            [1.0] * 6, tier4_metrics={"red_light": -5.0, "feasibility": -3.0}
        )
        assert base.score == with_extras.score
