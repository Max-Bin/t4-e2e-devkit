"""Regression tests for the cache-free online GPU PDM reference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from t4_e2e_devkit.common.dataclasses import SceneFilter
from t4_e2e_devkit.dataset.window import T4WindowBuilder
from t4_e2e_devkit.evaluation.gpu.geometry import TorchPolyline
from t4_e2e_devkit.evaluation.gpu.geometry_ops import (
    oriented_box_corners,
    polyline_polygon_buffer_intersects,
)
from t4_e2e_devkit.evaluation.gpu.reference import compute_online_pdm_references
from t4_e2e_devkit.evaluation.reference.pdm_closed import compute_t4_pdm_reference


def test_online_reference_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="require CUDA"):
        compute_online_pdm_references([], device=torch.device("cpu"))


def test_square_cap_extends_outward_from_subline_endpoint() -> None:
    path = TorchPolyline(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))
    near_cap = oriented_box_corners(
        torch.tensor([[-1.5, 0.0]]),
        torch.zeros(1),
        torch.full((1,), 0.2),
        torch.full((1,), 0.2),
    )
    beyond_cap = oriented_box_corners(
        torch.tensor([[-2.3, 0.0]]),
        torch.zeros(1),
        torch.full((1,), 0.2),
        torch.full((1,), 0.2),
    )

    near_result = polyline_polygon_buffer_intersects(
        path, near_cap, 2.0, torch.tensor(0.0), torch.tensor(10.0)
    )
    beyond_result = polyline_polygon_buffer_intersects(
        path, beyond_cap, 2.0, torch.tensor(0.0), torch.tensor(10.0)
    )

    assert near_result.tolist() == [True]
    assert beyond_result.tolist() == [False]


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cache_free_online_reference_matches_cpu_definition() -> None:
    root = Path(os.environ.get("T4E2E_TEST_ROOT", "data/t4_dataset"))
    scene_dirs = sorted(root.glob("prd_jt/*/*/*"))
    if not scene_dirs:
        pytest.skip("no T4 scene available")

    scene_dir = scene_dirs[0]
    center = 30
    builder = T4WindowBuilder(
        scene_dir,
        root,
        scene_filter=SceneFilter(num_history_frames=31, num_future_frames=80),
        reader_config={"t4_pdm_reference_device": "gpu"},
    )
    try:
        scene = builder.build(center)
        assert scene.pdm_progress is None

        cpu = compute_t4_pdm_reference(builder.reader, center)
        gpu = compute_online_pdm_references([scene], device=torch.device("cuda"))

        assert cpu.selected_proposal == int(gpu.selected_proposal[0].item())
        np.testing.assert_allclose(
            cpu.proposal_scores,
            gpu.proposal_scores[0].cpu().numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            cpu.reference_trajectory,
            gpu.reference_trajectory[0].cpu().numpy(),
            rtol=0.0,
            atol=5.0e-6,
        )
        np.testing.assert_allclose(
            cpu.pdm_progress,
            gpu.pdm_progress[0].item(),
            rtol=0.0,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            cpu.reference_nc,
            gpu.reference_nc[0].item(),
            rtol=0.0,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            cpu.reference_dac,
            gpu.reference_dac[0].item(),
            rtol=0.0,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            cpu.reference_raw_progress,
            gpu.reference_raw_progress[0].item(),
            rtol=0.0,
            atol=1.0e-9,
        )
    finally:
        builder.close()
