"""GPU rollout and scoring must not depend on the process's precision settings.

A trainer sets ``set_float32_matmul_precision("high")`` for its model and then
asks this devkit for labels or scores.  Those two uses want different things
from the same process-global, and the rollout is where it shows: its batched
least-squares fits become TF32-eligible, and at training batch sizes cuBLAS
takes that eligibility.  See :mod:`t4_e2e_devkit.evaluation.gpu.precision`.

``torch.autocast`` is the second such setting and the sharper one, because a
mixed-precision training loop leaves it enabled around everything it calls.  It
is covered here on the CPU device type as well as CUDA: the exposed kernels are
plain ``einsum``/``matmul``, so the hazard reproduces without a GPU and these
tests need no ``gpu`` marker.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from t4_e2e_devkit.evaluation.gpu.precision import (
    simulate_proposals_fp32,
    strict_fp32_matmul,
)
from t4_e2e_devkit.evaluation.gpu.simulate import TorchSimulatorConfig
from t4_e2e_devkit.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex,
)

ROWS = 256
STEPS = 40
INTERVAL = 0.1


@contextlib.contextmanager
def _matmul_precision(mode: str):
    previous = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision(mode)
    torch.backends.cudnn.allow_tf32 = mode != "highest"
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cudnn.allow_tf32 = cudnn


def _proposals(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A curving, accelerating batch -- the fits need a non-degenerate profile."""

    generator = torch.Generator(device="cpu").manual_seed(0)
    time = torch.arange(STEPS + 1, dtype=torch.float64) * INTERVAL
    speed = 3.0 + 4.0 * torch.rand((ROWS, 1), generator=generator, dtype=torch.float64)
    curvature = 0.4 * (torch.rand((ROWS, 1), generator=generator, dtype=torch.float64) - 0.5)
    heading = curvature * speed * time
    states = torch.zeros((ROWS, STEPS + 1, 11), dtype=torch.float64)
    states[..., StateIndex.X] = (speed * time) * torch.cos(heading)
    states[..., StateIndex.Y] = (speed * time) * torch.sin(heading)
    states[..., StateIndex.HEADING] = heading
    initial = torch.zeros((ROWS, 11), dtype=torch.float64)
    initial[:, StateIndex.VELOCITY_X] = speed[:, 0]
    return (
        states.to(device=device, dtype=torch.float32),
        initial.to(device=device, dtype=torch.float32),
    )


def test_strict_fp32_matmul_restores_the_caller_s_settings():
    with _matmul_precision("high"):
        with strict_fp32_matmul():
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is True


def test_strict_fp32_matmul_restores_settings_after_an_exception():
    with _matmul_precision("high"):
        with pytest.raises(RuntimeError):
            with strict_fp32_matmul():
                raise RuntimeError("boom")
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is True


_AUTOCAST_CASES = [("cpu", torch.bfloat16)]
if torch.cuda.is_available():
    # fp16 rather than bf16 so the case does not depend on the GPU's bf16 support.
    _AUTOCAST_CASES.append(("cuda", torch.float16))


@pytest.mark.parametrize("device_type,dtype", _AUTOCAST_CASES)
def test_strict_fp32_matmul_disables_and_restores_autocast(device_type, dtype):
    with torch.autocast(device_type, dtype=dtype):
        assert torch.is_autocast_enabled(device_type) is True
        with strict_fp32_matmul():
            assert torch.is_autocast_enabled(device_type) is False
        assert torch.is_autocast_enabled(device_type) is True


def test_strict_fp32_matmul_restores_autocast_after_an_exception():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError):
            with strict_fp32_matmul():
                raise RuntimeError("boom")
        assert torch.is_autocast_enabled("cpu") is True


def _touching_quads() -> tuple[torch.Tensor, torch.Tensor]:
    """A 4 m x 2 m box against copies of itself scattered across its own reach."""

    generator = torch.Generator(device="cpu").manual_seed(0)
    box = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
    offset = torch.rand((4096, 1, 2), generator=generator) * 8.0 - 4.0
    return box[None].expand(4096, 4, 2).contiguous(), box[None] + offset


def test_collision_verdicts_are_autocast_exposed_and_the_guard_covers_them():
    """SAT projection is an ``einsum``, so autocast reaches a discrete verdict.

    Measured with this fixture under ``bfloat16``: the projection comes back
    bf16 and **3 of 4096** intersection verdicts flip.  A flipped verdict would
    be a flipped ``no_at_fault_collisions`` label, which is why the guard is held
    over scoring and not only over the rollout -- though on real windows no label
    has actually been measured to move; see
    ``ProposalOracle.score_with_total`` in OnePlanner for that measurement.
    """

    from t4_e2e_devkit.evaluation.gpu.collisions import _project, quads_intersect

    quad_a, quad_b = _touching_quads()
    reference = quads_intersect(quad_a, quad_b)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _project(quad_a, quad_b)[0].dtype is torch.bfloat16, (
            "autocast no longer reaches the SAT projection; re-measure the "
            "exposure before narrowing strict_fp32_matmul()"
        )
        with strict_fp32_matmul():
            assert _project(quad_a, quad_b)[0].dtype is torch.float32
            assert torch.equal(quads_intersect(quad_a, quad_b), reference)


def test_comfort_filter_is_autocast_exposed_and_the_guard_covers_it():
    """The Savitzky-Golay filter is applied as a matmul, then rounded to 1e-8.

    Measured with this fixture under ``bfloat16``: the filtered signal moves by
    1.56e-02, seven orders of magnitude above the rounding step it feeds.
    """

    from t4_e2e_devkit.evaluation.gpu.comfort import _apply_savgol, _round8

    generator = torch.Generator(device="cpu").manual_seed(0)
    signal = torch.randn((256, 40), generator=generator)

    def filtered() -> torch.Tensor:
        return _round8(_apply_savgol(signal, 8, poly_order=2))

    reference = filtered()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        exposed = filtered()
        assert exposed.dtype is torch.bfloat16
        with strict_fp32_matmul():
            guarded = filtered()
    assert float((reference - exposed.float()).abs().max()) > 1.0e-3
    assert torch.equal(guarded, reference)


@pytest.mark.gpu
def test_rollout_survives_autocast_and_is_bit_identical_under_it():
    """A training loop calls this from inside its own autocast scope.

    Unguarded, the profile fits hand a bf16 matrix to ``torch.linalg.solve``,
    which raises ``NotImplementedError: "lu_factor_magma_batched" not implemented
    for 'BFloat16'``; an fp16 autocast would instead return a silently 10-bit
    least-squares fit.  Guarded, the result is the fp32 one, bit for bit.
    """

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    states, initial = _proposals(device)
    config = TorchSimulatorConfig(wheel_base=2.7, discretization_time=INTERVAL)

    plain = simulate_proposals_fp32(states, initial, config).clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        under_autocast = simulate_proposals_fp32(states, initial, config).clone()

    assert under_autocast.dtype is torch.float32
    assert torch.equal(plain, under_autocast)


@pytest.mark.gpu
@pytest.mark.parametrize("compile_rollout", [False, True])
def test_rollout_is_bit_identical_across_matmul_precisions(compile_rollout):
    """The guarded entry point, driven once per precision, must agree exactly.

    ``compile_rollout=True`` is covered separately because the CUDA-graph cache is
    keyed by config, shapes, dtype and device -- not by precision.  An unguarded
    capture would pin whichever precision was active first and hand it to every
    later caller of that shape.
    """

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    states, initial = _proposals(device)
    config = TorchSimulatorConfig(wheel_base=2.7, discretization_time=INTERVAL)

    def rollout(mode: str) -> torch.Tensor:
        with _matmul_precision(mode):
            return simulate_proposals_fp32(
                states, initial, config, compile_rollout=compile_rollout
            ).clone()

    strict, tf32 = rollout("highest"), rollout("high")
    assert torch.equal(strict, tf32), (
        f"max|delta| {float((strict - tf32).abs().max()):.3e} on "
        f"{int((strict != tf32).sum())} of {strict.numel()} elements"
    )
