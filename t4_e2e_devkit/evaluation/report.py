"""Family-separated evaluation reports.

The report shape follows the same principle as metric engines used by mature
planning toolkits: each family has its own per-window records and aggregate.
There is deliberately no cross-family total score.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from t4_e2e_devkit.evaluation.closed_loop import (
    ClosedLoopMetrics,
    aggregate_closed_loop_metrics,
)
from t4_e2e_devkit.evaluation.navsim_score import aggregate_navsim_results
from t4_e2e_devkit.evaluation.open_loop import (
    OpenLoopMetrics,
    aggregate_open_loop_metrics,
)


def aggregate_evaluation(
    *,
    pdm: Sequence[Mapping[str, float]] = (),
    open_loop: Sequence[OpenLoopMetrics] = (),
    closed_loop: Sequence[ClosedLoopMetrics] = (),
    num_failed: int = 0,
) -> dict[str, dict[str, float]]:
    """Aggregate the requested metric families independently.

    The returned mapping is suitable for YAML/JSON serialization. Each family
    is aggregated independently and no family contributes to another family's
    denominator.
    """

    if num_failed < 0:
        raise ValueError("num_failed must be non-negative")
    return {
        "pdm": aggregate_navsim_results(pdm),
        "open_loop": aggregate_open_loop_metrics(open_loop),
        "closed_loop": aggregate_closed_loop_metrics(closed_loop),
        "run": {"num_failed": float(num_failed)},
    }


__all__ = ["aggregate_evaluation"]
