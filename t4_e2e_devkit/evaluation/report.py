"""Family-separated evaluation reports.

The report shape follows the same principle as metric engines used by mature
planning toolkits: each family has its own per-window records and aggregate.
There is deliberately no cross-family total score.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from t4_e2e_devkit.common.dataclasses import PDMResults, aggregate_results
from t4_e2e_devkit.evaluation.closed_loop import (
    ClosedLoopMetrics,
    aggregate_closed_loop_metrics,
)
from t4_e2e_devkit.evaluation.open_loop import (
    OpenLoopMetrics,
    aggregate_open_loop_metrics,
)
from t4_e2e_devkit.evaluation.tier4_metrics import aggregate_tier4_metrics


def aggregate_evaluation(
    *,
    pdm: Sequence[PDMResults] = (),
    open_loop: Sequence[OpenLoopMetrics] = (),
    tier4: Sequence[Mapping[str, float]] = (),
    closed_loop: Sequence[ClosedLoopMetrics] = (),
    num_failed: int = 0,
) -> dict[str, dict[str, float]]:
    """Aggregate the requested metric families independently.

    The returned mapping is suitable for YAML/JSON serialization. ``pdm``,
    ``open_loop``, ``tier4`` and ``closed_loop`` never contribute to one
    another's denominators or aggregates.
    """

    if num_failed < 0:
        raise ValueError("num_failed must be non-negative")
    return {
        "pdm": aggregate_results(pdm),
        "open_loop": aggregate_open_loop_metrics(open_loop),
        "tier4": aggregate_tier4_metrics(tier4),
        "closed_loop": aggregate_closed_loop_metrics(closed_loop),
        "run": {"num_failed": float(num_failed)},
    }


__all__ = ["aggregate_evaluation"]
