"""Independent PDM, open-loop, T4 and closed-loop evaluation families."""

from t4_e2e_devkit.common.dataclasses import (
    PDMResults,
    aggregate_pdm_results,
    aggregate_pdm_score,
    aggregate_results,
)
from t4_e2e_devkit.evaluation.closed_loop import (
    ClosedLoopMetricConfig,
    ClosedLoopMetrics,
    ClosedLoopTrace,
    aggregate_closed_loop_metrics,
    compute_closed_loop_metrics,
)
from t4_e2e_devkit.evaluation.closed_loop_artifact import (
    load_rollout_artifact,
    load_rollout_metrics,
)
from t4_e2e_devkit.evaluation.closed_loop_report import (
    write_static_html_report,
)
from t4_e2e_devkit.evaluation.executor import LocalExecutor, shard_indices
from t4_e2e_devkit.evaluation.metric_cache import MetricCache
from t4_e2e_devkit.evaluation.metric_engine import (
    MetricContext,
    MetricDefinition,
    MetricEngine,
    MetricRecord,
    MetricReport,
)
from t4_e2e_devkit.evaluation.open_loop import (
    OpenLoopMetricConfig,
    OpenLoopMetrics,
    aggregate_open_loop_metrics,
    compute_open_loop_metrics,
)
from t4_e2e_devkit.evaluation.pdm_score import (
    BACKENDS,
    ScoringError,
    T4PDMScorer,
    T4PDMScorerConfig,
    compare_backends,
)
from t4_e2e_devkit.evaluation.report import aggregate_evaluation
from t4_e2e_devkit.evaluation.tier4_metrics import RewardConfig, aggregate_tier4_metrics

__all__ = [
    "BACKENDS",
    "ClosedLoopMetricConfig",
    "ClosedLoopMetrics",
    "ClosedLoopTrace",
    "OpenLoopMetricConfig",
    "OpenLoopMetrics",
    "PDMResults",
    "RewardConfig",
    "ScoringError",
    "T4PDMScorer",
    "T4PDMScorerConfig",
    "aggregate_closed_loop_metrics",
    "aggregate_evaluation",
    "aggregate_open_loop_metrics",
    "aggregate_pdm_results",
    "aggregate_pdm_score",
    "aggregate_results",
    "aggregate_tier4_metrics",
    "compare_backends",
    "compute_closed_loop_metrics",
    "compute_open_loop_metrics",
    "load_rollout_artifact",
    "load_rollout_metrics",
    "MetricCache",
    "MetricContext",
    "MetricDefinition",
    "MetricEngine",
    "MetricRecord",
    "MetricReport",
    "LocalExecutor",
    "shard_indices",
    "write_static_html_report",
]
