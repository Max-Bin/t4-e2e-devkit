"""Aggregator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class AbstractMetricAggregator(ABC):
    @abstractmethod
    def aggregate_metric_statistics(self, metric_statistics: Iterable[Any], **kwargs: Any) -> list[Any]:
        """Aggregate results from one metric family."""


__all__ = ["AbstractMetricAggregator"]
