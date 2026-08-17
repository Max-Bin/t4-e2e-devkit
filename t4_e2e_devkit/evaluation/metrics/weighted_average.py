"""Family-preserving weighted metric aggregation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .metric_result import MetricStatistics, MetricStatisticsType, Statistic


class WeightedAverageMetricAggregator:
    """Average compatible metric statistics without mixing categories."""

    def aggregate(
        self,
        results: Iterable[MetricStatistics],
        *,
        weights: Optional[Mapping[str, float]] = None,
    ) -> list[MetricStatistics]:
        values = list(results)
        if not values:
            return []
        key = (values[0].name, values[0].metric_category)
        if any((item.name, item.metric_category) != key for item in values):
            raise ValueError("weighted aggregation requires one metric and category")
        grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
        for result in values:
            weight = 1.0 if weights is None else float(weights.get(result.name, 1.0))
            if weight < 0.0:
                raise ValueError("metric weights must be non-negative")
            for statistic in result.statistics:
                if isinstance(statistic.value, bool):
                    continue
                grouped.setdefault((statistic.name, statistic.unit, statistic.type.serialize()), []).append((float(statistic.value), weight))
        statistics = []
        for (name, unit, kind), samples in sorted(grouped.items()):
            denominator = sum(weight for _, weight in samples)
            value = float(sum(sample * weight for sample, weight in samples) / denominator) if denominator else float("nan")
            statistics.append(Statistic(name, unit, MetricStatisticsType.deserialize(kind), value))
        return [
            MetricStatistics(
                metric_computator=type(self).__name__,
                name=key[0],
                metric_category=key[1],
                statistics=statistics,
            )
        ]

    def aggregate_metric_statistics(self, results: Iterable[MetricStatistics], **kwargs: Any) -> list[MetricStatistics]:
        return self.aggregate(results, **kwargs)


__all__ = ["WeightedAverageMetricAggregator"]
