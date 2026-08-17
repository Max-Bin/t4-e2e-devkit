"""Small pandas-backed view of metric result rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Optional, Sequence


@dataclass
class MetricStatisticsDataFrame:
    metric_statistic_name: str
    metric_statistics_dataframe: Any

    time_series_unit_column: ClassVar[str] = "time_series_unit"
    time_series_timestamp_column: ClassVar[str] = "time_series_timestamps"
    time_series_values_column: ClassVar[str] = "time_series_values"
    time_series_selected_frames_column: ClassVar[str] = "time_series_selected_frames"

    @classmethod
    def from_results(
        cls,
        results: Iterable[Any],
        name: str = "metric",
        *,
        scenario_fields: Optional[Iterable[dict[str, Any]]] = None,
    ) -> "MetricStatisticsDataFrame":
        import pandas as pd

        rows = []
        fields = iter(scenario_fields or ())
        for result in results:
            serializer = getattr(result, "serialize_dataframe", None)
            if serializer is None:
                row = {
                    "metric_statistics_name": str(getattr(result, "name", name)),
                    "metric_score": getattr(result, "metric_score", None),
                }
                row.update(
                    {
                        str(statistic.name): statistic.value
                        for statistic in getattr(result, "statistics", ())
                    }
                )
            else:
                row = dict(serializer())
            row.update(next(fields, {}))
            rows.append(row)
        return cls(name, pd.DataFrame(rows))

    @property
    def dataframe(self) -> Any:
        return self.metric_statistics_dataframe

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricStatisticsDataFrame):
            return NotImplemented
        return self.metric_statistic_name == other.metric_statistic_name and self.dataframe.equals(
            other.dataframe
        )

    @property
    def metric_statistics_names(self) -> list[str]:
        if "metric_statistics_name" in self.dataframe:
            return [str(item) for item in self.dataframe["metric_statistics_name"].unique()]
        if "metric_name" in self.dataframe:
            return [str(item) for item in self.dataframe["metric_name"].unique()]
        return [self.metric_statistic_name]

    @property
    def column_names(self) -> list[str]:
        return [str(item) for item in self.dataframe.columns]

    @property
    def statistic_names(self) -> list[str]:
        return sorted(
            {
                column.removesuffix("_stat_type")
                for column in self.column_names
                if column.endswith("_stat_type")
            }
        )

    @property
    def metric_computator(self) -> str:
        return str(self.dataframe["metric_computator"].iloc[0])

    @property
    def metric_category(self) -> str:
        return str(self.dataframe["metric_category"].iloc[0])

    @property
    def metric_score_unit(self) -> Optional[str]:
        if "metric_score_unit" not in self.dataframe:
            return None
        value = self.dataframe["metric_score_unit"].iloc[0]
        return None if value is None else str(value)

    @property
    def planner_names(self) -> list[str]:
        return self._unique("planner_name")

    @property
    def scenario_names(self) -> list[str]:
        return self._unique("scenario_name")

    @property
    def scenario_types(self) -> list[str]:
        return self._unique("scenario_type")

    @property
    def time_series_headers(self) -> list[str]:
        return [
            self.time_series_unit_column,
            self.time_series_timestamp_column,
            self.time_series_values_column,
            self.time_series_selected_frames_column,
        ]

    @property
    def time_series_dataframe(self) -> Any:
        columns = [column for column in self.time_series_headers if column in self.dataframe]
        return self.dataframe.loc[:, columns]

    def query_scenarios(
        self,
        *,
        scenario_names: Optional[Sequence[str]] = None,
        scenario_types: Optional[Sequence[str]] = None,
        planner_names: Optional[Sequence[str]] = None,
        log_names: Optional[Sequence[str]] = None,
    ) -> Any:
        """Filter rows by portable scenario/report fields."""

        result = self.dataframe
        for column, values in (
            ("scenario_name", scenario_names),
            ("scenario_type", scenario_types),
            ("planner_name", planner_names),
            ("log_name", log_names),
        ):
            if values and column in result:
                result = result[result[column].isin(tuple(values))]
        return result

    def statistics_dataframe(self, statistic_names: Optional[Sequence[str]] = None) -> Any:
        """Return statistic value/type/unit columns."""

        if statistic_names:
            wanted = []
            for name in statistic_names:
                wanted.extend(
                    column
                    for column in (f"{name}_stat_type", f"{name}_stat_unit", f"{name}_stat_value")
                    if column in self.dataframe
                )
            return self.dataframe.loc[:, wanted]
        columns = [
            column
            for column in self.column_names
            if "_stat_type" in column or "_stat_unit" in column or "_stat_value" in column
        ]
        return self.dataframe.loc[:, columns]

    def to_csv(self, path: str) -> None:
        self.dataframe.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str, name: Optional[str] = None) -> "MetricStatisticsDataFrame":
        import pandas as pd

        return cls(name or path.rsplit("/", 1)[-1].rsplit(".", 1)[0], pd.read_csv(path))

    def _unique(self, column: str) -> list[str]:
        if column not in self.dataframe:
            return []
        return [str(item) for item in self.dataframe[column].dropna().unique()]


__all__ = ["MetricStatisticsDataFrame"]
