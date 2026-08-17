"""Build NuPlan-like scenario objects directly from T4 data lists."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, Optional

from t4_e2e_devkit.common.constants import T4_INTERVAL_LENGTH
from t4_e2e_devkit.common.dataclasses import SceneFilter, SensorConfig
from t4_e2e_devkit.dataset.datalist import DataList, load_data_list
from t4_e2e_devkit.dataset.window import T4WindowBuilder
from t4_e2e_devkit.planning.scenario_builder.abstract_scenario import T4Scenario


class T4ScenarioBuilder:
    """Enumerate T4 windows as independent scenario objects.

    The builder owns only reader construction. A yielded scenario contains
    materialized arrays and remains usable after the builder closes its scene
    reader. Data-list rows are already the reproducible scenario index, so no
    database or hidden directory scan is performed here.
    """

    def __init__(
        self,
        data_list: str | Path | DataList,
        *,
        scene_filter: Optional[SceneFilter] = None,
        sensor_config: Optional[SensorConfig] = None,
        reader_config: Optional[Dict[str, Any]] = None,
        interval_length: float = T4_INTERVAL_LENGTH,
        include_history_annotations: bool = True,
    ) -> None:
        self.data_list = (
            data_list if isinstance(data_list, DataList) else load_data_list(data_list)
        )
        self.scene_filter = scene_filter or SceneFilter()
        self.sensor_config = sensor_config or SensorConfig.build_no_sensors()
        self.reader_config = dict(reader_config or {})
        self.reader_config["t4_include_history_annotations"] = bool(include_history_annotations)
        self.interval_length = float(interval_length)
        if self.interval_length <= 0.0:
            raise ValueError("interval_length must be positive")

    def get_scenarios(self, limit: Optional[int] = None) -> Iterator[T4Scenario]:
        """Yield scenarios in data-list order, closing readers on completion."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        builders: dict[str, T4WindowBuilder] = {}
        try:
            for row_index, (scene_dir, center_frame) in enumerate(self.data_list):
                if limit is not None and row_index >= limit:
                    break
                builder = builders.get(scene_dir)
                if builder is None:
                    builder = T4WindowBuilder(
                        self.data_list.absolute_scene_dir(scene_dir),
                        self.data_list.root,
                        sensor_config=self.sensor_config,
                        scene_filter=self.scene_filter,
                        reader_config=self.reader_config,
                    )
                    builders[scene_dir] = builder
                scene = builder.build(int(center_frame))
                yield T4Scenario(
                    scene,
                    self.interval_length,
                    map_api=builder.map_api,
                )
        finally:
            for builder in builders.values():
                builder.close()

    def build_all(self, limit: Optional[int] = None) -> list[T4Scenario]:
        """:return: materialized scenarios, useful for small evaluation sets."""
        return list(self.get_scenarios(limit=limit))

    def get_scenario(self, token: str) -> T4Scenario:
        """Build one scenario from ``scene_relative_path@center_frame``."""
        scene_dir, separator, center = str(token).rpartition("@")
        if not separator or not scene_dir:
            raise ValueError(f"invalid T4 scenario token: {token!r}")
        try:
            center_frame = int(center)
        except ValueError as error:
            raise ValueError(f"invalid center frame in scenario token: {token!r}") from error
        for row_scene, row_center in self.data_list:
            if row_scene == scene_dir and int(row_center) == center_frame:
                return self._build_row(row_scene, row_center)
        raise KeyError(f"scenario token is not present in the data list: {token}")

    def _build_row(self, scene_dir: str, center_frame: int) -> T4Scenario:
        """Build one row with a private reader whose lifetime ends on return."""

        builder = T4WindowBuilder(
            self.data_list.absolute_scene_dir(scene_dir),
            self.data_list.root,
            sensor_config=self.sensor_config,
            scene_filter=self.scene_filter,
            reader_config=self.reader_config,
        )
        try:
            scene = builder.build(int(center_frame))
            return T4Scenario(scene, self.interval_length, map_api=builder.map_api)
        finally:
            builder.close()


__all__ = ["T4ScenarioBuilder"]
