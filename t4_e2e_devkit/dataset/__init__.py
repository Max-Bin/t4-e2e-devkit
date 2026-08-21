"""T4 dataset: the scene reader, window assembly, data lists and the Dataset."""

from t4_e2e_devkit.dataset.contract import (
    BUNDLE_TO_CONTRACT,
    CONTRACT_MAP_FIELDS,
    T4Batch,
    T4Sample,
    assert_batch_contract,
)
from t4_e2e_devkit.dataset.datalist import (
    DataList,
    describe_data_list,
    load_data_list,
)
from t4_e2e_devkit.dataset.dataset import (
    T4Dataset,
    T4SceneLocalitySampler,
    build_dataset_from_agent,
    collate_t4,
)
from t4_e2e_devkit.dataset.pack_writers import write_bundle, write_lidar_pack
from t4_e2e_devkit.dataset.route import (
    T4RouteMetadata,
    T4RoutePrimitive,
    T4RouteSegment,
    load_t4_route,
)
from t4_e2e_devkit.dataset.scenario_filter import (
    ScenarioFilter,
    ScenarioSampling,
    filter_scenarios,
    sample_scenarios,
    select_scenarios,
)
from t4_e2e_devkit.dataset.scene_tags import T4SceneTag, T4SceneTagIndex
from t4_e2e_devkit.dataset.training_window import (
    TrainingSceneHandles,
    TrainingWindowBuilder,
    expected_map_shapes,
    valid_window_centers,
)
from t4_e2e_devkit.dataset.window import T4WindowBuilder, WindowError

__all__ = [
    "BUNDLE_TO_CONTRACT",
    "CONTRACT_MAP_FIELDS",
    "DataList",
    "T4Batch",
    "T4Dataset",
    "T4Sample",
    "T4SceneLocalitySampler",
    "T4SceneTag",
    "T4SceneTagIndex",
    "T4RouteMetadata",
    "T4RoutePrimitive",
    "T4RouteSegment",
    "T4WindowBuilder",
    "TrainingSceneHandles",
    "TrainingWindowBuilder",
    "ScenarioFilter",
    "ScenarioSampling",
    "WindowError",
    "assert_batch_contract",
    "build_dataset_from_agent",
    "collate_t4",
    "describe_data_list",
    "expected_map_shapes",
    "load_data_list",
    "load_t4_route",
    "filter_scenarios",
    "sample_scenarios",
    "select_scenarios",
    "valid_window_centers",
    "write_bundle",
    "write_lidar_pack",
]
