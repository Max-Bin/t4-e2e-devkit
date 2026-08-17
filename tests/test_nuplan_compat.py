"""Contract tests for the database-free NuPlan-shaped adapters."""

from __future__ import annotations

import inspect

import numpy as np

from t4_e2e_devkit.common.actor_state.state_representation import Point2D
from t4_e2e_devkit.common.maps.abstract_map_objects import Lane, LaneConnector
from t4_e2e_devkit.common.maps.maps_datatypes import SemanticMapLayer
from t4_e2e_devkit.common.maps.t4_map_adapter import T4MapAdapter
from t4_e2e_devkit.common.t4_map import T4MapAPI
from t4_e2e_devkit.dataset.scenario_filter import (
    ScenarioFilter,
    ScenarioSampling,
    filter_scenarios,
    sample_scenarios,
)
from t4_e2e_devkit.evaluation.distributed import (
    DistributedExecutor,
    DistributedRunConfig,
    merge_worker_manifests,
)
from t4_e2e_devkit.evaluation.metric_api import (
    CallableMetricBuilder,
    MetricAggregator,
    MetricBuilderRegistry,
    MetricCallback,
    MetricTimeSeries,
)
from t4_e2e_devkit.evaluation.metric_builders import MappingMetricBuilder
from t4_e2e_devkit.evaluation.worker_pool import (
    WorkerPool,
    WorkerResult,
    WorkerTask,
    merge_worker_results,
)
from t4_e2e_devkit.planning.simulation.planner.abstract_planner import (
    PlannerInput,
)
from t4_e2e_devkit.planning.simulation.runtime import (
    SimulationHistory,
    SimulationManager,
    SimulationRunner,
    SimulationSetup,
    StepSimulationTimeController,
)
from t4_e2e_devkit.planning.simulation.simulation_iteration import SimulationIteration


def _square(value: int) -> int:
    return value * value


def test_worker_pool_partitions_and_merges_results_deterministically():
    with WorkerPool(workers=2, backend="thread", rank=1, world_size=2) as pool:
        values = pool.map(_square, list(range(6)))
        indexed = pool.map_indexed(_square, list(range(6)))

    assert values == [1, 9, 25]
    assert indexed == [(1, 1), (3, 9), (5, 25)]
    merged = merge_worker_results(
        [WorkerResult("b", 2, rank=1), WorkerResult("a", 1, rank=0)],
        expected_task_ids=["a", "b"],
    )
    assert [result.task_id for result in merged] == ["a", "b"]


def test_metric_builder_registry_preserves_time_series_and_family_aggregation():
    series = MetricTimeSeries(np.array([0, 100_000]), np.array([1.0, 3.0]), unit="m")
    builder = CallableMetricBuilder(
        "comfort",
        lambda history: {"max_acceleration": float(len(history)), "series": 2.0},
    )
    registry = MetricBuilderRegistry([builder])
    results = registry.compute([1, 2], scenario_token="scene@0")
    assert results[0].values["max_acceleration"] == 2.0
    assert MetricAggregator().aggregate(results)["comfort"]["num_scenarios"] == 1.0
    assert series.as_dict()["values"] == [1.0, 3.0]


def test_metric_callback_and_family_builder_are_history_agnostic(tmp_path):
    callback = MetricCallback([MappingMetricBuilder("safety", ("score",))], scenario_token="x")
    callback.on_simulation_end({"score": 0.75})
    assert callback.results[0].values == {"score": 0.75}
    assert callback.results[0].scenario_token == "x"

    left = DistributedExecutor(DistributedRunConfig("run", rank=0, world_size=2, backend="serial"))
    right = DistributedExecutor(DistributedRunConfig("run", rank=1, world_size=2, backend="serial"))
    paths = [tmp_path / "rank0.json", tmp_path / "rank1.json"]
    left.run(
        [WorkerTask(str(index), _square, args=(index,)) for index in range(4)],
        manifest_path=paths[0],
    )
    right.run(
        [WorkerTask(str(index), _square, args=(index,)) for index in range(4)],
        manifest_path=paths[1],
    )
    merged = merge_worker_manifests(paths, output_path=tmp_path / "merged.json")
    assert [result.task_id for result in merged] == ["0", "1", "2", "3"]
    assert (tmp_path / "merged.json").is_file()


class _Scenario:
    def __init__(self, token: str, scenario_type: str, log_name: str, map_name: str):
        self.token = token
        self.scenario_type = scenario_type
        self.log_name = log_name
        self.map_name = map_name
        self.scene = type("Scene", (), {"scene_metadata": type("Metadata", (), {"scene_tags": ()})()})()


def test_scenario_filter_and_type_balanced_sampling():
    scenarios = [
        _Scenario("a", "left", "log0", "map0"),
        _Scenario("b", "right", "log0", "map0"),
        _Scenario("c", "left", "log1", "map1"),
    ]
    selected = filter_scenarios(
        scenarios,
        ScenarioFilter(scenario_types=("left",), map_names=("map0",)),
    )
    assert [scenario.token for scenario in selected] == ["a"]
    balanced = sample_scenarios(scenarios, ScenarioSampling(num_samples=2, strategy="type_balanced"))
    assert [scenario.token for scenario in balanced] == ["a", "b"]


class _ScenarioForRuntime:
    token = "runtime@0"
    initial_ego_state = 0

    def get_route_roadblock_ids(self):
        return ()

    def get_mission_goal(self):
        return None

    def get_map_api(self):
        return None


class _Observation:
    def reset(self):
        return None

    def get_observation(self, iteration: SimulationIteration, history):
        return {"iteration": iteration.index, "history": len(history)}


class _Controller:
    def reset(self):
        return None

    def update_state(self, trajectory, iteration):
        del iteration
        return int(trajectory)


class _Planner:
    def initialize(self, initialization):
        self.initialization = initialization

    def name(self):
        return "planner"

    def compute_planner_trajectory(self, current_input: PlannerInput):
        return current_input.iteration.index + 1


def test_generic_simulation_runner_records_history_and_reports():
    setup = SimulationSetup(
        scenario=_ScenarioForRuntime(),
        planner=_Planner(),
        observation=_Observation(),
        ego_controller=_Controller(),
        time_controller=StepSimulationTimeController(0, 100_000, 3),
    )
    history = SimulationRunner().run(setup)
    assert isinstance(history, SimulationHistory)
    assert len(history) == 3
    assert [sample.ego_state for sample in history] == [1, 2, 3]
    assert history[0].planner_report is not None
    assert len(SimulationManager().run_many([setup])) == 1


def test_replay_clock_and_history_round_trip(tmp_path):
    from t4_e2e_devkit.planning.simulation.runtime import ReplaySimulationTimeController

    clock = ReplaySimulationTimeController([10, 20, 35])
    values = []
    while not clock.reached_end:
        values.append(clock.current_iteration.time_us)
        clock.advance()
    assert values == [10, 20, 35]

    history = SimulationHistory()
    history.extend(
        [
            # Portable values make this test independent of a domain serializer.
            type(history).from_dict(
                {
                    "samples": [
                        {
                            "iteration": {"index": 0, "time_us": 10},
                            "ego_state": {"x": 1},
                            "observation": {"frame": 0},
                            "trajectory": [1],
                            "planner_report": None,
                        }
                    ]
                }
            )[0]
        ]
    )
    path = tmp_path / "history.json"
    history.to_json(path)
    assert SimulationHistory.from_json(path).as_dict() == history.as_dict()


def _write_map_fixture(path):
    path.write_text(
        """<osm>
  <node id="1"><tag k="local_x" v="0"/><tag k="local_y" v="0"/></node>
  <node id="2"><tag k="local_x" v="0"/><tag k="local_y" v="10"/></node>
  <node id="3"><tag k="local_x" v="3"/><tag k="local_y" v="10"/></node>
  <node id="4"><tag k="local_x" v="3"/><tag k="local_y" v="0"/></node>
  <node id="5"><tag k="local_x" v="0"/><tag k="local_y" v="20"/></node>
  <node id="6"><tag k="local_x" v="3"/><tag k="local_y" v="20"/></node>
  <node id="50"><tag k="local_x" v="1.5"/><tag k="local_y" v="10"/><tag k="type" v="traffic_light"/></node>
  <way id="10"><nd ref="1"/><nd ref="2"/></way>
  <way id="11"><nd ref="4"/><nd ref="3"/></way>
  <way id="12"><nd ref="2"/><nd ref="5"/></way>
  <way id="13"><nd ref="3"/><nd ref="6"/></way>
  <way id="20"><nd ref="2"/><nd ref="3"/><tag k="subtype" v="stop_line"/></way>
  <way id="30"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="type" v="roadblock"/></way>
  <relation id="100"><member type="way" role="left" ref="10"/><member type="way" role="right" ref="11"/><tag k="type" v="lanelet"/></relation>
  <relation id="101"><member type="way" role="left" ref="12"/><member type="way" role="right" ref="13"/><tag k="type" v="lanelet"/><tag k="subtype" v="lane_connector"/><tag k="turn_direction" v="left"/></relation>
</osm>""",
        encoding="utf-8",
    )


def test_t4_map_adapter_exposes_vector_and_raster_queries(tmp_path):
    osm = tmp_path / "lanelet2_map.osm"
    _write_map_fixture(osm)
    adapter = T4MapAdapter(T4MapAPI(osm, map_name="fixture"), raster_precision=0.5)

    assert not inspect.isabstract(T4MapAdapter)
    lane = adapter.get_map_object("100", SemanticMapLayer.LANE)
    connector = adapter.get_map_object("101", SemanticMapLayer.LANE_CONNECTOR)
    assert isinstance(lane, Lane)
    assert isinstance(connector, LaneConnector)
    assert connector.turn_type.name == "LEFT"
    assert lane.outgoing_edges and lane.outgoing_edges[0].id == "101"
    assert adapter.get_one_map_object(Point2D(1.0, 5.0), SemanticMapLayer.LANE).id == "100"
    nearby = adapter.get_proximal_map_objects(
        Point2D(1.5, 10.0), 1.0, [SemanticMapLayer.TRAFFIC_LIGHT, SemanticMapLayer.STOP_LINE]
    )
    assert nearby[SemanticMapLayer.TRAFFIC_LIGHT][0].id == "50"
    assert nearby[SemanticMapLayer.STOP_LINE][0].id == "20"
    object_id, distance = adapter.get_distance_to_nearest_map_object(
        Point2D(1.0, 5.0), SemanticMapLayer.LANE
    )
    assert object_id == "100" and distance == 0.0
    raster = adapter.get_raster_map_layer(SemanticMapLayer.LANE)
    assert raster.data.ndim == 2 and raster.data.dtype == np.uint8
    assert raster.transform.shape == (4, 4)
