"""t4-e2e-devkit: one dataset, agent and evaluation interface for T4 planning models.

The three things almost every user needs::

    from t4_e2e_devkit import AbstractT4Agent, T4Dataset, T4PDMScorer

Imports here are lazy.  A data-list build should not pay for CUDA
initialization, and a config-only dry run should not import the scoring stack --
so the heavy submodules are resolved on first attribute access rather than at
package import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_LAZY: dict[str, tuple[str, str]] = {
    # interface
    "AbstractT4Agent": ("t4_e2e_devkit.agents.abstract_agent", "AbstractT4Agent"),
    "build_agent": ("t4_e2e_devkit.agents.registry", "build_agent"),
    "register_agent": ("t4_e2e_devkit.agents.registry", "register_agent"),
    "available_agents": ("t4_e2e_devkit.agents.registry", "available_agents"),
    "ComponentRegistry": ("t4_e2e_devkit.config.registry", "ComponentRegistry"),
    "resolve_component": ("t4_e2e_devkit.config.registry", "resolve_component"),
    "DatasetConfig": ("t4_e2e_devkit.config.schema", "DatasetConfig"),
    "EvaluationConfig": ("t4_e2e_devkit.config.schema", "EvaluationConfig"),
    "ExperimentConfig": ("t4_e2e_devkit.config.schema", "ExperimentConfig"),
    "SimulationConfig": ("t4_e2e_devkit.config.schema", "SimulationConfig"),
    "WorkerConfig": ("t4_e2e_devkit.config.schema", "WorkerConfig"),
    "OutputConfig": ("t4_e2e_devkit.config.schema", "OutputConfig"),
    "load_experiment_config": ("t4_e2e_devkit.config.schema", "load_experiment_config"),
    # data
    "T4Dataset": ("t4_e2e_devkit.dataset.dataset", "T4Dataset"),
    "T4WindowBuilder": ("t4_e2e_devkit.dataset.window", "T4WindowBuilder"),
    "load_data_list": ("t4_e2e_devkit.dataset.datalist", "load_data_list"),
    "T4SceneTag": ("t4_e2e_devkit.dataset.scene_tags", "T4SceneTag"),
    "T4SceneTagIndex": ("t4_e2e_devkit.dataset.scene_tags", "T4SceneTagIndex"),
    "T4RouteMetadata": ("t4_e2e_devkit.dataset.route", "T4RouteMetadata"),
    "T4RoutePrimitive": ("t4_e2e_devkit.dataset.route", "T4RoutePrimitive"),
    "T4RouteSegment": ("t4_e2e_devkit.dataset.route", "T4RouteSegment"),
    "load_t4_route": ("t4_e2e_devkit.dataset.route", "load_t4_route"),
    "ScenarioFilter": ("t4_e2e_devkit.dataset.scenario_filter", "ScenarioFilter"),
    "ScenarioSampling": ("t4_e2e_devkit.dataset.scenario_filter", "ScenarioSampling"),
    "filter_scenarios": ("t4_e2e_devkit.dataset.scenario_filter", "filter_scenarios"),
    "sample_scenarios": ("t4_e2e_devkit.dataset.scenario_filter", "sample_scenarios"),
    "select_scenarios": ("t4_e2e_devkit.dataset.scenario_filter", "select_scenarios"),
    "filter_scenarios_for_rank": (
        "t4_e2e_devkit.dataset.scenario_filter",
        "filter_scenarios_for_rank",
    ),
    "collate_t4": ("t4_e2e_devkit.dataset.dataset", "collate_t4"),
    "AbstractScenario": (
        "t4_e2e_devkit.planning.scenario_builder.abstract_scenario",
        "AbstractScenario",
    ),
    "T4Scenario": (
        "t4_e2e_devkit.planning.scenario_builder.abstract_scenario",
        "T4Scenario",
    ),
    "T4TrafficLightStatus": (
        "t4_e2e_devkit.planning.scenario_builder.abstract_scenario",
        "T4TrafficLightStatus",
    ),
    "T4ScenarioBuilder": (
        "t4_e2e_devkit.planning.scenario_builder.t4_scenario",
        "T4ScenarioBuilder",
    ),
    # types
    "SceneFilter": ("t4_e2e_devkit.common.dataclasses", "SceneFilter"),
    "SensorConfig": ("t4_e2e_devkit.common.dataclasses", "SensorConfig"),
    "MapObjectMatch": ("t4_e2e_devkit.common.dataclasses", "MapObjectMatch"),
    "MapObjectIds": ("t4_e2e_devkit.common.dataclasses", "MapObjectIds"),
    "T4Lanelet": ("t4_e2e_devkit.common.t4_map", "T4Lanelet"),
    "T4MapObject": ("t4_e2e_devkit.common.t4_map", "T4MapObject"),
    "T4MapAPI": ("t4_e2e_devkit.common.t4_map", "T4MapAPI"),
    "T4MapAdapter": ("t4_e2e_devkit.common.maps.t4_map_adapter", "T4MapAdapter"),
    "T4AgentInput": ("t4_e2e_devkit.common.dataclasses", "T4AgentInput"),
    "T4Scene": ("t4_e2e_devkit.common.dataclasses", "T4Scene"),
    "Trajectory": ("t4_e2e_devkit.common.dataclasses", "Trajectory"),
    "PDMResults": ("t4_e2e_devkit.common.dataclasses", "PDMResults"),
    # independent evaluation families
    "OpenLoopMetricConfig": (
        "t4_e2e_devkit.evaluation.open_loop",
        "OpenLoopMetricConfig",
    ),
    "OpenLoopMetrics": ("t4_e2e_devkit.evaluation.open_loop", "OpenLoopMetrics"),
    "compute_open_loop_metrics": (
        "t4_e2e_devkit.evaluation.open_loop",
        "compute_open_loop_metrics",
    ),
    "ClosedLoopMetricConfig": (
        "t4_e2e_devkit.evaluation.closed_loop",
        "ClosedLoopMetricConfig",
    ),
    "ClosedLoopMetrics": (
        "t4_e2e_devkit.evaluation.closed_loop",
        "ClosedLoopMetrics",
    ),
    "ClosedLoopTrace": (
        "t4_e2e_devkit.evaluation.closed_loop",
        "ClosedLoopTrace",
    ),
    "compute_closed_loop_metrics": (
        "t4_e2e_devkit.evaluation.closed_loop",
        "compute_closed_loop_metrics",
    ),
    # closed-loop simulation
    "KinematicState": (
        "t4_e2e_devkit.planning.simulation.closed_loop",
        "KinematicState",
    ),
    "PerfectTracker": (
        "t4_e2e_devkit.planning.simulation.closed_loop",
        "PerfectTracker",
    ),
    "T4ClosedLoopConfig": (
        "t4_e2e_devkit.planning.simulation.closed_loop",
        "T4ClosedLoopConfig",
    ),
    "T4ClosedLoopResult": (
        "t4_e2e_devkit.planning.simulation.closed_loop",
        "T4ClosedLoopResult",
    ),
    "T4ClosedLoopRunner": (
        "t4_e2e_devkit.planning.simulation.closed_loop",
        "T4ClosedLoopRunner",
    ),
    "run_t4_closed_loop": (
        "t4_e2e_devkit.planning.simulation.closed_loop",
        "run_t4_closed_loop",
    ),
    "EgoController": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "EgoController",
    ),
    "ObservationProvider": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "ObservationProvider",
    ),
    "TrafficPolicy": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "TrafficPolicy",
    ),
    "CallableObservationProvider": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "CallableObservationProvider",
    ),
    "CallableTrafficPolicy": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "CallableTrafficPolicy",
    ),
    "CallableTrafficAgentController": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "CallableTrafficAgentController",
    ),
    "ReplayObservationProvider": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "ReplayObservationProvider",
    ),
    "ReplayTrafficPolicy": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "ReplayTrafficPolicy",
    ),
    "ConstantVelocityTrafficPolicy": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "ConstantVelocityTrafficPolicy",
    ),
    "ConstantVelocityTrafficAgentController": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "ConstantVelocityTrafficAgentController",
    ),
    "ReactiveTrafficPolicy": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "ReactiveTrafficPolicy",
    ),
    "IDMTrafficAgentController": (
        "t4_e2e_devkit.planning.simulation.multi_agent",
        "IDMTrafficAgentController",
    ),
    "MultiAgentTrafficPolicy": (
        "t4_e2e_devkit.planning.simulation.multi_agent",
        "MultiAgentTrafficPolicy",
    ),
    "TrafficPolicyConfig": (
        "t4_e2e_devkit.planning.simulation.multi_agent",
        "TrafficPolicyConfig",
    ),
    "TrafficAgentController": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "TrafficAgentController",
    ),
    "TrafficAgentState": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "TrafficAgentState",
    ),
    "SimulationCallback": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "SimulationCallback",
    ),
    "SimulationTick": (
        "t4_e2e_devkit.planning.simulation.interfaces",
        "SimulationTick",
    ),
    "SimulationRequest": (
        "t4_e2e_devkit.planning.simulation.manager",
        "SimulationRequest",
    ),
    "T4SimulationManager": (
        "t4_e2e_devkit.planning.simulation.manager",
        "T4SimulationManager",
    ),
    "T4AgentPlanner": (
        "t4_e2e_devkit.planning.simulation.planner.agent_planner",
        "T4AgentPlanner",
    ),
    "AbstractEgoController": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "AbstractEgoController",
    ),
    "AbstractObservation": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "AbstractObservation",
    ),
    "KinematicEgoController": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "KinematicEgoController",
    ),
    "LogPlaybackController": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "LogPlaybackController",
    ),
    "PlannerReport": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "PlannerReport",
    ),
    "SimulationRunReport": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationRunReport",
    ),
    "ReplaySimulationTimeController": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "ReplaySimulationTimeController",
    ),
    "SimulationHistory": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationHistory",
    ),
    "SimulationHistoryBuffer": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationHistoryBuffer",
    ),
    "SimulationHistorySample": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationHistorySample",
    ),
    "SimulationManager": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationManager",
    ),
    "SimulationRunner": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationRunner",
    ),
    "SimulationSetup": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationSetup",
    ),
    "SimulationTimeController": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "SimulationTimeController",
    ),
    "StepSimulationTimeController": (
        "t4_e2e_devkit.planning.simulation.runtime",
        "StepSimulationTimeController",
    ),
    "Simulation": (
        "t4_e2e_devkit.planning.simulation.simulation",
        "Simulation",
    ),
    "SimulationLog": (
        "t4_e2e_devkit.planning.simulation.simulation_log",
        "SimulationLog",
    ),
    "AbstractSimulationCallback": (
        "t4_e2e_devkit.planning.simulation.callback.abstract_callback",
        "AbstractCallback",
    ),
    "MultiCallback": (
        "t4_e2e_devkit.planning.simulation.callback.multi_callback",
        "MultiCallback",
    ),
    "TimingCallback": (
        "t4_e2e_devkit.planning.simulation.callback.timing_callback",
        "TimingCallback",
    ),
    "SimulationLogCallback": (
        "t4_e2e_devkit.planning.simulation.callback.serialization_callback",
        "SimulationLogCallback",
    ),
    "VisualizationCallback": (
        "t4_e2e_devkit.planning.simulation.callback.visualization_callback",
        "VisualizationCallback",
    ),
    "SimulationMetricCallback": (
        "t4_e2e_devkit.planning.simulation.callback.metric_callback",
        "MetricCallback",
    ),
    "SimulationController": (
        "t4_e2e_devkit.planning.simulation.controller.kinematic_bicycle",
        "KinematicBicycleController",
    ),
    "ControllerLogPlayback": (
        "t4_e2e_devkit.planning.simulation.controller.log_playback",
        "LogPlaybackController",
    ),
    "PerfectTrackingController": (
        "t4_e2e_devkit.planning.simulation.controller.perfect_tracking",
        "PerfectTrackingController",
    ),
    "TwoStageController": (
        "t4_e2e_devkit.planning.simulation.controller.two_stage_controller",
        "TwoStageController",
    ),
    "AbstractMotionModel": (
        "t4_e2e_devkit.planning.simulation.controller.motion_model",
        "AbstractMotionModel",
    ),
    "KinematicBicycleMotionModel": (
        "t4_e2e_devkit.planning.simulation.controller.motion_model",
        "KinematicBicycleMotionModel",
    ),
    "NuPlanSimulationRunner": (
        "t4_e2e_devkit.planning.simulation.runner.simulations_runner",
        "SimulationRunner",
    ),
    "SimulationsRunner": (
        "t4_e2e_devkit.planning.simulation.runner.simulations_runner",
        "SimulationsRunner",
    ),
    "RunnerReport": (
        "t4_e2e_devkit.planning.simulation.runner.runner_report",
        "RunnerReport",
    ),
    "RunnerExecutor": (
        "t4_e2e_devkit.planning.simulation.runner.executor",
        "RunnerExecutor",
    ),
    "AbstractMainCallback": (
        "t4_e2e_devkit.planning.simulation.main_callback.abstract_main_callback",
        "AbstractMainCallback",
    ),
    "MultiMainCallback": (
        "t4_e2e_devkit.planning.simulation.main_callback.multi_main_callback",
        "MultiMainCallback",
    ),
    "TimeCallback": (
        "t4_e2e_devkit.planning.simulation.main_callback.time_callback",
        "TimeCallback",
    ),
    "CompletionCallback": (
        "t4_e2e_devkit.planning.simulation.main_callback.completion_callback",
        "CompletionCallback",
    ),
    "MetricSummaryCallback": (
        "t4_e2e_devkit.planning.simulation.main_callback.metric_summary_callback",
        "MetricSummaryCallback",
    ),
    "PlannerReportStatistics": (
        "t4_e2e_devkit.planning.simulation.planner.planner_report",
        "PlannerReport",
    ),
    "LogFuturePlanner": (
        "t4_e2e_devkit.planning.simulation.planner.log_future_planner",
        "LogFuturePlanner",
    ),
    "SimplePlanner": (
        "t4_e2e_devkit.planning.simulation.planner.simple_planner",
        "SimplePlanner",
    ),
    "AbstractPredictor": (
        "t4_e2e_devkit.planning.simulation.predictor.abstract_predictor",
        "AbstractPredictor",
    ),
    "PredictorInitialization": (
        "t4_e2e_devkit.planning.simulation.predictor.abstract_predictor",
        "PredictorInitialization",
    ),
    "PredictorInput": (
        "t4_e2e_devkit.planning.simulation.predictor.abstract_predictor",
        "PredictorInput",
    ),
    "PredictorReport": (
        "t4_e2e_devkit.planning.simulation.predictor.predictor_report",
        "PredictorReport",
    ),
    "LogFuturePredictor": (
        "t4_e2e_devkit.planning.simulation.predictor.log_future_predictor",
        "LogFuturePredictor",
    ),
    "TracksObservation": (
        "t4_e2e_devkit.planning.simulation.observation.observation_type",
        "TracksObservation",
    ),
    "T4ReplayObservation": (
        "t4_e2e_devkit.planning.simulation.observation.replay",
        "T4ReplayObservation",
    ),
    "T4ReplayObservationSource": (
        "t4_e2e_devkit.planning.simulation.observation.replay",
        "T4ReplayObservationSource",
    ),
    "MetricCache": ("t4_e2e_devkit.evaluation.metric_cache", "MetricCache"),
    "MetricCatalog": ("t4_e2e_devkit.evaluation.metric_catalog", "MetricCatalog"),
    "MetricSpec": ("t4_e2e_devkit.evaluation.metric_catalog", "MetricSpec"),
    "evaluate_data_list": ("t4_e2e_devkit.script.evaluate", "evaluate_data_list"),
    "FileBackedBarrier": (
        "t4_e2e_devkit.evaluation.file_backed_barrier",
        "FileBackedBarrier",
    ),
    "MetricContext": ("t4_e2e_devkit.evaluation.metric_engine", "MetricContext"),
    "MetricDefinition": ("t4_e2e_devkit.evaluation.metric_engine", "MetricDefinition"),
    "MetricEngine": ("t4_e2e_devkit.evaluation.metric_engine", "MetricEngine"),
    "MetricRecord": ("t4_e2e_devkit.evaluation.metric_engine", "MetricRecord"),
    "MetricReport": ("t4_e2e_devkit.evaluation.metric_engine", "MetricReport"),
    "LeaderboardReport": ("t4_e2e_devkit.evaluation.leaderboard", "LeaderboardReport"),
    "LeaderboardRow": ("t4_e2e_devkit.evaluation.leaderboard", "LeaderboardRow"),
    "build_leaderboard": ("t4_e2e_devkit.evaluation.leaderboard", "build_leaderboard"),
    "AbstractMetricBuilder": ("t4_e2e_devkit.evaluation.metric_api", "AbstractMetricBuilder"),
    "CallableMetricBuilder": ("t4_e2e_devkit.evaluation.metric_api", "CallableMetricBuilder"),
    "MetricAggregator": ("t4_e2e_devkit.evaluation.metric_api", "MetricAggregator"),
    "MetricBuilderRegistry": ("t4_e2e_devkit.evaluation.metric_api", "MetricBuilderRegistry"),
    "MetricCallback": ("t4_e2e_devkit.evaluation.metric_api", "MetricCallback"),
    "MetricResult": ("t4_e2e_devkit.evaluation.metric_api", "MetricResult"),
    "MetricStatistic": ("t4_e2e_devkit.evaluation.metric_api", "MetricStatistic"),
    "MetricStatistics": ("t4_e2e_devkit.evaluation.metric_api", "MetricStatistics"),
    "MetricTimeSeries": ("t4_e2e_devkit.evaluation.metric_api", "MetricTimeSeries"),
    "MetricStatisticsType": (
        "t4_e2e_devkit.evaluation.metrics.metric_result",
        "MetricStatisticsType",
    ),
    "Statistic": ("t4_e2e_devkit.evaluation.metrics.metric_result", "Statistic"),
    "TimeSeries": ("t4_e2e_devkit.evaluation.metrics.metric_result", "TimeSeries"),
    "MetricViolation": (
        "t4_e2e_devkit.evaluation.metrics.metric_result",
        "MetricViolation",
    ),
    "TypedMetricStatistics": (
        "t4_e2e_devkit.evaluation.metrics.metric_result",
        "MetricStatistics",
    ),
    "AbstractMetric": (
        "t4_e2e_devkit.evaluation.metrics.abstract_metric",
        "AbstractMetric",
    ),
    "ViolationMetricBase": (
        "t4_e2e_devkit.evaluation.metrics.abstract_metric",
        "ViolationMetricBase",
    ),
    "WithinBoundMetricBase": (
        "t4_e2e_devkit.evaluation.metrics.abstract_metric",
        "WithinBoundMetricBase",
    ),
    "WeightedAverageMetricAggregator": (
        "t4_e2e_devkit.evaluation.metrics.weighted_average",
        "WeightedAverageMetricAggregator",
    ),
    "MetricFile": ("t4_e2e_devkit.evaluation.metrics.metric_file", "MetricFile"),
    "MetricFileKey": ("t4_e2e_devkit.evaluation.metrics.metric_file", "MetricFileKey"),
    "MetricStatisticsDataFrame": (
        "t4_e2e_devkit.evaluation.metrics.metric_dataframe",
        "MetricStatisticsDataFrame",
    ),
    "DrivableAreaMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "DrivableAreaMetric",
    ),
    "LaneDepartureMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "LaneDepartureMetric",
    ),
    "TrafficLightMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "TrafficLightMetric",
    ),
    "GoalReachedMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "GoalReachedMetric",
    ),
    "AccelerationMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "AccelerationMetric",
    ),
    "JerkMetric": ("t4_e2e_devkit.evaluation.metrics.standard", "JerkMetric"),
    "YawRateMetric": ("t4_e2e_devkit.evaluation.metrics.standard", "YawRateMetric"),
    "StandardComfortMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "ComfortMetric",
    ),
    "StandardProgressMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "ProgressMetric",
    ),
    "StandardCollisionMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "CollisionMetric",
    ),
    "StandardTTCMetric": ("t4_e2e_devkit.evaluation.metrics.standard", "TTCMetric"),
    "StandardSpeedLimitMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "SpeedLimitMetric",
    ),
    "StandardStopLineMetric": (
        "t4_e2e_devkit.evaluation.metrics.standard",
        "StopLineViolationMetric",
    ),
    "MappingMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "MappingMetricBuilder",
    ),
    "OpenLoopMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "OpenLoopMetricBuilder",
    ),
    "ClosedLoopMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "ClosedLoopMetricBuilder",
    ),
    "PDMMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "PDMMetricBuilder",
    ),
    "T4SafetyMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "T4SafetyMetricBuilder",
    ),
    "ComfortMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "ComfortMetricBuilder",
    ),
    "ProgressMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "ProgressMetricBuilder",
    ),
    "CollisionMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "CollisionMetricBuilder",
    ),
    "DrivableAreaMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "DrivableAreaMetricBuilder",
    ),
    "TrafficLightMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "TrafficLightMetricBuilder",
    ),
    "StopLineViolationMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "StopLineViolationMetricBuilder",
    ),
    "TTCMetricBuilder": (
        "t4_e2e_devkit.evaluation.metric_builders",
        "TTCMetricBuilder",
    ),
    "WorkerPool": ("t4_e2e_devkit.evaluation.worker_pool", "WorkerPool"),
    "WorkerResources": ("t4_e2e_devkit.evaluation.worker_pool", "WorkerResources"),
    "WorkerResult": ("t4_e2e_devkit.evaluation.worker_pool", "WorkerResult"),
    "WorkerTask": ("t4_e2e_devkit.evaluation.worker_pool", "WorkerTask"),
    "DistributedExecutor": (
        "t4_e2e_devkit.evaluation.distributed",
        "DistributedExecutor",
    ),
    "LocalDistributedLauncher": (
        "t4_e2e_devkit.evaluation.orchestration",
        "LocalDistributedLauncher",
    ),
    "DistributedLaunchResult": (
        "t4_e2e_devkit.evaluation.orchestration",
        "DistributedLaunchResult",
    ),
    "RankLaunch": ("t4_e2e_devkit.evaluation.orchestration", "RankLaunch"),
    "DistributedRunConfig": (
        "t4_e2e_devkit.evaluation.distributed",
        "DistributedRunConfig",
    ),
    "WorkerManifest": (
        "t4_e2e_devkit.evaluation.distributed",
        "WorkerManifest",
    ),
    "merge_worker_manifests": (
        "t4_e2e_devkit.evaluation.distributed",
        "merge_worker_manifests",
    ),
    "merge_worker_results": (
        "t4_e2e_devkit.evaluation.worker_pool",
        "merge_worker_results",
    ),
    "TrajectorySubmission": (
        "t4_e2e_devkit.evaluation.submission",
        "TrajectorySubmission",
    ),
    "SubmissionPackage": (
        "t4_e2e_devkit.evaluation.submission",
        "SubmissionPackage",
    ),
    "SubmissionValidation": (
        "t4_e2e_devkit.evaluation.submission",
        "SubmissionValidation",
    ),
    "RewardConfig": ("t4_e2e_devkit.evaluation.tier4_metrics", "RewardConfig"),
    "FeatureCache": (
        "t4_e2e_devkit.planning.training.feature_cache",
        "FeatureCache",
    ),
    "FeatureBuilderRegistry": ("t4_e2e_devkit.agents.builders", "FeatureBuilderRegistry"),
    "TargetBuilderRegistry": ("t4_e2e_devkit.agents.builders", "TargetBuilderRegistry"),
    "LocalExecutor": ("t4_e2e_devkit.evaluation.executor", "LocalExecutor"),
    "rank_indices": ("t4_e2e_devkit.evaluation.executor", "rank_indices"),
    # evaluation
    "T4PDMScorer": ("t4_e2e_devkit.evaluation.pdm_score", "T4PDMScorer"),
    "T4PDMReferenceProvider": (
        "t4_e2e_devkit.evaluation.reference_provider",
        "T4PDMReferenceProvider",
    ),
    "aggregate_evaluation": (
        "t4_e2e_devkit.evaluation.report",
        "aggregate_evaluation",
    ),
    "aggregate_pdm_results": (
        "t4_e2e_devkit.common.dataclasses",
        "aggregate_pdm_results",
    ),
    "aggregate_results": ("t4_e2e_devkit.common.dataclasses", "aggregate_results"),
    # visualization/training integration
    "render_prediction_bev": (
        "t4_e2e_devkit.visualization.plots",
        "render_prediction_bev",
    ),
    "ResultsDashboard": (
        "t4_e2e_devkit.visualization.dashboard",
        "ResultsDashboard",
    ),
    "write_results_dashboard": (
        "t4_e2e_devkit.visualization.dashboard",
        "write_results_dashboard",
    ),
    "ExperimentDashboard": (
        "t4_e2e_devkit.visualization.experiment_dashboard",
        "ExperimentDashboard",
    ),
    "write_experiment_dashboard": (
        "t4_e2e_devkit.visualization.experiment_dashboard",
        "write_experiment_dashboard",
    ),
    "PredictionVizCallback": (
        "t4_e2e_devkit.planning.training.callbacks",
        "PredictionVizCallback",
    ),
}

__all__ = ["__version__", *sorted(_LAZY)]

if TYPE_CHECKING:  # let type checkers and IDEs see the real symbols
    from t4_e2e_devkit.agents.abstract_agent import AbstractT4Agent
    from t4_e2e_devkit.agents.builders import FeatureBuilderRegistry, TargetBuilderRegistry
    from t4_e2e_devkit.agents.registry import available_agents, build_agent, register_agent
    from t4_e2e_devkit.common.dataclasses import (
        MapObjectIds,
        MapObjectMatch,
        PDMResults,
        SceneFilter,
        SensorConfig,
        T4AgentInput,
        T4Scene,
        Trajectory,
        aggregate_pdm_results,
        aggregate_results,
    )
    from t4_e2e_devkit.common.maps.t4_map_adapter import T4MapAdapter
    from t4_e2e_devkit.common.t4_map import T4Lanelet, T4MapAPI, T4MapObject
    from t4_e2e_devkit.dataset.datalist import load_data_list
    from t4_e2e_devkit.dataset.dataset import T4Dataset, collate_t4
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
    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.evaluation.closed_loop import (
        ClosedLoopMetricConfig,
        ClosedLoopMetrics,
        ClosedLoopTrace,
        compute_closed_loop_metrics,
    )
    from t4_e2e_devkit.evaluation.distributed import (
        DistributedExecutor,
        DistributedRunConfig,
        WorkerManifest,
        merge_worker_manifests,
    )
    from t4_e2e_devkit.evaluation.executor import LocalExecutor, rank_indices
    from t4_e2e_devkit.evaluation.metric_api import (
        AbstractMetricBuilder,
        CallableMetricBuilder,
        MetricAggregator,
        MetricBuilderRegistry,
        MetricCallback,
        MetricResult,
        MetricStatistic,
        MetricStatistics,
        MetricTimeSeries,
    )
    from t4_e2e_devkit.evaluation.metric_builders import (
        ClosedLoopMetricBuilder,
        CollisionMetricBuilder,
        ComfortMetricBuilder,
        DrivableAreaMetricBuilder,
        MappingMetricBuilder,
        OpenLoopMetricBuilder,
        PDMMetricBuilder,
        ProgressMetricBuilder,
        StopLineViolationMetricBuilder,
        T4SafetyMetricBuilder,
        TrafficLightMetricBuilder,
        TTCMetricBuilder,
    )
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
        compute_open_loop_metrics,
    )
    from t4_e2e_devkit.evaluation.pdm_score import T4PDMScorer
    from t4_e2e_devkit.evaluation.reference_provider import T4PDMReferenceProvider
    from t4_e2e_devkit.evaluation.report import aggregate_evaluation
    from t4_e2e_devkit.evaluation.tier4_metrics import RewardConfig
    from t4_e2e_devkit.evaluation.worker_pool import (
        WorkerPool,
        WorkerResources,
        WorkerResult,
        WorkerTask,
        merge_worker_results,
    )
    from t4_e2e_devkit.planning.scenario_builder.abstract_scenario import (
        AbstractScenario,
        T4Scenario,
        T4TrafficLightStatus,
    )
    from t4_e2e_devkit.planning.scenario_builder.t4_scenario import T4ScenarioBuilder
    from t4_e2e_devkit.planning.simulation.closed_loop import (
        KinematicState,
        PerfectTracker,
        T4ClosedLoopConfig,
        T4ClosedLoopResult,
        T4ClosedLoopRunner,
        run_t4_closed_loop,
    )
    from t4_e2e_devkit.planning.simulation.interfaces import (
        CallableObservationProvider,
        CallableTrafficAgentController,
        CallableTrafficPolicy,
        ConstantVelocityTrafficAgentController,
        ConstantVelocityTrafficPolicy,
        EgoController,
        ObservationProvider,
        ReactiveTrafficPolicy,
        ReplayObservationProvider,
        ReplayTrafficPolicy,
        SimulationCallback,
        SimulationTick,
        TrafficAgentController,
        TrafficAgentState,
        TrafficPolicy,
    )
    from t4_e2e_devkit.planning.simulation.manager import SimulationRequest, T4SimulationManager
    from t4_e2e_devkit.planning.simulation.observation.observation_type import TracksObservation
    from t4_e2e_devkit.planning.simulation.observation.replay import (
        T4ReplayObservation,
        T4ReplayObservationSource,
    )
    from t4_e2e_devkit.planning.simulation.planner.agent_planner import T4AgentPlanner
    from t4_e2e_devkit.planning.simulation.runtime import (
        AbstractEgoController,
        AbstractObservation,
        KinematicEgoController,
        LogPlaybackController,
        PlannerReport,
        ReplaySimulationTimeController,
        SimulationHistory,
        SimulationHistoryBuffer,
        SimulationHistorySample,
        SimulationManager,
        SimulationRunner,
        SimulationRunReport,
        SimulationSetup,
        SimulationTimeController,
        StepSimulationTimeController,
    )
    from t4_e2e_devkit.planning.training.callbacks import PredictionVizCallback
    from t4_e2e_devkit.planning.training.feature_cache import FeatureCache
    from t4_e2e_devkit.visualization.plots import render_prediction_bev


def __getattr__(name: str) -> Any:
    """Resolve a lazily exported symbol.

    :param name: attribute name.
    :return: the symbol.
    :raises AttributeError: for a name this package does not export.
    """
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value  # cache, so the import happens once
    return value


def __dir__() -> list[str]:
    return list(__all__)
