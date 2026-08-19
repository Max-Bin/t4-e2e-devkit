"""Rendering T4 windows: bird's-eye view, camera views, and score panels.

Frames are decoded through :mod:`t4_e2e_devkit.dataset.camera_source`, the same
decoder the data loader uses, so a rendered view shows the pixels a model saw.

Quick start::

    from t4_e2e_devkit.dataset.rigs import sensor_config_for_scene
    from t4_e2e_devkit.dataset.window import T4WindowBuilder
    from t4_e2e_devkit.visualization import plot_scene_summary, save_figure

    builder = T4WindowBuilder(scene_dir, root, sensor_config=sensor_config_for_scene(scene_dir))
    scene = builder.build(builder.valid_centers()[40])
    figure, _ = plot_scene_summary(scene, {"ground_truth": scene.get_future_trajectory()})
    save_figure(figure, "window.png")
"""

from t4_e2e_devkit.visualization.bev import (
    add_annotations_to_bev_ax,
    add_bev_status_text,
    add_box_to_bev_ax,
    add_ego_future_to_bev_ax,
    add_ego_to_bev_ax,
    add_future_annotations_to_bev_ax,
    add_goal_to_bev_ax,
    add_lidar_to_bev_ax,
    add_map_to_bev_ax,
    add_polylines_to_bev_ax,
    add_trajectory_to_bev_ax,
    boundaries_from_segments,
    centerlines_from_segments,
    traffic_light_states,
)
from t4_e2e_devkit.visualization.camera import (
    add_annotations_to_camera_ax,
    add_camera_ax,
    add_lidar_to_camera_ax,
    add_sync_comparison_to_camera_ax,
    add_trajectory_to_camera_ax,
    box_corners_3d,
    camera_grid_layout,
    project_ego_points,
    project_with_distortion,
)
from t4_e2e_devkit.visualization.lidar import (
    filter_lidar_pc,
    get_lidar_pc_color,
    prepare_lidar_pc,
    subsample_lidar_pc,
)
from t4_e2e_devkit.visualization.planning_video import (
    FFmpegVideoWriter,
    final_displacement_error,
    front_camera_name,
    manifest_trajectory,
    render_planning_frame,
    render_planning_video,
)
from t4_e2e_devkit.visualization.plots import (
    add_fixed_bev_legend,
    add_score_panel,
    configure_bev_ax,
    describe_scene,
    figure_to_rgb,
    frames_to_gif,
    plot_agent_comparison,
    plot_bev_frame,
    plot_bev_with_agent,
    plot_bev_with_score,
    plot_cameras_frame,
    plot_scene_summary,
    reference_trajectories,
    render_prediction_bev,
    save_figure,
)

from .dashboard import ResultsDashboard, write_results_dashboard
from .experiment_dashboard import ExperimentDashboard, write_experiment_dashboard

__all__ = [
    # bev
    "add_fixed_bev_legend",
    "add_annotations_to_bev_ax",
    "add_bev_status_text",
    "add_box_to_bev_ax",
    "add_ego_to_bev_ax",
    "add_ego_future_to_bev_ax",
    "add_future_annotations_to_bev_ax",
    "add_goal_to_bev_ax",
    "add_lidar_to_bev_ax",
    "add_map_to_bev_ax",
    "add_polylines_to_bev_ax",
    "add_trajectory_to_bev_ax",
    "boundaries_from_segments",
    "centerlines_from_segments",
    "traffic_light_states",
    # camera
    "add_annotations_to_camera_ax",
    "add_camera_ax",
    "add_lidar_to_camera_ax",
    "add_sync_comparison_to_camera_ax",
    "add_trajectory_to_camera_ax",
    "box_corners_3d",
    "camera_grid_layout",
    "project_ego_points",
    "project_with_distortion",
    # lidar
    "filter_lidar_pc",
    "get_lidar_pc_color",
    "prepare_lidar_pc",
    "subsample_lidar_pc",
    # planning video
    "FFmpegVideoWriter",
    "final_displacement_error",
    "front_camera_name",
    "manifest_trajectory",
    "render_planning_frame",
    "render_planning_video",
    # plots
    "add_score_panel",
    "configure_bev_ax",
    "describe_scene",
    "figure_to_rgb",
    "frames_to_gif",
    "plot_agent_comparison",
    "plot_bev_frame",
    "plot_bev_with_agent",
    "plot_bev_with_score",
    "plot_cameras_frame",
    "plot_scene_summary",
    "render_prediction_bev",
    "reference_trajectories",
    "save_figure",
    "ResultsDashboard",
    "write_results_dashboard",
    "ExperimentDashboard",
    "write_experiment_dashboard",
]
