"""Colours and rendering configuration.

Map colours are keyed by T4 tensor fields and object colours by T4 class id.
Trajectory roles use separate colours and line styles so map, GT and plans do
not collapse into one visual layer.
"""

from __future__ import annotations

from typing import Any, Dict

from t4_e2e_devkit.common.enums import T4TrackLabel

# --------------------------------------------------------------------------- #
# Base palette
# --------------------------------------------------------------------------- #

WHITE: str = "#FFFFFF"
BLACK: str = "#000000"
LIGHT_GREY: str = "#D3D3D3"
MID_GREY: str = "#95A5A6"
DARK_GREY: str = "#4A4A4A"

# Tableau-10, for anything that needs an arbitrary categorical colour.
TAB_10: Dict[int, str] = {
    0: "#4e79a7",  # blue
    1: "#f28e2b",  # orange
    2: "#e15759",  # red
    3: "#76b7b2",  # cyan
    4: "#59a14f",  # green
    5: "#edc948",  # yellow
    6: "#b07aa1",  # violet
    7: "#ff9da7",  # pink
    8: "#9c755f",  # brown
    9: "#bab0ac",  # grey
}

# --------------------------------------------------------------------------- #
# Semantic roles
# --------------------------------------------------------------------------- #

EGO_COLOR: str = "red"
# Trajectory roles use colours that are not used by the neutral map layers.
# Ground truth uses a time gradient rather than this fixed colour.
PREDICTION_COLOR: str = "#007C91"
GROUND_TRUTH_COLOR: str = "#C2185B"
HISTORY_COLOR: str = "#6B7280"
GOAL_COLOR: str = "blue"

# BEV groups road users by broad visual category, while camera overlays retain
# the complete T4 class palette.
BEV_AGENT_COLORS: Dict[int, str] = {
    int(T4TrackLabel.CAR): "blue",
    int(T4TrackLabel.TRUCK): "blue",
    int(T4TrackLabel.BUS): "blue",
    int(T4TrackLabel.PEDESTRIAN): "green",
    int(T4TrackLabel.BICYCLE): "purple",
}

# --------------------------------------------------------------------------- #
# Tracked objects, by T4 class id
# --------------------------------------------------------------------------- #

TRACK_COLORS: Dict[int, str] = {
    int(T4TrackLabel.CAR): "#4e79a7",  # blue
    int(T4TrackLabel.TRUCK): "#2E5E8A",  # darker blue -- still a vehicle
    int(T4TrackLabel.BUS): "#1B3F5C",  # darkest blue
    int(T4TrackLabel.BICYCLE): "#f28e2b",  # orange
    int(T4TrackLabel.PEDESTRIAN): "#e15759",  # red
}

#: Colour for a class id the palette does not know. Deliberately garish: an
#: unmapped class means the converter's vocabulary changed, and that should be
#: visible in a plot rather than blending in.
UNKNOWN_TRACK_COLOR: str = "#FF00FF"

TRACK_CONFIG: Dict[str, Any] = {
    "fill_alpha": 0.45,
    "line_width": 1.2,
    "edge_color": BLACK,
    "zorder": 3,
    # Draw a short heading tick so a stationary box still shows its orientation.
    "heading_tick": True,
    "heading_tick_length": 0.6,
}

# --------------------------------------------------------------------------- #
# Map layers, by tensor field
# --------------------------------------------------------------------------- #

# Visual hierarchy is deliberate. For a planning devkit the lane network and the
# route are the subject, so lane centrelines are drawn stronger than the road
# borders -- the reverse of what looks natural on a map, and the reverse of what
# an untuned palette produces, where dark border polylines dominate and the lanes
# disappear into the background.
MAP_CONFIG: Dict[str, Dict[str, Any]] = {
    "lanes": {
        "color": "grey",
        "alpha": 0.25,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    "lane_boundaries": {
        "color": "grey",
        "alpha": 0.25,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    "route_lanes": {
        "color": "olive",
        "alpha": 0.5,
        "line_width": 2.0,
        "line_style": "--",
        "zorder": 3,
    },
    "polygons": {
        "color": "gray",
        "alpha": 0.5,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 0,
    },
    "line_strings": {
        "color": "#B8A36A",
        "alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    "road_borders": {
        "color": "#68727C",
        "alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
}

#: Traffic-light state -> colour, for route lanes. ``white`` is the converter's
#: "unresolved" state and is drawn in the neutral route colour rather than as a
#: signal, because rendering an unknown as green or red would be a claim the data
#: does not make.
TRAFFIC_LIGHT_COLORS: Dict[str, str] = {
    "green": "green",
    "yellow": "yellow",
    "red": "red",
    "white": "purple",
    "none": "black",
}

# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #

TRAJECTORY_CONFIG: Dict[str, Dict[str, Any]] = {
    "prediction": {
        "color": PREDICTION_COLOR,
        "alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": None,
        "marker_size": 0,
        "zorder": 5,
        "label": "prediction",
    },
    "ground_truth": {
        "color": GROUND_TRUTH_COLOR,
        "alpha": 0.5,
        "line_width": 1.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 20,
        "zorder": 5,
        "label": "GT",
    },
    "history": {
        "color": HISTORY_COLOR,
        "alpha": 0.5,
        "line_width": 1.5,
        "line_style": "--",
        "marker": None,
        "marker_size": 0,
        "zorder": 4,
        # Keep the public kind label stable for callers/tests; the reference
        # renderer's human-readable label is only visible when a legend is
        # explicitly enabled.
        "label": "history",
    },
}

# --------------------------------------------------------------------------- #
# LiDAR
# --------------------------------------------------------------------------- #

LIDAR_CONFIG: Dict[str, Any] = {
    # T4 points are [x, y, z, intensity, ring_or_time] -- five columns, no id.
    "color_element": "z",  # none | distance | x | y | z | intensity | ring
    "color_map": "viridis",
    "x_lim": [-64.0, 64.0],
    "y_lim": [-64.0, 64.0],
    "z_lim": [-3.0, 5.0],
    "point_size": 0.35,
    "alpha": 0.55,
    "zorder": 2,
    # A full T4 sweep is ~440k points. Rendering all of them makes a 3 MB PNG
    # that looks identical to a subsampled one, so plots subsample by default and
    # say so rather than silently thinning.
    "max_points": 120_000,
}

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

BEV_PLOT_CONFIG: Dict[str, Any] = {
    "figure_size": (8.0, 8.0),
    "dpi": 100,
    "background_color": WHITE,
    # Half-extent of the window, in metres.
    #
    # 40 m rather than 60: the scorer horizon is 4 s, which is under 40 m at urban
    # speeds, and a wider window shrinks a 0.6 m pedestrian box below one pixel --
    # so the agents nearest the ego, the ones that decide the collision terms,
    # become invisible exactly when they matter most.
    "view_range": 60.0,
    "layers": ["polygons", "line_strings", "lanes", "route_lanes", "annotations"],
    "grid": True,
    "grid_alpha": 0.3,
    # Keep the reference default uncluttered; documentation samples can enable
    # the semantic legend explicitly.
    "legend": False,
    "legend_loc": "upper right",
    "axis_labels": True,
    "status_text": True,
    "show_history": True,
    "show_neighbor_future": True,
    "show_ego_future_footprints": True,
}

#: Camera grid layout for the default five-view T4 rig.
#:
#: T4 has no centred ``CAM_BACK``, so the rear is covered by two wide views that
#: bracket that direction. Laying them out as a 2x3 with the rear pair on the
#: bottom row keeps left-right adjacency spatially truthful, which a flat strip
#: of five images does not.
CAMERA_GRID_LAYOUTS: Dict[int, list] = {
    5: [
        ["CAM_FRONT_LEFT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT_WIDE"],
        ["CAM_BACK_LEFT_WIDE", None, "CAM_BACK_RIGHT_WIDE"],
    ],
    3: [["CAM_FRONT_LEFT_WIDE", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT_WIDE"]],
    1: [["CAM_FRONT_WIDE"]],
}

CAMERAS_PLOT_CONFIG: Dict[str, Any] = {
    "figure_size": (15.0, 7.0),
    "dpi": 100,
    "background_color": WHITE,
    "title_size": 9,
}

#: How PDM components are rendered in a score overlay.
SCORE_PANEL_CONFIG: Dict[str, Any] = {
    "gate_color": "#E74C3C",  # NC and DAC: multiplicative, so a zero is fatal
    "weighted_color": "#3498DB",
    "zero_color": "#E74C3C",
    "text_size": 9,
    "bar_height": 0.6,
}
