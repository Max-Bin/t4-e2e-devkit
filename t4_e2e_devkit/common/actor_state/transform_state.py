# =============================================================================
# VENDORED - do not edit by hand.
#
# Source : nuplan:nuplan/common/actor_state/transform_state.py
# Commit : e924167
# Tool   : tools/vendor.py
#
# Re-run ``python tools/vendor.py sync`` to update this file, and
# ``python tools/vendor.py check`` to detect drift against its source.
#
# Only ``import`` statements were rewritten; every numeric expression is
# byte-identical to the source. Edits belong upstream, or in a devkit module
# that wraps this one.
# =============================================================================

from t4_e2e_devkit.common.actor_state.state_representation import Point2D, StateSE2
from t4_e2e_devkit.common.geometry.transform import translate_longitudinally_and_laterally


def get_front_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, half_width).point


def get_front_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, -half_width).point


def get_rear_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, half_width).point


def get_rear_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, -half_width).point
