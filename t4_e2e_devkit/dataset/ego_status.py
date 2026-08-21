"""Ego pose and status in the window's own frame.

The recorded trajectory is global; a model reads the window's centre frame.
These three functions are that conversion and nothing else -- a pose into centre
coordinates, an :class:`EgoStatus` for one frame, and the agent boxes moved with
it -- so they sit apart from the readers that produce the arrays.

Split out of :mod:`t4_e2e_devkit.dataset.scene`.
"""

from __future__ import annotations

import numpy as np


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def global_to_ego(poses: np.ndarray, center_pose: np.ndarray) -> np.ndarray:
    """Convert global T4 poses ``[x, y, cos, sin]`` to local ``[x, y, yaw]``.

    ``center_pose`` is the current ego pose.  The calculation is float64 until
    the caller casts to float32, matching the source sample construction and
    avoiding precision loss from large global coordinates.
    """

    poses = np.asarray(poses, dtype=np.float64)
    center_pose = np.asarray(center_pose, dtype=np.float64)
    heading = float(np.arctan2(center_pose[3], center_pose[2]))
    c, s = np.cos(-heading), np.sin(-heading)
    dx = poses[..., 0] - center_pose[0]
    dy = poses[..., 1] - center_pose[1]
    return np.stack(
        (
            dx * c - dy * s,
            dx * s + dy * c,
            _wrap_angle(np.arctan2(poses[..., 3], poses[..., 2]) - heading),
        ),
        axis=-1,
    )


def build_ego_status(global_poses: np.ndarray, center_pose: np.ndarray, dt: float) -> np.ndarray:
    """Build the ``[N, 7]`` ego-status history from a pose history alone.

    ``global_poses`` is ``[N, 4]`` global ``[x, y, cos, sin]`` ordered oldest to
    newest and ``center_pose`` is the frame everything is expressed in.  The
    columns are local ``[x, y, yaw]`` followed by velocity ``[vx, vy]`` and
    acceleration ``[ax, ay]``, all in the centre frame and all differenced from
    those same poses.

    ``scalars["velocity"]`` and ``scalars["acceleration"]`` are deliberately not
    read here.  They come from ``/localization/kinematic_state`` and
    ``/localization/acceleration``, which carry three properties that make them
    unusable as a per-frame history feature: ``ekf_localizer``'s stop filter
    rewrites any speed below 0.1 m/s to exactly zero (33.6% of ``prd_jt``
    frames), the lateral components are never populated (``vy`` and ``ay`` are
    identically zero across the dataset), and the twist rides the CAN/IMU chain,
    which trails the NDT-driven pose by roughly 130 ms -- so a row would pair a
    pose with a velocity describing the state 1.3 frames earlier.  Those signals
    are also per-frame body-frame while the pose columns are centre-frame, so
    the two halves of a row disagreed about their axes during a turn.
    Differencing the poses puts every column in one frame, on one clock, and
    gives the lateral channels real values.

    The oldest rows have no predecessor to difference against.  They repeat the
    nearest well-defined row rather than reading zero, which would claim the ego
    was stationary at the start of the history window.
    """

    local = global_to_ego(global_poses, center_pose)
    n = local.shape[0]
    velocity = np.zeros((n, 2), dtype=np.float64)
    acceleration = np.zeros((n, 2), dtype=np.float64)
    if n > 1:
        velocity[1:] = np.diff(local[:, :2], axis=0) / dt
        velocity[0] = velocity[1]
    if n > 2:
        # Differencing velocity[1:] keeps the fabricated velocity[0] out of it.
        acceleration[2:] = np.diff(velocity[1:], axis=0) / dt
        acceleration[:2] = acceleration[2]
    return np.concatenate((local, velocity, acceleration), axis=-1).astype(np.float32)


def _transform_agent_boxes_to_center(
    boxes: np.ndarray,
    frame_pose: np.ndarray,
    center_pose: np.ndarray,
) -> np.ndarray:
    """Transform T4 ``[x,y,z,w,l,h,yaw,vx,vy]`` boxes into center-ego.

    T4 stores each frame's agent annotations in that frame's ego
    coordinate system.  The oracle compares all 40 future frames against one
    center-frame prediction, so both the box pose and its velocity must be
    rotated through ``heading(frame)-heading(center)``.  The box dimensions and
    height are unchanged.
    """

    values = np.asarray(boxes, dtype=np.float64).reshape(-1, 9)
    if values.shape[0] == 0:
        return np.zeros((0, 9), dtype=np.float32)
    frame_pose = np.asarray(frame_pose, dtype=np.float64)
    center_pose = np.asarray(center_pose, dtype=np.float64)
    frame_heading = float(np.arctan2(frame_pose[3], frame_pose[2]))
    center_heading = float(np.arctan2(center_pose[3], center_pose[2]))
    relative_heading = frame_heading - center_heading
    c_rel, s_rel = np.cos(relative_heading), np.sin(relative_heading)
    origin = global_to_ego(frame_pose[None], center_pose)[0, :2]

    out = values.copy()
    x_local, y_local = values[:, 0].copy(), values[:, 1].copy()
    out[:, 0] = origin[0] + x_local * c_rel - y_local * s_rel
    out[:, 1] = origin[1] + x_local * s_rel + y_local * c_rel
    # Match the source window extraction contract: it adds the
    # relative heading without wrapping.  The downstream box geometry consumes
    # sin/cos-equivalent angles, but preserving the raw representation matters
    # for byte-level oracle-label parity and avoids making this adapter's GT
    # contract subtly different from the reference loader.
    out[:, 6] = values[:, 6] + relative_heading
    vx_local, vy_local = values[:, 7].copy(), values[:, 8].copy()
    out[:, 7] = vx_local * c_rel - vy_local * s_rel
    out[:, 8] = vx_local * s_rel + vy_local * c_rel
    return out.astype(np.float32)
