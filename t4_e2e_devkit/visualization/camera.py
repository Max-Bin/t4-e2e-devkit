"""Camera-view rendering: images, projected boxes and projected LiDAR.

Frames come from :mod:`t4_e2e_devkit.dataset.camera_source`, the same decoder the
data loader uses, so a rendered view is the pixels the model saw. That matters
more than it sounds: these scenes store some cameras as JPEG and others as HEVC,
and a second decoder here would put a systematic difference between what is shown
and what was trained on.

PROJECTION
----------

The extrinsic is camera-to-ego, so a point goes the other way::

    p_cam = (p_ego - camera2ego_translation) @ camera2ego_rotation
    uv    = (p_cam @ intrinsics.T)[:2] / p_cam[2]        # only where p_cam[2] > 0

Verified against real scenes rather than assumed: the rotation maps camera
``(x right, y down, z forward)`` onto ego ``(-y, -z, +x)``; the translation is
1.30 m forward and 1.89 m up, which is where a windshield camera is; ground-level
LiDAR points land in the lower half of the image (mean row 490 of 672) and points
above 2 m land in the upper half (mean row 306). Intrinsics are already rescaled
to the stored resolution by the reader.

**Distortion is not applied, and for these datasets that is exact rather than an
approximation.** ``camera_distortion`` is all zeros for every camera channel of
``prd_jt`` and of ``x2_dev`` -- 6600 camera records over 600 sampled ``x2_dev``
scenes, and every channel of the ``jpntaxigen2`` rig -- so the images are already
rectified and a pinhole projection is the correct model. :class:`Camera` carries a
``distortion`` field and :func:`project_with_distortion` implements the full
OpenCV model for a rig that ever ships non-zero coefficients; the gen-1 jpntaxi
perception DBs do, and are out of scope here.

SENSOR TIME CORRECTION
----------------------

Boxes are drawn at **the camera's own capture time**, not the LiDAR sweep's.
``derived/`` exposes one frame index per sensor, which reads as though they were
simultaneous; the per-sensor timestamps in ``annotation/sample_data.json`` show
each camera trailing its LiDAR frame by a stable per-channel constant, from
+50 ms to +116 ms, differing by 66 ms between channels.

Uncorrected, that displaces a projected box by 0.3-0.8 m at 6.3 m/s, which on
``CAM_FRONT_LEFT_WIDE`` measures **54 px median, 77 px maximum on a 1148 px-wide
image** -- boxes visibly off their objects, not a sub-pixel refinement.

:mod:`t4_e2e_devkit.dataset.sync` performs the correction and
:attr:`Camera.annotations` carries the result, which
:func:`add_annotations_to_camera_ax` prefers by default. Use
:func:`add_sync_comparison_to_camera_ax` to see both sets at once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.dataclasses import Annotations, Camera, Lidar
from t4_e2e_devkit.common.enums import T4BoxIndex
from t4_e2e_devkit.visualization.config import (
    LIDAR_CONFIG,
    TRACK_COLORS,
    UNKNOWN_TRACK_COLOR,
)
from t4_e2e_devkit.visualization.lidar import get_lidar_pc_color, subsample_lidar_pc

#: Edges of a 3D box, indexing the corner order from :func:`box_corners_3d`.
BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)


def box_corners_3d(box: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """The eight corners of one T4 box, in ego coordinates.

    Columns follow :class:`~t4_e2e_devkit.common.enums.T4BoxIndex`, so width comes
    from column 3 and length from column 4.

    ``z`` is the box **centre**, established from the data rather than assumed: for
    a 1.48 m car the stored ``z`` is 1.04, which as a bottom would float the wheels
    a metre up and as a top would sink the body 0.44 m into the road. Measured
    across classes the bottom then sits about 0.3 m above the ego's ground plane,
    consistently enough to be a frame offset rather than per-class error -- so
    expect projected boxes to hover slightly above the road surface.

    :param box: one row of ``[x, y, z, w, l, h, yaw, vx, vy]``.
    :return: ``[8, 3]`` corners, bottom face first, counter-clockwise from the
        front-left.
    """
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    half_length = max(float(values[T4BoxIndex.LENGTH]), 1e-3) / 2.0
    half_width = max(float(values[T4BoxIndex.WIDTH]), 1e-3) / 2.0
    half_height = max(float(values[T4BoxIndex.HEIGHT]), 1e-3) / 2.0

    local = np.array(
        [
            [half_length, half_width, -half_height],
            [half_length, -half_width, -half_height],
            [-half_length, -half_width, -half_height],
            [-half_length, half_width, -half_height],
            [half_length, half_width, half_height],
            [half_length, -half_width, half_height],
            [-half_length, -half_width, half_height],
            [-half_length, half_width, half_height],
        ]
    )
    heading = float(values[T4BoxIndex.HEADING])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos_h, -sin_h, 0.0], [sin_h, cos_h, 0.0], [0.0, 0.0, 1.0]])
    centre = values[[T4BoxIndex.X, T4BoxIndex.Y, T4BoxIndex.Z]]
    return local @ rotation.T + centre


def project_with_distortion(
    points_camera: npt.NDArray[np.floating],
    intrinsics: npt.NDArray[np.floating],
    distortion: Optional[npt.NDArray[np.floating]] = None,
) -> npt.NDArray[np.float64]:
    """Project camera-frame points to pixels, applying the OpenCV distortion model.

    Coefficients follow OpenCV's order, with 4, 5, 8, 12 or 14 entries:
    ``(k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, tau_x, tau_y]]]])``.

    The model is applied to **normalised** coordinates ``(X/Z, Y/Z)``, which is
    where OpenCV defines it. This differs from ``t4-devkit``'s ``view_points``,
    which evaluates the radial term on the unnormalised ``(X, Y)`` and divides the
    resulting pixel coordinates by ``Z`` afterwards; the two agree exactly when
    ``Z = 1``, which is the case in every one of its unit tests, and diverge
    otherwise. Since the coefficients are zero throughout the datasets in scope,
    nothing observable turns on the difference today -- but a future rig with real
    coefficients would need the normalised form.

    :param points_camera: ``[N, 3]`` in camera coordinates.
    :param intrinsics: ``[3, 3]`` matrix.
    :param distortion: coefficients, or ``None``/all-zero for a pure pinhole.
    :return: ``[N, 2]`` pixel coordinates. Rows with ``Z <= 0`` are meaningless and
        must be masked by the caller.
    """
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    depth = np.where(np.abs(points[:, 2]) < 1e-12, 1e-12, points[:, 2])
    x, y = points[:, 0] / depth, points[:, 1] / depth

    if distortion is not None and np.any(np.abs(np.asarray(distortion, np.float64)) > 0):
        coefficients = np.zeros(14, dtype=np.float64)
        given = np.asarray(distortion, dtype=np.float64).reshape(-1)
        coefficients[: given.size] = given
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y = coefficients

        r2 = x * x + y * y
        radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (
            1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        )
        xy = x * y
        x_d = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2**2
        y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * xy + s3 * r2 + s4 * r2**2

        if tau_x != 0.0 or tau_y != 0.0:
            rotation = np.array(
                [
                    [np.cos(tau_y), 0, -np.sin(tau_y)],
                    [
                        np.sin(tau_x) * np.sin(tau_y),
                        np.cos(tau_x),
                        np.sin(tau_x) * np.cos(tau_y),
                    ],
                    [
                        np.cos(tau_x) * np.sin(tau_y),
                        -np.sin(tau_x),
                        np.cos(tau_x) * np.cos(tau_y),
                    ],
                ]
            )
            r13, r23, r33 = rotation[0, 2], rotation[1, 2], rotation[2, 2]
            tilt = np.array([[r33, 0, -r13], [0, r33, -r23], [0, 0, 1]]) @ rotation
            tilted = tilt @ np.vstack([x_d, y_d, np.ones_like(x_d)])
            x_d, y_d = tilted[0] / tilted[2], tilted[1] / tilted[2]
        x, y = x_d, y_d

    homogeneous = np.stack([x, y, np.ones_like(x)], axis=-1) @ matrix.T
    return homogeneous[:, :2]


def project_ego_points(
    camera: Camera,
    points_ego: npt.NDArray[np.floating],
    min_depth: float = 0.5,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Project ego-frame points into one camera, honouring its distortion if present.
    :param camera: the calibrated view.
    :param points_ego: ``[N, 3]`` in ego coordinates.
    :param min_depth: drop points closer than this to the pinhole.
    :return: ``([N, 2]`` pixels, ``[N]`` in-front mask``)``.
    """
    camera_points = camera.ego_to_camera(points_ego)
    valid = camera_points[:, 2] > float(min_depth)
    pixels = project_with_distortion(camera_points, camera.intrinsics, camera.distortion)
    return pixels, valid


# --------------------------------------------------------------------------- #
# Axis primitives
# --------------------------------------------------------------------------- #


def add_camera_ax(ax, camera: Camera, title: Optional[str] = None):
    """
    Draw one camera image.
    :param ax: target axes.
    :param camera: the view.
    :param title: axis title; the camera name by default.
    :return: the axes.
    """
    if camera.image is not None:
        ax.imshow(camera.image)
    else:
        # A missing view is a fact about the data, not an empty panel: the reader
        # fills it with the ImageNet mean and the window stays trainable.
        ax.imshow(np.full((10, 18, 3), 128, dtype=np.uint8))
        ax.text(
            0.5, 0.5, "no image", transform=ax.transAxes,
            ha="center", va="center", color="white", fontsize=11,
        )
    ax.set_title(title if title is not None else camera.name, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def add_annotations_to_camera_ax(
    ax,
    camera: Camera,
    annotations: Optional[Annotations] = None,
    line_width: float = 1.4,
    min_depth: float = 0.5,
    use_synced: bool = True,
):
    """Draw projected 3D boxes as wireframes.

    Prefers ``camera.annotations`` -- the boxes moved to **this camera's** capture
    time and ego frame -- over the frame-level ones. That is not a refinement: the
    channels of one frame trail the LiDAR sweep by 50 to 116 ms and differ from
    each other by 66 ms, which at urban speed is 0.3 to 0.8 m of displacement, so
    drawing the LiDAR-time boxes puts every object visibly off its image position.

    A box is drawn only when **every** corner is in front of the pinhole. Partially
    visible boxes are skipped rather than clipped: a box straddling the image plane
    projects some corners to wild coordinates, and drawing those produces long
    stray lines across the frame that read as detections.

    :param ax: target axes, already showing the image.
    :param camera: the calibrated view.
    :param annotations: fallback boxes, used when the camera carries no synced set.
    :param line_width: wireframe width.
    :param min_depth: corners nearer than this disqualify the box.
    :param use_synced: prefer the camera's time-corrected boxes.
    :return: the axes.
    """
    if not camera.is_calibrated or camera.image is None:
        return ax
    source = camera.annotations if (use_synced and camera.annotations is not None) else annotations
    if source is None:
        return ax
    height, width = camera.image.shape[:2]
    boxes = np.asarray(source.boxes, dtype=np.float64)
    labels = np.asarray(source.labels, dtype=np.int64)

    for box, label in zip(boxes, labels, strict=True):
        corners = box_corners_3d(box)
        pixels, valid = project_ego_points(camera, corners, min_depth=min_depth)
        if not valid.all():
            continue
        if pixels[:, 0].max() < 0 or pixels[:, 0].min() > width:
            continue
        if pixels[:, 1].max() < 0 or pixels[:, 1].min() > height:
            continue
        color = TRACK_COLORS.get(int(label), UNKNOWN_TRACK_COLOR)
        for start, end in BOX_EDGES:
            ax.plot(
                [pixels[start, 0], pixels[end, 0]],
                [pixels[start, 1], pixels[end, 1]],
                color=color,
                linewidth=line_width,
                zorder=3,
            )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    return ax


def add_sync_comparison_to_camera_ax(
    ax,
    camera: Camera,
    annotations: Annotations,
    line_width: float = 1.6,
):
    """Draw the uncorrected and corrected boxes on the same image.

    Both sets on one axis on purpose: rendering them as two panels invites the
    reader to compare across figures whose axes scale independently, which makes
    a real displacement look like a viewpoint change. Dashed grey is the LiDAR
    timestamp, solid colour is this camera's.

    :param ax: target axes, already showing the image.
    :param camera: the calibrated view; must carry synced annotations.
    :param annotations: the uncorrected, frame-level boxes.
    :param line_width: wireframe width.
    :return: the axes.
    :raises ValueError: when the camera has no corrected set to compare against.
    """
    if camera.annotations is None:
        raise ValueError(
            f"camera {camera.name!r} carries no time-corrected annotations, so there "
            "is nothing to compare; the scene may have no annotation/ tables"
        )
    add_annotations_to_camera_ax(
        ax, camera, annotations, line_width=line_width * 0.8, use_synced=False
    )
    for line in ax.get_lines():
        line.set_color("#8A8A8A")
        line.set_linestyle("--")
    add_annotations_to_camera_ax(ax, camera, line_width=line_width, use_synced=True)
    return ax


def add_lidar_to_camera_ax(
    ax,
    camera: Camera,
    lidar: Lidar,
    config: Optional[Dict[str, Any]] = None,
    point_size: float = 1.2,
    max_points: int = 40_000,
    seed: int = 0,
):
    """Overlay the projected point cloud.

    Coloured by depth in the camera frame rather than by the BEV colour element,
    since on an image the useful cue is distance from the viewer.

    :param ax: target axes, already showing the image.
    :param camera: the calibrated view.
    :param lidar: the sweep, in the same ego frame.
    :param config: overrides for :data:`LIDAR_CONFIG`.
    :param point_size: scatter size.
    :param max_points: subsample cap before projection.
    :param seed: subsample seed.
    :return: the axes.
    """
    if not camera.is_calibrated or camera.image is None or lidar is None:
        return ax
    if lidar.lidar_pc is None:
        return ax

    settings = {**LIDAR_CONFIG, **(config or {})}
    points, _ = subsample_lidar_pc(lidar.lidar_pc, max_points, seed=seed)
    pixels, valid = project_ego_points(camera, points[:, :3])
    inside = camera.image_bounds_mask(pixels, valid)
    if not inside.any():
        return ax

    visible = points[inside]
    depth = camera.ego_to_camera(visible[:, :3])[:, 2]
    colors = get_lidar_pc_color(
        np.column_stack([depth, depth, depth, visible[:, 3:]]),
        {**settings, "color_element": "x"},
    )
    ax.scatter(
        pixels[inside, 0],
        pixels[inside, 1],
        s=point_size,
        c=colors,
        alpha=0.6,
        linewidths=0,
        zorder=2,
    )
    height, width = camera.image.shape[:2]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    return ax


def add_trajectory_to_camera_ax(
    ax,
    camera: Camera,
    poses: npt.NDArray[np.floating],
    color: str,
    ground_z: float = 0.0,
    line_width: float = 2.4,
    label: Optional[str] = None,
):
    """Project a planned path onto the road surface.

    The trajectory is planar, so it is lifted to ``ground_z`` before projection.
    Only the leading run of in-front points is drawn: a path that starts behind
    the camera would otherwise reappear as a spurious segment when it crosses the
    image plane.

    :param ax: target axes, already showing the image.
    :param camera: the calibrated view.
    :param poses: ``[T, >=2]`` ego-frame poses.
    :param color: line colour.
    :param ground_z: height of the road surface in the ego frame.
    :param line_width: line width.
    :param label: legend label.
    :return: the axes.
    """
    if not camera.is_calibrated or camera.image is None:
        return ax
    xy = np.asarray(poses, dtype=np.float64)[:, :2]
    xyz = np.column_stack([xy, np.full(len(xy), float(ground_z))])
    pixels, valid = project_ego_points(camera, xyz)
    if not valid.any():
        return ax
    first_invalid = np.argmin(valid) if not valid.all() else len(valid)
    run = pixels[:first_invalid]
    if len(run) < 2:
        return ax
    ax.plot(
        run[:, 0], run[:, 1],
        color=color, linewidth=line_width, alpha=0.9, zorder=4, label=label,
    )
    return ax


def camera_grid_layout(camera_names: Sequence[str]) -> List[List[Optional[str]]]:
    """Arrange a register so left-right adjacency is spatially truthful.

    A flat strip of five images tells a reader nothing about where the views point.
    This places front views on the top row and rear views below, in left-to-right
    order, and puts a centred rear camera under the centre column when the rig has
    one -- ``x2_dev`` does, ``prd_jt`` does not, which is exactly the kind of thing
    a fixed layout gets wrong.

    Cameras that are not part of the surround -- signal heads and roof views --
    are appended on their own row, because their geometry does not compose with
    the others.

    :param camera_names: the register to lay out.
    :return: a grid of names, with ``None`` for blank cells.
    """
    remaining = list(camera_names)

    def take(*candidates: str) -> Optional[str]:
        for candidate in candidates:
            for name in remaining:
                if name.upper() == candidate:
                    remaining.remove(name)
                    return name
        return None

    front_left = take("CAM_FRONT_LEFT_WIDE", "CAM_FRONT_LEFT")
    front = take("CAM_FRONT_WIDE", "CAM_FRONT")
    front_right = take("CAM_FRONT_RIGHT_WIDE", "CAM_FRONT_RIGHT")
    back_left = take("CAM_BACK_LEFT_WIDE", "CAM_BACK_LEFT")
    back = take("CAM_BACK_WIDE", "CAM_BACK")
    back_right = take("CAM_BACK_RIGHT_WIDE", "CAM_BACK_RIGHT")

    rows: List[List[Optional[str]]] = []
    if any((front_left, front, front_right)):
        rows.append([front_left, front, front_right])
    if any((back_left, back, back_right)):
        rows.append([back_left, back, back_right])
    while remaining:
        rows.append([*remaining[:3], *([None] * (3 - len(remaining[:3])))])
        remaining = remaining[3:]
    return rows or [[None, None, None]]
