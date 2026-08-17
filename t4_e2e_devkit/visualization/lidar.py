"""Filtering and colouring a T4 point cloud for display.

T4 points are ``[N, 5]`` -- ``(x, y, z, intensity, ring_or_time)`` -- laid out
row-per-point. The concatenated cloud has no per-point object id, so asking for
``id`` raises rather than returning zeros.
* Subsampling is explicit. A full T4 sweep is ~440k points; a scatter of all of
  them produces a multi-megabyte figure indistinguishable from a subsampled one,
  so :func:`subsample_lidar_pc` is applied by default and reports what it did.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from t4_e2e_devkit.common.enums import LidarIndex
from t4_e2e_devkit.visualization.config import LIDAR_CONFIG

#: Column name -> index, for ``LIDAR_CONFIG["color_element"]``.
_COLOR_ELEMENTS: Dict[str, int] = {
    "x": LidarIndex.X,
    "y": LidarIndex.Y,
    "z": LidarIndex.Z,
    "intensity": LidarIndex.INTENSITY,
    "ring": LidarIndex.RING,
}


def filter_lidar_pc(
    lidar_pc: npt.NDArray[np.floating],
    config: Optional[Dict[str, Any]] = None,
) -> npt.NDArray[np.float32]:
    """
    Crop a point cloud to the configured view box.
    :param lidar_pc: ``[N, 5]`` T4 points.
    :param config: overrides for :data:`LIDAR_CONFIG`.
    :return: the points inside the box, ``[M, 5]``.
    """
    config = {**LIDAR_CONFIG, **(config or {})}
    points = np.asarray(lidar_pc, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"expected [N, >=3] T4 points; got shape {points.shape}")

    x_lim, y_lim, z_lim = config["x_lim"], config["y_lim"], config["z_lim"]
    mask = (
        (points[:, LidarIndex.X] > x_lim[0])
        & (points[:, LidarIndex.X] < x_lim[1])
        & (points[:, LidarIndex.Y] > y_lim[0])
        & (points[:, LidarIndex.Y] < y_lim[1])
        & (points[:, LidarIndex.Z] > z_lim[0])
        & (points[:, LidarIndex.Z] < z_lim[1])
    )
    return points[mask]


def subsample_lidar_pc(
    lidar_pc: npt.NDArray[np.floating],
    max_points: Optional[int] = None,
    seed: int = 0,
) -> Tuple[npt.NDArray[np.float32], int]:
    """Thin a cloud to a renderable size.

    Uniform random selection rather than a stride: a stride over a cloud ordered
    by ring or by azimuth removes whole scan lines or a whole angular sector,
    which changes what the plot appears to show.

    :param lidar_pc: ``[N, 5]`` T4 points.
    :param max_points: cap; :data:`LIDAR_CONFIG` default otherwise.
    :param seed: selection seed, so a figure is reproducible.
    :return: ``(points, dropped)`` -- the kept points and how many were removed,
        so a caller can say so in a title instead of implying it drew them all.
    """
    points = np.asarray(lidar_pc, dtype=np.float32)
    cap = LIDAR_CONFIG["max_points"] if max_points is None else max_points
    if cap is None or points.shape[0] <= cap:
        return points, 0
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=int(cap), replace=False)
    return points[np.sort(keep)], int(points.shape[0] - cap)


def get_lidar_pc_color(
    lidar_pc: npt.NDArray[np.floating],
    config: Optional[Dict[str, Any]] = None,
    as_hex: bool = False,
) -> List[Any]:
    """
    Colour a point cloud by one of its columns, or by range.
    :param lidar_pc: ``[N, 5]`` T4 points.
    :param config: overrides for :data:`LIDAR_CONFIG`.
    :param as_hex: return hex strings rather than RGBA rows.
    :return: one colour per point.
    :raises ValueError: for an unknown ``color_element``. ``"id"`` is rejected
        because T4 clouds do not carry per-point object ids.
    """
    import matplotlib

    config = {**LIDAR_CONFIG, **(config or {})}
    points = np.asarray(lidar_pc, dtype=np.float32)
    element = str(config["color_element"]).lower()

    if element == "none":
        return ["#000000"] * points.shape[0] if as_hex else [(0.0, 0.0, 0.0, 1.0)] * points.shape[0]
    if element == "distance":
        values = np.linalg.norm(points[:, LidarIndex.POSITION], axis=-1)
    elif element in _COLOR_ELEMENTS:
        values = points[:, _COLOR_ELEMENTS[element]]
    else:
        raise ValueError(
            f"unknown LiDAR color_element {element!r}; T4 points are "
            f"[x, y, z, intensity, ring_or_time], so valid choices are "
            f"{['none', 'distance', *sorted(_COLOR_ELEMENTS)]}"
        )

    span = float(values.max() - values.min()) if values.size else 0.0
    normalized = (values - values.min()) / span if span > 1e-9 else np.zeros_like(values)
    colormap = matplotlib.colormaps[config["color_map"]]
    colors = colormap(normalized)
    if as_hex:
        return [matplotlib.colors.to_hex(color) for color in colors]
    return list(colors)


def prepare_lidar_pc(
    lidar_pc: npt.NDArray[np.floating],
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Tuple[npt.NDArray[np.float32], List[Any], int]:
    """Filter, subsample and colour in one call -- what the plots actually use.

    :param lidar_pc: ``[N, 5]`` T4 points.
    :param config: overrides for :data:`LIDAR_CONFIG`.
    :param seed: subsample seed.
    :return: ``(points, colors, dropped)``.
    """
    merged = {**LIDAR_CONFIG, **(config or {})}
    points = filter_lidar_pc(lidar_pc, merged)
    points, dropped = subsample_lidar_pc(points, merged.get("max_points"), seed=seed)
    return points, get_lidar_pc_color(points, merged), dropped
