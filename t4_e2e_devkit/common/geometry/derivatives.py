"""Savitzky-Golay derivative approximation.

``approximate_derivatives`` is copied verbatim from ``nuplan.planning.metrics.
utils.state_extractors``; it is re-homed here because the nuPlan module it
lives in imports ``SimulationHistory`` for the rest of its contents, and this
one function is the only thing the devkit needs from it.

The parameters matter: the PDM comfort metric and the LQR tracker both read
accelerations and yaw rates through this filter, so window length, polynomial
order and the equal-spacing assumption are part of the scoring contract rather
than an implementation detail.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter


def approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f'{poly_order} < {window_length} does not hold!')

    dx = np.diff(x)
    if not (dx > 0).all():
        raise RuntimeError('dx is not monotonically increasing!')

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y, polyorder=poly_order, window_length=window_length, deriv=deriv_order, delta=dx, axis=axis
    )
    return derivative
