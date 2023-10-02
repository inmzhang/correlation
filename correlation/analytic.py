"""Calculate correlation up to 2nd order analytically."""
import functools
from typing import Optional

import numpy as np

from correlation.result import CorrelationResult, AnalyticalResult
from correlation.utils import cal_two_points_expects


def cal_2nd_order_correlations(
        detection_events: np.ndarray,
        detector_mask: Optional[np.ndarray] = None,
) -> CorrelationResult:
    """Calculate the 2nd order correlation analytically.

    Args:
        detection_events: The detection events.
        detector_mask: A boolean mask of the shape (num_dets_per_shot, ). If True, the
            corresponding detection events will be excluded.

    Returns:
        The correlation result.
    """
    if detector_mask is not None:
        detection_events = detection_events[:, ~detector_mask]
    data = _analytical_core(detection_events)
    return CorrelationResult(data)


def _analytical_core(detection_events: np.ndarray) -> AnalyticalResult:
    num_shots, num_dets = detection_events.shape
    correlation_edges = np.zeros((num_dets, num_dets), dtype=float)
    correlation_bdy = np.zeros((num_dets,), dtype=float)
    expect_ixj = cal_two_points_expects(detection_events)

    # for i in range(num_dets):
    #     xi = expect_ixj[i, i]
    #     for j in range(i):
    #         xj = expect_ixj[j, j]
    #         xij = expect_ixj[i, j]
    #         try:
    #             pij = 0.5 - 0.5 * np.sqrt(1 - 4 * (xij - xi * xj) / (1 - 2 * xi - 2 * xj + 4 * xij))
    #         except ValueError:  # pragma: no cover
    #             pij = 0
    #         correlation_edges[i, j] = pij
    # correlation_edges = correlation_edges + correlation_edges.T

    i_indices, j_indices = np.tril_indices(num_dets, -1)  # Lower triangle indices excluding the diagonal

    # Extract diagonal values
    diagonal_values = expect_ixj.diagonal()
    xi_values = diagonal_values[i_indices]
    xj_values = diagonal_values[j_indices]
    xij_values = expect_ixj[i_indices, j_indices]

    # Vectorized computation of pij
    denominator = 1 - 2 * xi_values - 2 * xj_values + 4 * xij_values
    numerator = xij_values - xi_values * xj_values

    under_sqrt = 1 - 4 * numerator / denominator
    # Handle potential invalid values
    valid_mask = np.where(under_sqrt > 0, True, False)
    pij_values = np.where(valid_mask, 0.5 - 0.5 * np.sqrt(under_sqrt), 0)

    correlation_edges[i_indices, j_indices] = pij_values
    correlation_edges += correlation_edges.T

    for i in range(num_dets):
        xi = expect_ixj[i, i]
        all_edges_of_i = np.delete(correlation_edges[i, :], i)
        pi_sum = functools.reduce(lambda p, q: p + q - 2 * p * q, all_edges_of_i)
        pi_bdy = (xi - pi_sum) / (1 - 2 * pi_sum)
        correlation_bdy[i] = pi_bdy
    return correlation_bdy, correlation_edges
