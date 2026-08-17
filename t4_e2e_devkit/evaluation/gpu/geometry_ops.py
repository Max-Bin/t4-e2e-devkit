"""Reusable GPU geometry operations for the online PDM reference.

The low-level collision predicates remain the single source of truth for
convex-quad intersection.  This module composes them with polyline projection
to represent the exact geometry used by ``PDMGenerator``:

* a finite path subline has round joins;
* its two caps are square, as requested by Shapely's ``cap_style=square``;
* polygon distances include containment and edge intersection.

All tensors stay on the caller's device.  Variable-length rings are the only
place where a small Python loop remains; their coordinates and every geometry
operation remain device-resident.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from t4_e2e_devkit.evaluation.gpu.areas import points_in_polygons_torch
from t4_e2e_devkit.evaluation.gpu.collisions import (
    quads_intersect,
    segment_intersects_quad,
)
from t4_e2e_devkit.evaluation.gpu.geometry import TorchPolyline


@dataclass(frozen=True)
class SublineGeometry:
    """Device-resident finite path geometry reused across rollout steps."""

    starts: torch.Tensor
    ends: torch.Tensor
    valid: torch.Tensor
    endpoints: torch.Tensor
    headings: torch.Tensor


def oriented_box_corners(
    centers: torch.Tensor,
    headings: torch.Tensor,
    half_lengths: torch.Tensor,
    half_widths: torch.Tensor,
) -> torch.Tensor:
    """Return FL/RL/RR/FR corners for oriented rectangles."""

    longitudinal = torch.stack((torch.cos(headings), torch.sin(headings)), dim=-1)
    lateral = torch.stack((-torch.sin(headings), torch.cos(headings)), dim=-1)
    front = centers + half_lengths[..., None] * longitudinal
    rear = centers - half_lengths[..., None] * longitudinal
    return torch.stack(
        (
            front + half_widths[..., None] * lateral,
            rear + half_widths[..., None] * lateral,
            rear - half_widths[..., None] * lateral,
            front - half_widths[..., None] * lateral,
        ),
        dim=-2,
    )


def _cross_2d(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def point_segment_distance(
    points: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Distance between broadcast points and finite segments."""

    segment = ends - starts
    denominator = (segment * segment).sum(dim=-1).clamp(min=1.0e-12)
    fraction = ((points - starts) * segment).sum(dim=-1) / denominator
    fraction = fraction.clamp(0.0, 1.0)
    nearest = starts + fraction[..., None] * segment
    return torch.linalg.vector_norm(points - nearest, dim=-1)


def segment_polygon_distance(
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    polygon_corners: torch.Tensor,
) -> torch.Tensor:
    """Distance from each segment to each convex quadrilateral.

    ``segment_starts`` and ``segment_ends`` are ``[S, 2]`` and
    ``polygon_corners`` is ``[N, 4, 2]``.  The result is ``[S, N]``.
    """

    left_starts = segment_starts[:, None, None, :]
    left_ends = segment_ends[:, None, None, :]
    right_starts = polygon_corners[None, :, :, :]
    right_ends = torch.roll(right_starts, shifts=-1, dims=-2)
    distances = torch.stack(
        (
            point_segment_distance(left_starts, right_starts, right_ends),
            point_segment_distance(left_ends, right_starts, right_ends),
            point_segment_distance(right_starts, left_starts, left_ends),
            point_segment_distance(right_ends, left_starts, left_ends),
        ),
        dim=-1,
    )
    intersects = segment_intersects_quad(
        segment_starts[:, None, :],
        segment_ends[:, None, :],
        polygon_corners[None, :, :, :],
    )
    distance = distances.amin(dim=(-1, -2))
    return torch.where(intersects, torch.zeros_like(distance), distance)


def polygon_polygon_distance(
    left_corners: torch.Tensor,
    right_corners: torch.Tensor,
) -> torch.Tensor:
    """Distance between every pair of convex quadrilaterals."""

    left_starts = left_corners[:, None, :, None, :]
    left_ends = torch.roll(left_corners, shifts=-1, dims=-2)[:, None, :, None, :]
    right_starts = right_corners[None, :, None, :, :]
    right_ends = torch.roll(right_corners, shifts=-1, dims=-2)[None, :, None, :, :]
    distances = torch.stack(
        (
            point_segment_distance(left_starts, right_starts, right_ends),
            point_segment_distance(left_ends, right_starts, right_ends),
            point_segment_distance(right_starts, left_starts, left_ends),
            point_segment_distance(right_ends, left_starts, left_ends),
        ),
        dim=-1,
    )
    distance = distances.amin(dim=(-1, -2, -3))
    intersects = quads_intersect(left_corners[:, None], right_corners[None])
    return torch.where(intersects, torch.zeros_like(distance), distance)


def _square_endpoint_caps(
    endpoints: torch.Tensor,
    headings: torch.Tensor,
    width: float,
) -> torch.Tensor:
    """Build Shapely square caps that extend outward from each endpoint."""

    tangent = torch.stack((torch.cos(headings), torch.sin(headings)), dim=-1)
    direction = torch.tensor((-0.5, 0.5), device=endpoints.device, dtype=endpoints.dtype)
    centers = endpoints + direction[:, None] * float(width) * tangent
    half_length = torch.full(
        (2,), float(width) / 2.0, device=endpoints.device, dtype=endpoints.dtype
    )
    half_width = torch.full((2,), float(width), device=endpoints.device, dtype=endpoints.dtype)
    return oriented_box_corners(centers, headings, half_length, half_width)


def _subline_segments(
    path: TorchPolyline,
    start_distance: torch.Tensor,
    end_distance: torch.Tensor,
) -> SublineGeometry:
    """Return the exact finite path geometry used by ``PDMPath.substring``."""

    start = torch.as_tensor(start_distance, device=path.vertices.device, dtype=path.vertices.dtype)
    end = torch.as_tensor(end_distance, device=path.vertices.device, dtype=path.vertices.dtype)
    clipped_start = start.clamp(min=0.0, max=path.length)
    clipped_end = end.clamp(min=0.0, max=path.length)

    # ``PDMPath.substring`` has a fast path that keeps complete discrete
    # vertices whenever at least two lie in the interval.  Reproduce that
    # observable behavior before falling back to Shapely's interpolated
    # substring equivalent for a one-segment interval.
    vertex_mask = (path._cumulative >= clipped_start) & (path._cumulative <= clipped_end)
    if int(vertex_mask.sum().item()) >= 2:
        vertex_indices = torch.where(vertex_mask)[0]
        segment_starts = path._starts
        segment_ends = path._starts + path._segments
        valid = vertex_mask[:-1] & vertex_mask[1:]
        endpoints = path.vertices[vertex_indices[[0, -1]]]
        headings = path.headings[vertex_indices[[0, -1]]]
        return SublineGeometry(segment_starts, segment_ends, valid, endpoints, headings)

    lower = torch.maximum(path._cumulative[:-1], clipped_start)
    upper = torch.minimum(path._cumulative[1:], clipped_end)
    valid = upper >= lower
    lengths = path._lengths.clamp(min=1.0e-12)
    lower_fraction = (lower - path._cumulative[:-1]) / lengths
    upper_fraction = (upper - path._cumulative[:-1]) / lengths
    segment_starts = path._starts + lower_fraction[..., None] * path._segments
    segment_ends = path._starts + upper_fraction[..., None] * path._segments
    endpoint_distances = torch.stack((clipped_start, clipped_end))
    endpoints = path.interpolate(endpoint_distances)
    return SublineGeometry(
        segment_starts, segment_ends, valid, endpoints[..., :2], endpoints[..., 2]
    )


def polyline_polygon_buffer_intersects(
    path: TorchPolyline,
    polygon_corners: torch.Tensor,
    width: float,
    start_distance: torch.Tensor,
    end_distance: torch.Tensor,
    subline: SublineGeometry | None = None,
) -> torch.Tensor:
    """Test polygons against ``path.substring(...).buffer(square)``.

    The finite subline itself has Shapely's round joins, represented by the
    exact distance-to-segment test.  Shapely's square caps are explicit
    oriented rectangles at the two subline endpoints.
    """

    geometry = (
        subline
        if subline is not None
        else _subline_segments(path, start_distance, end_distance)
    )
    starts, ends, valid = geometry.starts, geometry.ends, geometry.valid
    endpoints, headings = geometry.endpoints, geometry.headings
    distances = segment_polygon_distance(starts, ends, polygon_corners)
    distances = torch.where(valid[:, None], distances, torch.full_like(distances, float("inf")))
    line_hit = distances.amin(dim=0) <= float(width)
    cap_corners = _square_endpoint_caps(endpoints, headings, width)
    cap_hit = quads_intersect(cap_corners[:, None], polygon_corners[None]).any(dim=0)
    return line_hit | cap_hit


def points_in_ring(
    points: torch.Tensor,
    edge_starts: torch.Tensor,
    edge_ends: torch.Tensor,
) -> torch.Tensor:
    """Use the shared even-odd GPU kernel with one ring."""

    point_shape = points.shape[:-1]
    return points_in_polygons_torch(
        points.reshape(-1, 2), edge_starts[None], edge_ends[None]
    )[0].reshape(point_shape)


def segment_ring_distance(
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    ring: torch.Tensor,
) -> torch.Tensor:
    """Distance from each segment to one closed polygon ring."""

    ring_starts = ring[:-1]
    ring_ends = ring[1:]
    left_starts = segment_starts[:, None, :]
    left_ends = segment_ends[:, None, :]
    right_starts = ring_starts[None, :, :]
    right_ends = ring_ends[None, :, :]
    distances = torch.stack(
        (
            point_segment_distance(left_starts, right_starts, right_ends),
            point_segment_distance(left_ends, right_starts, right_ends),
            point_segment_distance(right_starts, left_starts, left_ends),
            point_segment_distance(right_ends, left_starts, left_ends),
        ),
        dim=-1,
    )
    left_vector = left_ends - left_starts
    right_vector = right_ends - right_starts
    denominator = _cross_2d(left_vector, right_vector)
    safe_denominator = torch.where(
        denominator.abs() > 1.0e-12, denominator, torch.ones_like(denominator)
    )
    offset = right_starts - left_starts
    t = _cross_2d(offset, right_vector) / safe_denominator
    u = _cross_2d(offset, left_vector) / safe_denominator
    intersects = (
        (denominator.abs() > 1.0e-12)
        & (t >= 0.0)
        & (t <= 1.0)
        & (u >= 0.0)
        & (u <= 1.0)
    )
    distance = distances.amin(dim=(-1, -2))
    distance = torch.where(intersects.any(dim=-1), torch.zeros_like(distance), distance)
    inside = points_in_ring(segment_starts, ring_starts, ring_ends) | points_in_ring(
        segment_ends, ring_starts, ring_ends
    )
    return torch.where(inside, torch.zeros_like(distance), distance)


def quad_ring_distance(quads: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """Distance from each convex quadrilateral to one polygon ring."""

    quad_starts = quads[:, :, None, :]
    quad_ends = torch.roll(quads, shifts=-1, dims=-2)[:, :, None, :]
    ring_starts = ring[:-1][None, None, :, :]
    ring_ends = ring[1:][None, None, :, :]
    distances = torch.stack(
        (
            point_segment_distance(quad_starts, ring_starts, ring_ends),
            point_segment_distance(quad_ends, ring_starts, ring_ends),
            point_segment_distance(ring_starts, quad_starts, quad_ends),
            point_segment_distance(ring_ends, quad_starts, quad_ends),
        ),
        dim=-1,
    )
    left_vector = quad_ends - quad_starts
    right_vector = ring_ends - ring_starts
    denominator = _cross_2d(left_vector, right_vector)
    safe_denominator = torch.where(
        denominator.abs() > 1.0e-12, denominator, torch.ones_like(denominator)
    )
    offset = ring_starts - quad_starts
    t = _cross_2d(offset, right_vector) / safe_denominator
    u = _cross_2d(offset, left_vector) / safe_denominator
    intersects = (
        (denominator.abs() > 1.0e-12)
        & (t >= 0.0)
        & (t <= 1.0)
        & (u >= 0.0)
        & (u <= 1.0)
    )
    distance = distances.amin(dim=(-1, -2, -3))
    distance = torch.where(intersects.any(dim=(-1, -2)), torch.zeros_like(distance), distance)
    left_inside = points_in_ring(quads, ring_starts[0, 0], ring_ends[0, 0]).any(dim=-1)
    right_inside = torch.stack(
        [
            points_in_ring(
                ring_starts[0, 0], quad, torch.roll(quad, shifts=-1, dims=-2)
            ).any()
            for quad in quads
        ]
    )
    return torch.where(left_inside | right_inside, torch.zeros_like(distance), distance)


def polyline_ring_buffer_intersects(
    path: TorchPolyline,
    rings: torch.Tensor,
    lengths: torch.Tensor,
    width: float,
    start_distance: torch.Tensor,
    end_distance: torch.Tensor,
    subline: SublineGeometry | None = None,
) -> torch.Tensor:
    """Test rings against ``path.substring(...).buffer(square)``."""

    geometry = (
        subline
        if subline is not None
        else _subline_segments(path, start_distance, end_distance)
    )
    starts, ends, valid = geometry.starts, geometry.ends, geometry.valid
    endpoints, headings = geometry.endpoints, geometry.headings
    cap_corners = _square_endpoint_caps(endpoints, headings, width)
    hits = []
    for ring, length_value in zip(rings, lengths.detach().cpu().tolist(), strict=True):
        length = int(length_value)
        values = ring[:length]
        distances = segment_ring_distance(starts, ends, values)
        distances = torch.where(valid, distances, torch.full_like(distances, float("inf")))
        line_hit = distances.amin() <= float(width)
        cap_hit = quad_ring_distance(cap_corners, values).amin() <= 0.0
        hits.append(line_hit | cap_hit)
    if not hits:
        return torch.empty((0,), device=path.vertices.device, dtype=torch.bool)
    return torch.stack(hits)


def ring_centroids(rings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Compute polygon centroids with the shoelace definition."""

    centroids = []
    for ring, length_value in zip(rings, lengths.detach().cpu().tolist(), strict=True):
        values = ring[: int(length_value)]
        if values.shape[0] > 1 and bool(torch.equal(values[0], values[-1])):
            values = values[:-1]
        next_values = torch.roll(values, shifts=-1, dims=0)
        cross = values[:, 0] * next_values[:, 1] - values[:, 1] * next_values[:, 0]
        area_twice = cross.sum()
        non_degenerate = area_twice.abs() > 1.0e-12
        safe_area = torch.where(non_degenerate, area_twice, torch.ones_like(area_twice))
        centroid = torch.stack(
            (
                ((values[:, 0] + next_values[:, 0]) * cross).sum() / (3.0 * safe_area),
                ((values[:, 1] + next_values[:, 1]) * cross).sum() / (3.0 * safe_area),
            )
        )
        centroids.append(torch.where(non_degenerate, centroid, values.mean(dim=0)))
    return torch.stack(centroids)


__all__ = [
    "SublineGeometry",
    "oriented_box_corners",
    "point_segment_distance",
    "polygon_polygon_distance",
    "polyline_polygon_buffer_intersects",
    "polyline_ring_buffer_intersects",
    "quad_ring_distance",
    "ring_centroids",
    "segment_polygon_distance",
]
