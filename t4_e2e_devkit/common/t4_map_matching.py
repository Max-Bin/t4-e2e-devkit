"""Matching scene-local map tensor rows back to source map objects.

The numeric map tensor a model reads carries no identifiers, so recovering them
is a geometric question: take a row, place it in the world through the window's
centre pose, and find the source object it describes.  That is an algorithm with
its own state -- two spatial indexes, built lazily because most callers never ask
for them -- and it was 293 lines of it living inside the query API, whose other
41 methods are one-line dictionary lookups.

So the matcher owns the lanes and its indexes, and
:class:`~t4_e2e_devkit.common.t4_map.T4MapAPI` keeps the vocabulary: it resolves
which objects are candidates for a layer, and delegates the geometry.  Nothing
here reads a file or a scene.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from t4_e2e_devkit.common.dataclasses import MapObjectMatch
from t4_e2e_devkit.common.t4_map_geometry import (
    _geometry_score,
    _local_to_global,
    _polyline_score,
    _repair_polygon,
    _row_points,
    _source_geometry,
)
from t4_e2e_devkit.common.t4_map_types import T4Lanelet, T4MapObject


class MapMatcher:
    """Row matching and spatial queries over one parsed map's lanelets.

    :param lanes: the map's lanelets, in parse order.
    :param source_label: the map file's portable label, stamped into every match
        so a record says which map produced it.
    """

    def __init__(self, lanes: Sequence[T4Lanelet], source_label: str) -> None:
        self.lanes = tuple(lanes)
        self.source_label = str(source_label)
        self._centerline_tree: Optional[STRtree] = None
        self._centerline_geometries: Optional[tuple[LineString, ...]] = None
        self._polygon_tree: Optional[STRtree] = None
        self._polygon_geometries: Optional[tuple[BaseGeometry, ...]] = None

    def get_proximal_lanes(self, point: Sequence[float], radius: float) -> tuple[T4Lanelet, ...]:
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if not self.lanes:
            return ()
        query = Point(float(point[0]), float(point[1]))
        tree = self._get_polygon_tree()
        indices = tree.query(query.buffer(float(radius)))
        lanes = [self.lanes[int(index)] for index in indices]
        return tuple(
            sorted(
                (lane for lane in lanes if lane.polygon.distance(query) <= radius),
                key=lambda lane: lane.id,
            )
        )

    def get_nearest_lane(self, point: Sequence[float]) -> Optional[T4Lanelet]:
        if not self.lanes:
            return None
        query = Point(float(point[0]), float(point[1]))
        tree = self._get_polygon_tree()
        index = int(tree.nearest(query))
        return self.lanes[index]

    def match_local_centerlines_detailed(
        self,
        segments: np.ndarray,
        center_pose: Sequence[float],
        *,
        layer: str = "lanes",
        frame_index: Optional[int] = None,
        allowed_ids: Optional[Sequence[str]] = None,
        max_distance: float = 3.0,
        candidate_limit: int = 5,
    ) -> tuple[MapObjectMatch, ...]:
        """Match centerline rows and retain evidence for every row.

        ``match_distance_m`` is the mean point-to-point distance after
        resampling both polylines. It is a matching score, not a geometric
        Hausdorff distance. Rows that cannot be assigned a source lane keep
        their nearest candidates and an explicit ``reason``.
        """
        values = np.asarray(segments, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] < 2:
            raise ValueError(f"centerline segments must be [N, P, >=2], got {values.shape}")
        if max_distance < 0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        if candidate_limit <= 0:
            raise ValueError(f"candidate_limit must be positive, got {candidate_limit}")
        if not str(layer):
            raise ValueError("layer must not be empty")
        pose = np.asarray(center_pose, dtype=np.float64).reshape(-1)
        if pose.size < 4:
            raise ValueError(f"center_pose must contain [x, y, cos, sin], got {pose.shape}")
        allowed = None if allowed_ids is None else {str(value) for value in allowed_ids}
        result: list[MapObjectMatch] = []
        for row_index, row in enumerate(values):
            points, has_data = _row_points(row)
            if len(points) < 2:
                result.append(
                    self._record(
                        layer=layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="padding" if not has_data else "invalid_geometry",
                    )
                )
                continue
            if not self.lanes:
                result.append(
                    self._record(
                        layer=layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="no_source_geometry",
                    )
                )
                continue
            global_points = _local_to_global(points, pose)
            ranked = self._rank_polyline_matches(global_points, allowed, max_distance)
            candidates = tuple(candidate_id for candidate_id, _ in ranked[:candidate_limit])
            if not ranked:
                result.append(
                    self._record(
                        layer=layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="no_candidate",
                    )
                )
                continue
            best_id, best_score = ranked[0]
            matched = best_score <= max_distance
            result.append(
                self._record(
                    layer=layer,
                    row_index=row_index,
                    frame_index=frame_index,
                    source_object_id=best_id if matched else None,
                    match_distance_m=best_score,
                    candidate_ids=candidates,
                    reason="matched" if matched else "above_threshold",
                )
            )
        return tuple(result)

    def match_local_geometries_detailed(
        self,
        segments: np.ndarray,
        center_pose: Sequence[float],
        *,
        layer: str,
        candidates: Sequence["T4Lanelet | T4MapObject"],
        frame_index: Optional[int] = None,
        max_distance: float = 3.0,
        candidate_limit: int = 5,
    ) -> tuple[MapObjectMatch, ...]:
        """Match polygon/line tensor rows to source semantic objects.

        The numeric tensor does not carry IDs.  This method therefore records
        the best geometric evidence and leaves rows unmatched when the source
        map is too far away, rather than manufacturing an unstable row index.
        """

        values = np.asarray(segments, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] < 2:
            raise ValueError(f"map segments must be [N, P, >=2], got {values.shape}")
        if max_distance < 0.0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")
        if candidate_limit <= 0:
            raise ValueError(f"candidate_limit must be positive, got {candidate_limit}")
        pose = np.asarray(center_pose, dtype=np.float64).reshape(-1)
        if pose.size < 4:
            raise ValueError(f"center_pose must contain [x, y, cos, sin], got {pose.shape}")
        normalized_layer = str(layer)
        result: list[MapObjectMatch] = []
        for row_index, row in enumerate(values):
            points, has_data = _row_points(row)
            if len(points) < 2:
                result.append(
                    self._record(
                        layer=normalized_layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="padding" if not has_data else "invalid_geometry",
                    )
                )
                continue
            if not candidates:
                result.append(
                    self._record(
                        layer=normalized_layer,
                        row_index=row_index,
                        frame_index=frame_index,
                        source_object_id=None,
                        match_distance_m=None,
                        candidate_ids=(),
                        reason="no_source_geometry",
                    )
                )
                continue
            global_points = _local_to_global(points, pose)
            if normalized_layer in {"polygons", "areas"} and len(global_points) >= 3:
                query_geometry: BaseGeometry = _repair_polygon(global_points) or LineString(
                    global_points
                )
            else:
                query_geometry = LineString(global_points)
            ranked = sorted(
                (
                    (obj.id, _geometry_score(query_geometry, _source_geometry(obj)))
                    for obj in candidates
                ),
                key=lambda item: (item[1], item[0]),
            )
            candidates_ids = tuple(item[0] for item in ranked[:candidate_limit])
            best_id, best_score = ranked[0]
            matched = best_score <= max_distance
            result.append(
                self._record(
                    layer=normalized_layer,
                    row_index=row_index,
                    frame_index=frame_index,
                    source_object_id=best_id if matched else None,
                    match_distance_m=best_score,
                    candidate_ids=candidates_ids,
                    reason="matched" if matched else "above_threshold",
                )
            )
        return tuple(result)

    def unmatched_rows(
        self,
        segments: np.ndarray,
        *,
        layer: str,
        frame_index: Optional[int] = None,
        reason: str = "unsupported_source_type",
    ) -> tuple[MapObjectMatch, ...]:
        """Describe rows whose source-object type is not recovered.

        T4 tensors contain polygons and line strings, but the current source
        map parser only has reliable lanelet relation IDs. These rows are
        represented explicitly as unmatched instead of receiving fabricated
        IDs.
        """
        values = np.asarray(segments, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] < 2:
            raise ValueError(f"map segments must be [N, P, >=2], got {values.shape}")
        if not str(layer):
            raise ValueError("layer must not be empty")
        result: list[MapObjectMatch] = []
        for row_index, row in enumerate(values):
            _, has_data = _row_points(row)
            result.append(
                self._record(
                    layer=layer,
                    row_index=row_index,
                    frame_index=frame_index,
                    source_object_id=None,
                    match_distance_m=None,
                    candidate_ids=(),
                    reason="padding" if not has_data else reason,
                )
            )
        return tuple(result)

    def _record(
        self,
        *,
        layer: str,
        row_index: int,
        frame_index: Optional[int],
        source_object_id: Optional[str],
        match_distance_m: Optional[float],
        candidate_ids: tuple[str, ...],
        reason: str,
    ) -> MapObjectMatch:
        return MapObjectMatch(
            layer=str(layer),
            row_index=int(row_index),
            source_object_id=source_object_id,
            source_path=self.source_label,
            frame_index=None if frame_index is None else int(frame_index),
            match_distance_m=(None if match_distance_m is None else float(match_distance_m)),
            candidate_ids=tuple(candidate_ids),
            reason=str(reason),
        )

    def _rank_polyline_matches(
        self,
        points: np.ndarray,
        allowed: Optional[set[str]],
        max_distance: float,
    ) -> list[tuple[str, float]]:
        line = LineString(points)
        if self._centerline_tree is None:
            geometries = tuple(LineString(lane.centerline) for lane in self.lanes)
            self._centerline_geometries = geometries
            self._centerline_tree = STRtree(geometries)
        candidate_indices = self._centerline_tree.query(line.buffer(max_distance))
        if len(candidate_indices) == 0:
            candidate_indices = [self._centerline_tree.nearest(line)]
        ranked: list[tuple[str, float]] = []
        for index_value in candidate_indices:
            lane = self.lanes[int(index_value)]
            if allowed is not None and lane.id not in allowed:
                continue
            score = _polyline_score(points, lane.centerline)
            ranked.append((lane.id, score))
        return sorted(ranked, key=lambda item: (item[1], item[0]))

    def _get_polygon_tree(self) -> STRtree:
        if self._polygon_tree is None:
            geometries = tuple(lane.polygon for lane in self.lanes)
            self._polygon_geometries = geometries
            self._polygon_tree = STRtree(geometries)
        return self._polygon_tree
