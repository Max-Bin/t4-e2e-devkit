"""Move a window's numpy arrays onto the device as the metric tensors.

The extraction itself is device-independent and lives in
:mod:`t4_e2e_devkit.evaluation.window_arrays`; what is left here is the upload,
which is the only part that needs to know about a device or a dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from t4_e2e_devkit.evaluation.gpu.collisions import TrackTensors


@dataclass
class MapTensors:
    """Packed map polygons plus the index groups the area masks consume."""

    edge_starts: torch.Tensor  # [L, E, 2]
    edge_ends: torch.Tensor  # [L, E, 2]
    drivable_area_indices: torch.Tensor  # ROADBLOCK/INTERSECTION/... group
    lane_indices: torch.Tensor  # exact LANE polygons
    on_route_lane_indices: torch.Tensor
    intersection_indices: torch.Tensor  # for the TTC rear-axle layer test
    # [L, 4] = (min_x, min_y, max_x, max_y) per polygon, for the exact
    # reach-bbox prefilter: a polygon can only influence containment of a
    # point it contains, so polygons whose bbox misses the tested points'
    # bbox are droppable with no effect on any mask.
    bboxes: torch.Tensor | None = None

    def restricted_to(self, keep: torch.Tensor) -> "MapTensors":
        """A copy containing only polygons selected by boolean ``keep`` [L]."""

        kept_index = keep.nonzero(as_tuple=False).squeeze(-1)
        remap = torch.full((keep.shape[0],), -1, dtype=torch.long, device=keep.device)
        remap[kept_index] = torch.arange(kept_index.shape[0], device=keep.device, dtype=torch.long)

        def _group(indices: torch.Tensor) -> torch.Tensor:
            if indices.numel() == 0:
                return indices
            mapped = remap[indices]
            return mapped[mapped >= 0]

        return MapTensors(
            edge_starts=self.edge_starts[kept_index],
            edge_ends=self.edge_ends[kept_index],
            drivable_area_indices=_group(self.drivable_area_indices),
            lane_indices=_group(self.lane_indices),
            on_route_lane_indices=_group(self.on_route_lane_indices),
            intersection_indices=_group(self.intersection_indices),
            bboxes=None if self.bboxes is None else self.bboxes[kept_index],
        )


def track_tensors_from_arrays(
    arrays: dict,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TrackTensors:
    def _tensor(key: str, cast=None):
        value = arrays[key]
        tensor = value if torch.is_tensor(value) else torch.from_numpy(value)
        tensor = tensor.to(device=device)
        return tensor.to(dtype=dtype) if cast else tensor

    return TrackTensors(
        corners=_tensor("track_corners", cast=True),
        valid=_tensor("track_valid"),
        is_agent_instance=_tensor("track_is_agent_instance"),
        velocity=_tensor("track_velocity", cast=True),
        is_agent_type=_tensor("track_is_agent_type"),
        bboxes=_tensor("track_bboxes", cast=True),
    )


def window_scene_from_arrays(arrays: dict, device, dtype):
    """Build the metric-stage inputs from (possibly device-resident) arrays."""

    from t4_e2e_devkit.evaluation.gpu.geometry import TorchPolyline

    def _tensor(key: str, cast: bool = False, index: bool = False):
        value = arrays[key]
        tensor = value if torch.is_tensor(value) else torch.from_numpy(value)
        tensor = tensor.to(device=device)
        if index:
            return tensor.to(dtype=torch.long)
        return tensor.to(dtype=dtype) if cast else tensor

    rings = _tensor("map_rings", cast=True)
    map_tensors = MapTensors(
        edge_starts=rings[:, :-1],
        edge_ends=rings[:, 1:],
        drivable_area_indices=_tensor("map_drivable_idx", index=True),
        lane_indices=_tensor("map_lane_idx", index=True),
        on_route_lane_indices=_tensor("map_on_route_idx", index=True),
        intersection_indices=_tensor("map_intersection_idx", index=True),
        bboxes=_tensor("map_bboxes", cast=True),
    )
    centerline_heading = arrays.get("centerline_heading")
    centerline = TorchPolyline(
        _tensor("centerline", cast=True),
        None if centerline_heading is None else _tensor_value(centerline_heading, device, dtype),
    )
    tracks = track_tensors_from_arrays(arrays, device, dtype)
    return tracks, map_tensors, centerline


def _tensor_value(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Move one optional scene-array field to the oracle device."""

    tensor = value if torch.is_tensor(value) else torch.from_numpy(value)
    return tensor.to(device=device, dtype=dtype)
