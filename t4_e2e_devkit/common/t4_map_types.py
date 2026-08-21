"""What a parsed T4 Lanelet2 map is made of.

Two frozen records, and no behaviour beyond what they can answer about
themselves.  They sit apart from both the parser that builds them and the API
that queries them, because all three layers name them and none of them owns
them: :mod:`t4_map_parse` produces these, :mod:`t4_map_geometry` measures them,
:class:`~t4_e2e_devkit.common.t4_map.T4MapAPI` serves them.

Both stay importable from ``common.t4_map``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class T4Lanelet:
    """A lanelet reconstructed from one Lanelet2 ``lanelet`` relation."""

    id: str
    left_boundary_id: str
    right_boundary_id: str
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    centerline: np.ndarray
    polygon: Polygon
    speed_limit_mps: Optional[float]
    tags: Mapping[str, str]
    incoming_ids: tuple[str, ...] = ()
    outgoing_ids: tuple[str, ...] = ()
    lanelet_type: str = "lanelet"
    turn_direction: str = "unknown"
    regulatory_element_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class T4MapObject:
    """A source Lanelet2 object other than the lanelet graph.

    ``id`` is the source way/relation/node ID.  The object keeps the original
    tags and geometry so downstream code can use a semantic layer without
    depending on the compact model tensor.  Geometry is expressed in the map
    coordinate frame used by the T4 OSM export.
    """

    id: str
    object_type: str
    geometry: BaseGeometry
    tags: Mapping[str, str]
    source_kind: str
    member_ids: tuple[str, ...] = ()

    @property
    def polygon(self) -> Optional[BaseGeometry]:
        """Return the polygon geometry when this object has one."""

        return self.geometry if self.geometry.geom_type in {"Polygon", "MultiPolygon"} else None

    @property
    def is_area(self) -> bool:
        """:return: whether the object is represented by an areal geometry."""

        return self.geometry.geom_type in {"Polygon", "MultiPolygon"}
