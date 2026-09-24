"""Phase 5: draw the orthogonal wall graph as a blueprint."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry import Polygon


class TopologyBlueprint:
    """Stroke the shared walls and label each room at its centroid."""

    def render(self, shape: tuple[int, int], polygons: Sequence[Polygon]) -> np.ndarray:
        raise NotImplementedError("Phase 5 blueprint rendering is not implemented yet.")
