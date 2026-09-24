"""Phase 3: orthogonalize the shared-wall line graph."""

from __future__ import annotations

from typing import List

import numpy as np
from shapely.geometry import Polygon


class ManhattanLineGraph:
    """Turn watershed boundaries into corner-matched orthogonal room polygons."""

    def extract_segments(self, cells: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError("Phase 3 boundary extraction is not implemented yet.")

    def orthogonalize(self, segments: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError("Phase 3 Manhattan orthogonalization is not implemented yet.")

    def polygons(self, segments: List[np.ndarray]) -> List[Polygon]:
        raise NotImplementedError("Phase 3 polygon reconstruction is not implemented yet.")
