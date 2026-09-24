"""Phase 3: fit rectilinear rooms and snap them to a shared grid."""

from __future__ import annotations

from typing import List

import numpy as np
from shapely.geometry import Polygon


class RectilinearFitter:
    """Turn floor masks into grid-snapped orthogonal room rectangles."""

    def fit(self, masks: List[np.ndarray]) -> List[Polygon]:
        raise NotImplementedError("Phase 3 rectilinear fitting is not implemented yet.")
