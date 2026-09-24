"""Phase 4: scale from the unit area named in the filename."""

from __future__ import annotations

from typing import Optional, Sequence

from shapely.geometry import Polygon


class AreaScaleEstimator:
    """Pixels per foot from filename square footage, with a fixture fallback."""

    def estimate(self, polygons: Sequence[Polygon], filename: str) -> Optional[float]:
        raise NotImplementedError("Phase 4 area scale is not implemented yet.")
