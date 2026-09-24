"""Phase 2: extract walkable floor by targeting floor-wall base seams."""

from __future__ import annotations

from typing import List

import numpy as np


class FloorSeamSegmenter:
    """Run FastSAM on the flattened image and keep floor masks bounded by wall bases."""

    def segment(self, flattened: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError("Phase 2 floor-seam segmentation is not implemented yet.")
