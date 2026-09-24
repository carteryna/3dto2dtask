"""Phase 2: partition the walkable footprint into shared-edge room cells."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Seed = Tuple[int, int]


class TopologyPartitioner:
    """Marker watershed so every floor pixel belongs to exactly one room."""

    def seeds(self, masks: Sequence[np.ndarray]) -> List[Seed]:
        raise NotImplementedError("Phase 2 room seeds are not implemented yet.")

    def partition(self, footprint: np.ndarray, seeds: Sequence[Seed]) -> List[np.ndarray]:
        raise NotImplementedError("Phase 2 watershed partition is not implemented yet.")
