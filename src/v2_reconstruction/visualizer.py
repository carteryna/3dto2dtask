"""Phase 5: render a synthetic 2D CAD-style blueprint."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class BlueprintRenderer:
    """Draw rooms as black walls on a white canvas with labels and areas."""

    def render(self, shape: tuple[int, int], rooms: Sequence[dict]) -> np.ndarray:
        raise NotImplementedError("Phase 5 blueprint rendering is not implemented yet.")
