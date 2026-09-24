"""Phase 4: OCR scale with a heuristic architectural-prior fallback."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ScaleEstimator:
    """Return pixels-per-cm from OCR, or the median of standard-anchor ratios."""

    def estimate(self, flattened: np.ndarray) -> Optional[float]:
        raise NotImplementedError("Phase 4 scale estimation is not implemented yet.")
