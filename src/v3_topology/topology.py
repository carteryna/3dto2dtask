"""Phase 2: partition the walkable footprint into shared-edge room cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

Seed = Tuple[int, int]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class TopologyDebug:
    seeds: List[Seed]
    dropped_seeds: int = 0
    unseeded_components: int = 0
    notes: List[str] = field(default_factory=list)


class TopologyPartitioner:
    """Marker watershed so every floor pixel belongs to exactly one room.

    Seeds are the centroids of the Phase 1 floor masks. When those masks are
    passed in, each mask is the watershed marker, so a room keeps the pixels
    Phase 1 already gave it and only the gaps in the footprint are flooded.
    Point markers are the fallback. Boundary pixels left unlabeled are given
    to the nearest seed in the same component. Specks with no seed are
    absorbed into a neighbour. A large unseeded piece of floor stays its own
    cell. The returned masks are disjoint and their union is the footprint.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config["v3_topology"]
        self.debug: Optional[TopologyDebug] = None

    def seeds(self, masks: Sequence[np.ndarray]) -> List[Seed]:
        """One centroid per floor mask, snapped onto the mask when the centroid falls in a hole."""
        points: list[Seed] = []
        for mask in masks:
            point = self._centroid(mask)
            if point is not None:
                points.append(point)
        return points

    def partition(
        self,
        footprint: np.ndarray,
        seeds: Sequence[Seed],
        masks: Sequence[np.ndarray] | None = None,
    ) -> List[np.ndarray]:
        """Return one binary cell (uint8, 0/255) per room. The cells tile ``footprint``."""
        domain = np.where(footprint > 0, 255, 0).astype(np.uint8)
        height, width = domain.shape[:2]
        use_masks = masks is not None and len(masks) == len(seeds)
        if use_masks:
            placed, dropped = self._snap_seeds(domain, seeds)
        else:
            placed, dropped = self._place_seeds(domain, seeds)
            masks = None
        if not np.any(domain):
            self.debug = TopologyDebug(seeds=placed, dropped_seeds=dropped)
            return []

        labels = np.zeros((height, width), dtype=np.int32)
        if placed:
            labels = self._watershed(domain, placed, masks if use_masks else None)
        owned = np.zeros((height, width), dtype=np.int32)
        for index in range(1, len(placed) + 1):
            owned[labels == index] = index
        self._claim_boundaries(owned, domain, placed)
        self._absorb_dust(owned, domain)
        extra = self._claim_unseeded(owned, domain, len(placed))

        cells: list[np.ndarray] = []
        for index in range(1, len(placed) + extra + 1):
            cell = np.where(owned == index, 255, 0).astype(np.uint8)
            if np.any(cell):
                cells.append(cell)
        self.debug = TopologyDebug(
            seeds=placed,
            dropped_seeds=dropped,
            unseeded_components=extra,
        )
        return cells

    def _watershed(
        self,
        domain: np.ndarray,
        seeds: Sequence[Seed],
        masks: Sequence[np.ndarray] | None,
    ) -> np.ndarray:
        distance = cv2.distanceTransform(domain, cv2.DIST_L2, 5)
        peak = float(distance.max()) if np.any(distance) else 1.0
        relief = np.clip((peak - distance) * (255.0 / max(peak, 1.0)), 0, 255).astype(np.uint8)
        relief[domain == 0] = 255
        markers = np.zeros(domain.shape, dtype=np.int32)
        barrier = len(seeds) + 1
        markers[domain == 0] = barrier
        if masks is not None and len(masks) == len(seeds):
            for index, mask in enumerate(masks, start=1):
                markers[(mask > 0) & (domain > 0) & (markers == 0)] = int(index)
        radius = max(int(self.params["marker_radius_px"]), 1)
        for index, (x, y) in enumerate(seeds, start=1):
            if markers[y, x] == 0:
                cv2.circle(markers, (x, y), radius, int(index), thickness=-1)
        markers[domain == 0] = barrier
        cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)
        return markers

    def _absorb_dust(self, owned: np.ndarray, domain: np.ndarray) -> None:
        """Fold unseeded specks into the nearest room so they do not become cells."""
        minimum = int(self.params["min_cell_area_px"])
        pending = np.where((domain > 0) & (owned == 0), 255, 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(pending, connectivity=8)
        dust = np.zeros(domain.shape, dtype=bool)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) < minimum:
                dust[labels == index] = True
        if not np.any(dust) or not np.any(owned > 0):
            return
        sources = np.where(owned > 0, 0, 255).astype(np.uint8)
        _, nearest = cv2.distanceTransformWithLabels(
            sources,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        lookup = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
        lookup[nearest[owned > 0]] = owned[owned > 0]
        owned[dust] = lookup[nearest[dust]]

    def _claim_boundaries(self, owned: np.ndarray, domain: np.ndarray, seeds: Sequence[Seed]) -> None:
        """Give watershed boundary pixels to the nearest seed in the same component."""
        if not seeds:
            return
        pending = (domain > 0) & (owned == 0) & ~self._unseeded_component_mask(owned, domain)
        ys, xs = np.where(pending)
        if len(xs) == 0:
            return
        seed_x = np.array([point[0] for point in seeds], dtype=np.float32)
        seed_y = np.array([point[1] for point in seeds], dtype=np.float32)
        component = self._component_ids(domain)
        seed_component = np.array([component[point[1], point[0]] for point in seeds], dtype=np.int32)
        pixel_component = component[ys, xs]
        distance = (xs[:, None] - seed_x[None, :]) ** 2 + (ys[:, None] - seed_y[None, :]) ** 2
        distance[seed_component[None, :] != pixel_component[:, None]] = np.inf
        nearest = np.argmin(distance, axis=1)
        finite = np.isfinite(distance[np.arange(len(xs)), nearest])
        owned[ys[finite], xs[finite]] = nearest[finite] + 1

    def _claim_unseeded(self, owned: np.ndarray, domain: np.ndarray, seed_count: int) -> int:
        """A disconnected piece of floor with no seed is still one cell."""
        pending = np.where((domain > 0) & (owned == 0), 255, 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(pending, connectivity=8)
        added = 0
        minimum = int(self.params["min_cell_area_px"])
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) < minimum:
                continue
            added += 1
            owned[labels == index] = seed_count + added
        return added

    def _snap_seeds(self, domain: np.ndarray, seeds: Sequence[Seed]) -> tuple[list[Seed], int]:
        """Keep one seed per mask, moved onto the footprint when a centroid misses it."""
        height, width = domain.shape[:2]
        placed: list[Seed] = []
        dropped = 0
        for x, y in seeds:
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            if domain[y, x] == 0:
                snapped = self._nearest_on(domain, x, y)
                if snapped is None:
                    dropped += 1
                    placed.append((x, y))
                    continue
                x, y = snapped
            placed.append((x, y))
        return placed, dropped

    def _place_seeds(self, domain: np.ndarray, seeds: Sequence[Seed]) -> tuple[list[Seed], int]:
        height, width = domain.shape[:2]
        placed: list[Seed] = []
        dropped = 0
        minimum = max(int(self.params["marker_radius_px"]), 1) * 2
        for x, y in seeds:
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            if domain[y, x] == 0:
                snapped = self._nearest_on(domain, x, y)
                if snapped is None:
                    dropped += 1
                    continue
                x, y = snapped
            if any((x - other_x) ** 2 + (y - other_y) ** 2 <= minimum ** 2 for other_x, other_y in placed):
                dropped += 1
                continue
            placed.append((x, y))
        return placed, dropped

    @staticmethod
    def _unseeded_component_mask(owned: np.ndarray, domain: np.ndarray) -> np.ndarray:
        """Footprint components that contain no owned pixel."""
        count, labels = cv2.connectedComponents(domain, connectivity=8)
        unseeded = np.zeros(domain.shape, dtype=bool)
        for index in range(1, count):
            component = labels == index
            if not np.any(owned[component] > 0):
                unseeded[component] = True
        return unseeded

    @staticmethod
    def _component_ids(domain: np.ndarray) -> np.ndarray:
        _, labels = cv2.connectedComponents(domain, connectivity=8)
        return labels

    @staticmethod
    def _centroid(mask: np.ndarray) -> Optional[Seed]:
        if mask is None or not np.any(mask):
            return None
        moments = cv2.moments(np.where(mask > 0, 255, 0).astype(np.uint8))
        if moments["m00"] == 0:
            return None
        x = int(round(moments["m10"] / moments["m00"]))
        y = int(round(moments["m01"] / moments["m00"]))
        height, width = mask.shape[:2]
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if mask[y, x] == 0:
            snapped = TopologyPartitioner._nearest_on(mask, x, y)
            if snapped is None:
                return None
            return snapped
        return x, y

    @staticmethod
    def _nearest_on(mask: np.ndarray, x: int, y: int) -> Optional[Seed]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        choice = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        return int(xs[choice]), int(ys[choice])
