"""Phase 4: OCR scale with a heuristic architectural-prior fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon

from src.v1_baseline.ocr import OCRScaleParser
from src.v2_reconstruction.geometry import RectilinearFitter
from src.v2_reconstruction.segmentation import FloorSeamSegmenter, _line_kernel


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class ScaleAnchor:
    kind: str
    pixels: float
    centimeters: float
    pixels_per_cm: float
    bbox: Tuple[int, int, int, int]


@dataclass
class ScaleResult:
    pixels_per_cm: Optional[float]
    source: str
    anchors: List[ScaleAnchor] = field(default_factory=list)
    ocr_pixels_per_cm: Optional[float] = None
    callout_count: int = 0


class ScaleEstimator:
    """Return pixels per centimetre for a flattened floor plan.

    OCR callouts win when a dimension string can be matched to a fitted room.
    Otherwise each doorway, counter depth, and bathtub length is divided into
    its standard real size, and the median of those ratios is the plan scale.
    There is no pretrained anchor detector in this repo, so the anchors are
    measured from wall gaps, the modal edge depth beside a wall, and blue
    tub water.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config["scale"]
        self.preprocessing = self.config["preprocessing"]
        self._config_path = path
        self.result: Optional[ScaleResult] = None

    def estimate(
        self,
        flattened: np.ndarray,
        rooms: Optional[Sequence[Polygon]] = None,
    ) -> Optional[float]:
        """Pixels per centimetre, or ``None`` when neither OCR nor anchors fire."""
        anchors = self.detect_anchors(flattened)
        ocr_scale, callouts = self._ocr_pixels_per_cm(flattened, rooms)
        if ocr_scale is not None:
            source = "ocr"
            value: Optional[float] = ocr_scale
        elif anchors:
            source = "anchors"
            value = float(np.median([anchor.pixels_per_cm for anchor in anchors]))
        else:
            source = "none"
            value = None
        self.result = ScaleResult(
            pixels_per_cm=value,
            source=source,
            anchors=anchors,
            ocr_pixels_per_cm=ocr_scale,
            callout_count=callouts,
        )
        return value

    def detect_anchors(self, flattened: np.ndarray) -> List[ScaleAnchor]:
        """Door widths, one counter depth, and bathtub lengths, each as px/cm."""
        smooth = cv2.bilateralFilter(
            flattened,
            d=int(self.preprocessing["bilateral_d"]),
            sigmaColor=float(self.preprocessing["bilateral_sigma_color"]),
            sigmaSpace=float(self.preprocessing["bilateral_sigma_space"]),
        )
        segmenter = FloorSeamSegmenter(self._config_path)
        footprint = segmenter.detect_footprint(smooth)
        ridges = segmenter.detect_wall_ridges(smooth, footprint)
        anchors = self._doorways(ridges, footprint)
        counter = self._counter_depth(smooth, ridges, footprint)
        if counter is not None:
            anchors.append(counter)
        anchors.extend(self._bathtubs(flattened, footprint))
        return anchors

    def _ocr_pixels_per_cm(
        self,
        flattened: np.ndarray,
        rooms: Optional[Sequence[Polygon]],
    ) -> tuple[Optional[float], int]:
        """Match dimension callouts to room rectangles and convert px/ft to px/cm."""
        parser = OCRScaleParser(self._config_path)
        try:
            boxes = parser.extract_text_boxes(flattened)
            parsed, _ = parser.parse_dimensions(boxes)
        except Exception as exc:
            print(f"OCR scale failed: {exc}")
            return None, 0
        if not parsed or not rooms:
            return None, len(parsed)
        records = parser.associate_text_with_polygons(list(rooms), boxes)
        ratios = [
            ratio
            for record in records
            for ratio in (self._matching_room_scale(record),)
            if ratio is not None
        ]
        if not ratios:
            return None, len(parsed)
        px_per_ft = float(np.median(ratios))
        return px_per_ft / float(self.params["cm_per_foot"]), len(parsed)

    @staticmethod
    def _matching_room_scale(record: dict) -> Optional[float]:
        """Pixels per foot when the room's aspect agrees with the callout.

        A callout attached to the wrong rectangle (an L-wing, or a room merged
        with its closet) produces a scale about twice the true one. Those
        pairs are dropped instead of being averaged in.
        """
        polygon = record.get("polygon")
        dimensions = record.get("dimensions") or []
        if not isinstance(polygon, Polygon) or polygon.is_empty or not dimensions:
            return None
        min_x, min_y, max_x, max_y = polygon.bounds
        pixel_width = float(max_x - min_x)
        pixel_height = float(max_y - min_y)
        if pixel_width <= 1 or pixel_height <= 1:
            return None
        pixel_aspect = pixel_width / pixel_height
        ratios: list[float] = []
        for dimension in dimensions:
            width_ft = float(dimension.get("width_ft") or 0.0)
            height_ft = float(dimension.get("height_ft") or 0.0)
            if width_ft <= 0 or height_ft <= 0:
                continue
            direct = abs(pixel_aspect - (width_ft / height_ft))
            swapped = abs(pixel_aspect - (height_ft / width_ft))
            if min(direct, swapped) > 0.25:
                continue
            if direct <= swapped:
                ratios.append(((pixel_width / width_ft) + (pixel_height / height_ft)) / 2.0)
            else:
                ratios.append(((pixel_width / height_ft) + (pixel_height / width_ft)) / 2.0)
        if not ratios:
            return None
        return float(np.median(ratios))

    def _doorways(self, ridges: np.ndarray, footprint: np.ndarray) -> List[ScaleAnchor]:
        """Openings bridged along a wall, measured across the gap."""
        gap_px = int(self.config["floor_segmentation"]["door_gap_px"])
        bridged = ridges.copy()
        for angle in (0.0, 90.0):
            bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, _line_kernel(gap_px, angle))
        gaps = cv2.bitwise_and(cv2.subtract(bridged, ridges), footprint)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=8)
        min_px = int(self.params["doorway_min_px"])
        max_px = int(self.params["doorway_max_px"])
        thick = int(self.params["doorway_thickness_max_px"])
        cm = float(self.params["doorway_cm"])
        anchors: list[ScaleAnchor] = []
        for index in range(1, count):
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            span = max(width, height)
            thickness = min(width, height)
            if thickness > thick or not min_px <= span <= max_px:
                continue
            if not self._walled_on_both_ends(ridges, x, y, width, height):
                continue
            if not self._floor_on_both_sides(ridges, footprint, x, y, width, height):
                continue
            if not self._wall_continues(ridges, x, y, width, height, int(self.params["doorway_wall_run_px"])):
                continue
            anchors.append(self._anchor("doorway", float(span), cm, (x, y, width, height)))
        return anchors

    def _counter_depth(
        self,
        image: np.ndarray,
        ridges: np.ndarray,
        footprint: np.ndarray,
    ) -> Optional[ScaleAnchor]:
        """Modal distance from a wall to the parallel inner edge of a counter.

        One peak describes the typical counter depth in the render. Baseboard
        shadows sit closer to the wall than ``counter_depth_min_px`` and are
        ignored.
        """
        free = cv2.bitwise_and(footprint, cv2.bitwise_not(ridges))
        distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges[(footprint == 0) | (ridges > 0)] = 0
        low = int(self.params["counter_depth_min_px"])
        high = int(self.params["counter_depth_max_px"])
        density: list[float] = []
        for depth in range(low, high + 1):
            band = (distance >= depth - 1.0) & (distance < depth + 1.0)
            band_px = int(np.count_nonzero(band))
            if band_px < 50:
                density.append(0.0)
                continue
            density.append(float(np.count_nonzero(band & (edges > 0))) / float(band_px))
        if not density or max(density) <= 0:
            return None
        peak_index = int(np.argmax(density))
        # The first and last bins are the search boundary, not a counter edge.
        if peak_index <= 1 or peak_index >= len(density) - 2:
            return None
        if density[peak_index] < 1.8 * float(np.median(density)):
            return None
        peak = low + peak_index
        return self._anchor("countertop", float(peak), float(self.params["countertop_cm"]), (0, 0, peak, 0))

    def _bathtubs(self, image: np.ndarray, footprint: np.ndarray) -> List[ScaleAnchor]:
        """Blue water inside a tub. The long side is the 150 cm length."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(
            hsv,
            (int(self.params["tub_hue_min"]), int(self.params["tub_sat_min"]), 40),
            (int(self.params["tub_hue_max"]), 255, 255),
        )
        blue = cv2.bitwise_and(blue, footprint)
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
        min_area = int(self.params["tub_min_area_px"])
        max_area = int(self.params["tub_max_area_px"])
        min_aspect = float(self.params["tub_min_aspect"])
        cm = float(self.params["bathtub_cm"])
        best: Optional[ScaleAnchor] = None
        best_area = 0
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if not min_area <= area <= max_area or area <= best_area:
                continue
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if min(width, height) == 0 or max(width, height) / min(width, height) < min_aspect:
                continue
            best = self._anchor("bathtub", float(max(width, height)), cm, (x, y, width, height))
            best_area = area
        return [best] if best is not None else []

    @staticmethod
    def _walled_on_both_ends(ridges: np.ndarray, x: int, y: int, width: int, height: int) -> bool:
        image_h, image_w = ridges.shape[:2]
        if width >= height:
            left = ridges[y : y + height, max(0, x - 4) : x]
            right = ridges[y : y + height, x + width : min(image_w, x + width + 4)]
        else:
            left = ridges[max(0, y - 4) : y, x : x + width]
            right = ridges[y + height : min(image_h, y + height + 4), x : x + width]
        return bool(left.size and right.size and np.count_nonzero(left) and np.count_nonzero(right))

    @staticmethod
    def _floor_on_both_sides(
        ridges: np.ndarray,
        footprint: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool:
        """A doorway opens onto floor on both sides of the wall, not onto text or a fixture."""
        image_h, image_w = ridges.shape[:2]
        band = 8

        def open_floor(y0: int, y1: int, x0: int, x1: int) -> bool:
            y0, y1 = max(0, y0), min(image_h, y1)
            x0, x1 = max(0, x0), min(image_w, x1)
            if y1 <= y0 or x1 <= x0:
                return False
            region = footprint[y0:y1, x0:x1] > 0
            blocked = ridges[y0:y1, x0:x1] > 0
            if region.size == 0:
                return False
            return float(np.count_nonzero(region & ~blocked)) / float(region.size) >= 0.6

        if width >= height:
            return open_floor(y - band, y, x, x + width) and open_floor(y + height, y + height + band, x, x + width)
        return open_floor(y, y + height, x - band, x) and open_floor(y, y + height, x + width, x + width + band)

    @staticmethod
    def _wall_continues(ridges: np.ndarray, x: int, y: int, width: int, height: int, run_px: int) -> bool:
        """Both jambs must sit on a wall, so a gap between cushions is not a door.

        The ridge is often broken, so this counts wall pixels along the next
        ``run_px`` steps instead of requiring an unbroken line.
        """
        image_h, image_w = ridges.shape[:2]

        def occupancy(x0: int, y0: int, dx: int, dy: int) -> float:
            hits = 0
            checked = 0
            for step in range(1, run_px + 1):
                cx, cy = x0 + dx * step, y0 + dy * step
                if not (0 <= cy < image_h and 0 <= cx < image_w):
                    break
                checked += 1
                window = ridges[
                    max(0, cy - 3) : min(image_h, cy + 4),
                    max(0, cx - 3) : min(image_w, cx + 4),
                ]
                if np.count_nonzero(window):
                    hits += 1
            return hits / checked if checked >= run_px // 2 else 0.0

        if width >= height:
            mid_y = min(image_h - 1, y + max(height // 2, 0))
            left = occupancy(x - 1, mid_y, -1, 0)
            right = occupancy(x + width, mid_y, 1, 0)
        else:
            mid_x = min(image_w - 1, x + max(width // 2, 0))
            left = occupancy(mid_x, y - 1, 0, -1)
            right = occupancy(mid_x, y + height, 0, 1)
        return left >= 0.5 and right >= 0.5

    @staticmethod
    def _anchor(kind: str, pixels: float, centimeters: float, bbox: Tuple[int, int, int, int]) -> ScaleAnchor:
        return ScaleAnchor(
            kind=kind,
            pixels=float(pixels),
            centimeters=float(centimeters),
            pixels_per_cm=float(pixels) / float(centimeters),
            bbox=bbox,
        )


def rooms_from_rects(rects: Sequence[dict]) -> List[Polygon]:
    """Phase 3 rectangles as polygons, for OCR callout association."""
    return [RectilinearFitter.as_polygon(rect) for rect in rects]
