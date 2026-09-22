"""OCR scale parsing and spatial fusion of text with room polygons."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from shapely.geometry import Point, Polygon

TextBox = Dict[str, object]
RoomRecord = Dict[str, object]

_FEET_INCHES = re.compile(
    r"(\d{1,3})\s*'\s*(\d{1,2})?\s*(?:\"|in(?:ches)?)?"
    r"\s*x\s*"
    r"(\d{1,3})\s*'\s*(\d{1,2})?\s*(?:\"|in(?:ches)?)?",
    re.IGNORECASE,
)
_SINGLE_FEET = re.compile(
    r"^\s*(?:x\s*)?(\d{1,3})\s*'\s*(\d{1,2})?\s*\"?\s*$",
    re.IGNORECASE,
)
_METERS = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*m(?:eters?)?\s*x\s*(\d+(?:[.,]\d+)?)\s*m(?:eters?)?",
    re.IGNORECASE,
)
_SQUARE_METERS = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*m\s*(?:²|2|sq)",
    re.IGNORECASE,
)
_QUOTE_MAP = str.maketrans({
    "’": "'",
    "‘": "'",
    "′": "'",
    "´": "'",
    "`": "'",
    "″": '"',
    "“": '"',
    "”": '"',
    "×": "x",
    "X": "x",
    "*": "x",
})
_LABEL_STOPWORDS = (
    "rendering",
    "artist",
    "conception",
    "feature",
    "material",
    "finish",
    "layout",
    "3dplans",
    "www",
    "http",
    "general reference",
)
_METERS_TO_FEET = 3.280839895
_SQM_TO_SQFT = 10.7639104167


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class OCRScaleParser:
    """Read floor-plan callouts and attach them to regularized room polygons.

    Dimension strings such as ``13'3" x 13'3"`` become real widths, heights, and
    areas. The pixel-to-foot scale is the detection-box width divided by the
    parsed width. Plans with no dimension text stay in fallback mode and keep
    relative pixel areas only.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.pipeline = self.config["pipeline"]
        self.ocr = self.config["ocr"]
        self._reader = None
        self.fallback_mode = True
        self.scale_ratio: Optional[float] = None

    def extract_text_boxes(self, image: np.ndarray) -> List[TextBox]:
        """Run EasyOCR and return bbox, text, and confidence for each detection."""
        if not bool(self.ocr.get("enabled", True)):
            return []
        try:
            raw_rows = self._get_reader().readtext(image)
        except Exception as exc:
            print(f"EasyOCR failed: {exc}")
            return []

        boxes: list[TextBox] = []
        for row in raw_rows:
            try:
                bbox, text, confidence = row[0], str(row[1]).strip(), float(row[2])
            except (TypeError, ValueError, IndexError):
                continue
            if not text:
                continue
            points = [[float(point[0]), float(point[1])] for point in bbox]
            boxes.append({"bbox": points, "text": text, "confidence": confidence})
        return boxes

    def parse_dimensions(self, text_boxes: List[TextBox]) -> Tuple[List[TextBox], Optional[float]]:
        """Parse dimension callouts and return them with a pixel-per-foot scale.

        The scale is the median of ``bbox width (px) / parsed width (ft)`` across
        successful callouts. It is ``None`` when the render has no usable
        dimensions, which flags fallback mode.
        """
        parsed: list[TextBox] = []
        scales: list[float] = []
        self._pair_split_dimensions(text_boxes)
        for box in text_boxes:
            try:
                measurement = self._parse_measurement(str(box.get("text", "")))
            except (re.error, TypeError, ValueError):
                measurement = None
            box.pop("width_ft", None)
            box.pop("height_ft", None)
            box.pop("area_sqft", None)
            box.pop("unit", None)
            box.pop("scale_ratio", None)
            if box.get("paired") or measurement is None:
                continue
            box.update(measurement)
            width_px = self._bbox_size(box.get("bbox", []))[0]
            width_ft = float(measurement["width_ft"])
            if width_px > 0 and width_ft > 0:
                box["scale_ratio"] = width_px / width_ft
                scales.append(float(box["scale_ratio"]))
            parsed.append(box)

        self.scale_ratio = float(np.median(scales)) if scales else None
        self.fallback_mode = self.scale_ratio is None
        return parsed, self.scale_ratio

    def associate_text_with_polygons(
        self,
        polygons: List[Polygon],
        text_boxes: List[TextBox],
    ) -> List[RoomRecord]:
        """Attach nearby labels and dimension callouts to each room polygon.

        A text box is assigned to the room whose centroid is closest, provided
        the box lies inside the polygon or within a small margin of its edge.
        Matched dimension callouts also produce a plan scale from the room
        bounding-box width, used to estimate square footage for every room.
        """
        if not any("width_ft" in box or "area_sqft" in box for box in text_boxes):
            self.parse_dimensions(text_boxes)

        total_area = sum(float(polygon.area) for polygon in polygons if not polygon.is_empty) or 1.0
        rooms: list[RoomRecord] = []
        for index, polygon in enumerate(polygons):
            rooms.append(
                {
                    "polygon_index": index,
                    "polygon": polygon,
                    "area_px": 0.0 if polygon.is_empty else float(polygon.area),
                    "relative_area_pct": 0.0 if polygon.is_empty else 100.0 * float(polygon.area) / total_area,
                    "labels": [],
                    "dimensions": [],
                    "nominal_area_sqft": None,
                    "estimated_area_sqft": None,
                    "scale_ratio": self.scale_ratio,
                    "fallback_mode": self.fallback_mode,
                }
            )

        for box in text_boxes:
            centroid = self._bbox_centroid(box.get("bbox", []))
            if centroid is None:
                continue
            target = self._nearest_room(rooms, centroid)
            if target is None:
                continue
            is_dimension = "width_ft" in box or "area_sqft" in box
            if is_dimension:
                target = self._room_below_callout(rooms, centroid) or target
                target["dimensions"].append(box)
            elif self._is_room_label(str(box.get("text", ""))):
                target["labels"].append(str(box["text"]))

        plan_scales = [ratio for ratio in (self._room_scale(room) for room in rooms) if ratio is not None]
        plan_scale = float(np.median(plan_scales)) if plan_scales else None
        for room in rooms:
            dimensions = room["dimensions"]
            if dimensions:
                nominal = next((float(item["area_sqft"]) for item in dimensions if item.get("area_sqft")), None)
                room["nominal_area_sqft"] = nominal
            if plan_scale is not None and float(room["area_px"]) > 0:
                room["estimated_area_sqft"] = float(room["area_px"]) / (plan_scale ** 2)
                room["plan_scale_ratio"] = plan_scale
            else:
                room["plan_scale_ratio"] = None
        return rooms

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        import easyocr

        languages = list(self.ocr.get("languages") or ["en"])
        device = str(self.pipeline.get("device", "cpu")).lower()
        storage = _project_root() / "models" / "easyocr"
        storage.mkdir(parents=True, exist_ok=True)
        self._reader = easyocr.Reader(
            languages,
            gpu=device not in {"cpu", "mps"},
            model_storage_directory=str(storage),
            download_enabled=True,
            verbose=False,
        )
        return self._reader

    def _pair_split_dimensions(self, text_boxes: List[TextBox]) -> None:
        """Join side-by-side readings such as ``11'9"`` and ``10'9"`` into one callout.

        EasyOCR sometimes drops the ``x`` and returns the two sides as separate
        boxes. A lone measurement is not a scale by itself.
        """
        singles: list[tuple[TextBox, re.Match[str]]] = []
        for box in text_boxes:
            if box.get("paired"):
                continue
            try:
                normalized = str(box.get("text", "")).translate(_QUOTE_MAP)
                normalized = re.sub(r"\s+", " ", normalized).strip()
                match = _SINGLE_FEET.fullmatch(normalized)
            except re.error:
                match = None
            if match is not None and _FEET_INCHES.search(normalized) is None:
                singles.append((box, match))

        singles.sort(key=lambda item: self._bbox_centroid(item[0].get("bbox", [])) or (0.0, 0.0))
        used: set[int] = set()
        for left_index, (left_box, left_match) in enumerate(singles):
            if left_index in used:
                continue
            left_center = self._bbox_centroid(left_box.get("bbox", []))
            left_width, left_height = self._bbox_size(left_box.get("bbox", []))
            if left_center is None:
                continue
            partner: tuple[int, TextBox, re.Match[str]] | None = None
            best_gap = float("inf")
            for right_index in range(left_index + 1, len(singles)):
                if right_index in used:
                    continue
                right_box, right_match = singles[right_index]
                right_center = self._bbox_centroid(right_box.get("bbox", []))
                if right_center is None:
                    continue
                vertical_gap = abs(right_center[1] - left_center[1])
                horizontal_gap = right_center[0] - left_center[0]
                row_tolerance = max(left_height, self._bbox_size(right_box.get("bbox", []))[1], 12.0)
                if vertical_gap > row_tolerance or horizontal_gap <= 0 or horizontal_gap > max(140.0, left_width * 2.5):
                    continue
                if horizontal_gap < best_gap:
                    best_gap = horizontal_gap
                    partner = (right_index, right_box, right_match)
            if partner is None:
                continue
            right_index, right_box, right_match = partner
            used.add(left_index)
            used.add(right_index)
            left_box["text"] = (
                f"{left_match.group(1)}'{left_match.group(2) or 0}\" x "
                f"{right_match.group(1)}'{right_match.group(2) or 0}\""
            )
            left_box["bbox"] = self._union_bbox(left_box.get("bbox", []), right_box.get("bbox", []))
            left_box["confidence"] = min(float(left_box.get("confidence", 0.0)), float(right_box.get("confidence", 0.0)))
            right_box["paired"] = True

    @staticmethod
    def _union_bbox(first: object, second: object) -> list[list[float]]:
        points = []
        for bbox in (first, second):
            if isinstance(bbox, list):
                points.extend([float(point[0]), float(point[1])] for point in bbox)
        if not points:
            return []
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]

    def _parse_measurement(self, text: str) -> Optional[dict]:
        normalized = text.translate(_QUOTE_MAP)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None

        feet_match = _FEET_INCHES.search(normalized)
        if feet_match:
            width_ft = self._feet_and_inches(feet_match.group(1), feet_match.group(2))
            height_ft = self._feet_and_inches(feet_match.group(3), feet_match.group(4))
            return {
                "width_ft": width_ft,
                "height_ft": height_ft,
                "area_sqft": width_ft * height_ft,
                "unit": "ft",
            }

        meter_match = _METERS.search(normalized)
        if meter_match:
            width_m = float(meter_match.group(1).replace(",", "."))
            height_m = float(meter_match.group(2).replace(",", "."))
            width_ft = width_m * _METERS_TO_FEET
            height_ft = height_m * _METERS_TO_FEET
            return {
                "width_ft": width_ft,
                "height_ft": height_ft,
                "area_sqft": width_ft * height_ft,
                "unit": "m",
            }

        square_match = _SQUARE_METERS.search(normalized)
        if square_match:
            square_meters = float(square_match.group(1).replace(",", "."))
            return {
                "width_ft": 0.0,
                "height_ft": 0.0,
                "area_sqft": square_meters * _SQM_TO_SQFT,
                "unit": "m2",
            }
        return None

    @staticmethod
    def _feet_and_inches(feet: str, inches: str | None) -> float:
        return float(feet) + float(inches or 0) / 12.0

    @staticmethod
    def _is_room_label(text: str) -> bool:
        cleaned = text.strip()
        lowered = cleaned.lower()
        if any(stopword in lowered for stopword in _LABEL_STOPWORDS):
            return False
        if len(cleaned) < 2 or len(cleaned) > 24:
            return False
        letters = sum(character.isalpha() for character in cleaned)
        return letters >= 3 and letters / len(cleaned) >= 0.6

    def _nearest_room(self, rooms: Sequence[RoomRecord], centroid: tuple[float, float]) -> Optional[RoomRecord]:
        point = Point(centroid)
        best: Optional[RoomRecord] = None
        best_distance = float("inf")
        for room in rooms:
            polygon = room["polygon"]
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                continue
            margin = max(80.0, 0.05 * float(polygon.length))
            edge_distance = float(point.distance(polygon))
            if edge_distance > margin:
                continue
            centroid_distance = float(point.distance(polygon.centroid))
            if centroid_distance < best_distance:
                best = room
                best_distance = centroid_distance
        return best

    @staticmethod
    def _room_below_callout(rooms: Sequence[RoomRecord], centroid: tuple[float, float]) -> Optional[RoomRecord]:
        """Pick the room a dimension callout is sitting above.

        Printed sizes are placed over the top edge of a room, so the nearest
        centroid can be a small neighboring space such as a balcony.
        """
        center_x, center_y = centroid
        best: Optional[RoomRecord] = None
        best_gap = float("inf")
        for room in rooms:
            polygon = room["polygon"]
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                continue
            min_x, min_y, max_x, max_y = polygon.bounds
            if center_x < min_x - 40 or center_x > max_x + 40:
                continue
            height = max_y - min_y
            if height <= 1 or center_y > min_y + 0.35 * height:
                continue
            gap = abs(min_y - center_y)
            if gap < best_gap:
                best = room
                best_gap = gap
        return best

    @staticmethod
    def _room_scale(room: RoomRecord) -> Optional[float]:
        polygon = room["polygon"]
        dimensions = room["dimensions"]
        if not isinstance(polygon, Polygon) or polygon.is_empty or not dimensions:
            return None
        min_x, min_y, max_x, max_y = polygon.bounds
        pixel_width = float(max_x - min_x)
        pixel_height = float(max_y - min_y)
        if pixel_width <= 1 or pixel_height <= 1:
            return None
        ratios: list[float] = []
        for dimension in dimensions:
            width_ft = float(dimension.get("width_ft") or 0.0)
            height_ft = float(dimension.get("height_ft") or 0.0)
            if width_ft <= 0 or height_ft <= 0:
                continue
            direct_error = abs((pixel_width / pixel_height) - (width_ft / height_ft))
            swapped_error = abs((pixel_width / pixel_height) - (height_ft / width_ft))
            if direct_error <= swapped_error:
                ratios.append(((pixel_width / width_ft) + (pixel_height / height_ft)) / 2.0)
            else:
                ratios.append(((pixel_width / height_ft) + (pixel_height / width_ft)) / 2.0)
        if not ratios:
            return None
        return float(np.median(ratios))

    @staticmethod
    def _bbox_size(bbox: object) -> tuple[float, float]:
        if not isinstance(bbox, list) or not bbox:
            return 0.0, 0.0
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        return max(xs) - min(xs), max(ys) - min(ys)

    @staticmethod
    def _bbox_centroid(bbox: object) -> Optional[tuple[float, float]]:
        if not isinstance(bbox, list) or not bbox:
            return None
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        return sum(xs) / len(xs), sum(ys) / len(ys)
