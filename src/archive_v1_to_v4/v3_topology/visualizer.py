"""Phase 5: draw the orthogonal wall graph as a blueprint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import yaml
from shapely.geometry import LineString, MultiPolygon, Polygon, box as shapely_box
from shapely.ops import polylabel


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _faces(polygon: Polygon) -> list[Polygon]:
    if isinstance(polygon, MultiPolygon):
        return [face for face in polygon.geoms if not face.is_empty]
    if isinstance(polygon, Polygon) and not polygon.is_empty:
        return [polygon]
    return []


def _lines(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        found = []
        for part in geometry.geoms:
            found.extend(_lines(part))
        return found
    return []


def _chord_bounds(polygon: Polygon, x: float, y: float, horizontal: bool) -> tuple[float, float] | None:
    """The slice of the polygon that passes through (x, y), as (low, high)."""
    minx, miny, maxx, maxy = polygon.bounds
    if horizontal:
        cut = LineString([(minx - 2, y), (maxx + 2, y)])
    else:
        cut = LineString([(x, miny - 2), (x, maxy + 2)])
    best: tuple[float, float] | None = None
    best_span = -1.0
    for line in _lines(polygon.intersection(cut)):
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        if horizontal:
            lo, hi = min(point[0] for point in coords), max(point[0] for point in coords)
            if lo - 0.5 <= x <= hi + 0.5 and hi - lo > best_span:
                best, best_span = (lo, hi), hi - lo
        else:
            lo, hi = min(point[1] for point in coords), max(point[1] for point in coords)
            if lo - 0.5 <= y <= hi + 0.5 and hi - lo > best_span:
                best, best_span = (lo, hi), hi - lo
    return best


def _chord(polygon: Polygon, x: float, y: float, horizontal: bool) -> float:
    """Length of the polygon slice that passes through (x, y)."""
    bounds = _chord_bounds(polygon, x, y, horizontal)
    if bounds is None:
        return 0.0
    return bounds[1] - bounds[0]


def _place(polygon: Polygon, cx: float, cy: float, foot_w: float, foot_h: float) -> tuple[float, float] | None:
    """Slide the label along the local cross until the rectangle is inside the room."""
    pad = 3.0
    x, y = cx, cy
    for _ in range(4):
        horizontal = _chord_bounds(polygon, x, y, horizontal=True)
        vertical = _chord_bounds(polygon, x, y, horizontal=False)
        if horizontal is None or vertical is None:
            return None
        hlo, hhi = horizontal
        vlo, vhi = vertical
        if hhi - hlo < foot_w + 2 * pad or vhi - vlo < foot_h + 2 * pad:
            return None
        x = min(max(x, hlo + foot_w / 2 + pad), hhi - foot_w / 2 - pad)
        y = min(max(y, vlo + foot_h / 2 + pad), vhi - foot_h / 2 - pad)
    if _text_inside(polygon, x, y, foot_w, foot_h):
        return x, y
    return None


def _text_inside(polygon: Polygon, cx: float, cy: float, block_w: float, block_h: float) -> bool:
    """The label rectangle has to sit inside the room, including beside a notch."""
    pad = 3.0
    rect = shapely_box(cx - block_w / 2 - pad, cy - block_h / 2 - pad, cx + block_w / 2 + pad, cy + block_h / 2 + pad)
    return polygon.covers(rect)


def _anchor(polygon: Polygon):
    """Interior point farthest from the walls.

    The geometric centroid of an L-shaped room, or of a room with a furniture
    hole, can fall outside the face. The label needs a point that stays inside.
    """
    try:
        point = polylabel(polygon, tolerance=1.0)
    except Exception:
        point = polygon.representative_point()
    if polygon.contains(point):
        return point
    return polygon.representative_point()


class TopologyBlueprint:
    """Stroke the shared walls and label each room at its centroid.

    The canvas is blank white. Wall thickness is the stroke width. Each face
    is labeled with its room id, its share of the plan, and its square footage
    from the global pixels-per-foot scale.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config["v3_blueprint"]

    def render(
        self,
        shape: tuple[int, int],
        polygons: Sequence[Polygon],
        scale_px_per_ft: Optional[float] = None,
    ) -> np.ndarray:
        """Return a BGR blueprint. ``shape`` is ``(height, width)`` of the original render."""
        height, width = int(shape[0]), int(shape[1])
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        faces = [face for polygon in polygons for face in _faces(polygon)]
        thickness = int(self.params["wall_thickness_px"])
        for face in faces:
            self._stroke(canvas, face, thickness)
        for room in self.rooms(faces, scale_px_per_ft):
            self._draw_label(canvas, room, thickness)
        return canvas

    def rooms(self, polygons: Sequence[Polygon], scale_px_per_ft: Optional[float]) -> list[dict]:
        """One record per face: id, relative area, and square feet when a scale exists."""
        areas = [float(polygon.area) for polygon in polygons]
        total = sum(areas)
        scale = float(scale_px_per_ft) if scale_px_per_ft else 0.0
        records = []
        for index, (polygon, area) in enumerate(zip(polygons, areas), start=1):
            point = _anchor(polygon)
            sqft = area / (scale ** 2) if scale > 0 else None
            records.append(
                {
                    "room_id": index,
                    "pixel_area": area,
                    "relative_area_pct": 0.0 if total <= 0 else 100.0 * area / total,
                    "area_sqft": sqft,
                    "centroid": (float(point.x), float(point.y)),
                    "polygon": polygon,
                }
            )
        return records

    def _stroke(self, canvas: np.ndarray, polygon: Polygon, thickness: int) -> None:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            points = np.array(ring.coords, dtype=np.int32)
            if len(points) < 2:
                continue
            cv2.polylines(canvas, [points], True, (0, 0, 0), thickness, cv2.LINE_AA)

    def _draw_label(self, canvas: np.ndarray, room: dict, thickness: int) -> None:
        polygon: Polygon = room["polygon"]
        cx, cy = room["centroid"]
        margin = thickness + 10
        usable_w = _chord(polygon, cx, cy, horizontal=True) - margin
        usable_h = _chord(polygon, cx, cy, horizontal=False) - margin
        if usable_w < 10 or usable_h < 10:
            return
        stacked = [f"Room {room['room_id']}", f"{room['relative_area_pct']:.1f}%"]
        if room["area_sqft"] is not None:
            stacked.append(f"{room['area_sqft']:.1f} sq ft")
        compact = "  ".join(stacked)
        font = cv2.FONT_HERSHEY_SIMPLEX
        chosen = self._fit_lines(stacked, font, usable_w, usable_h, polygon, cx, cy, rotated=False)
        if chosen is None:
            chosen = self._fit_lines([compact], font, usable_w, usable_h, polygon, cx, cy, rotated=False)
        if chosen is not None:
            self._blit_upright(canvas, *chosen)
            return
        # A tall narrow room holds the same lines turned on their side.
        chosen = self._fit_lines(stacked, font, usable_h, usable_w, polygon, cx, cy, rotated=True)
        if chosen is None:
            chosen = self._fit_lines([compact], font, usable_h, usable_w, polygon, cx, cy, rotated=True)
        if chosen is None:
            sideways = usable_h > usable_w
            chosen = self._fit_lines(
                [str(room["room_id"])],
                font,
                max(usable_w, usable_h),
                min(usable_w, usable_h),
                polygon,
                cx,
                cy,
                rotated=sideways,
            )
            if chosen is None:
                return
            if sideways:
                self._blit_rotated(canvas, *chosen)
            else:
                self._blit_upright(canvas, *chosen)
            return
        self._blit_rotated(canvas, *chosen)

    def _blit_upright(self, canvas: np.ndarray, lines, scale, sizes, gap, cx: float, cy: float) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        block_h = gap * (len(lines) - 1) + sizes[0][1]
        height, width = canvas.shape[:2]
        origin_y = int(round(cy - block_h / 2 + sizes[0][1]))
        for line, (text_w, text_h) in zip(lines, sizes):
            origin_x = int(round(cx - text_w / 2))
            if origin_x < 0 or origin_y - text_h < 0 or origin_x + text_w >= width or origin_y >= height:
                return
            cv2.putText(canvas, line, (origin_x, origin_y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
            origin_y += gap

    def _blit_rotated(self, canvas: np.ndarray, lines, scale, sizes, gap, cx: float, cy: float) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        block_w = max(size[0] for size in sizes) + 2
        block_h = gap * (len(lines) - 1) + sizes[0][1] + 2
        patch = np.full((block_h, block_w, 3), 255, dtype=np.uint8)
        origin_y = sizes[0][1]
        for line, (text_w, _text_h) in zip(lines, sizes):
            origin_x = max(0, (block_w - text_w) // 2)
            cv2.putText(patch, line, (origin_x, origin_y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
            origin_y += gap
        rotated = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self._paste(canvas, rotated, cx, cy)

    @staticmethod
    def _paste(canvas: np.ndarray, patch: np.ndarray, cx: float, cy: float) -> None:
        ph, pw = patch.shape[:2]
        height, width = canvas.shape[:2]
        x0 = int(round(cx - pw / 2))
        y0 = int(round(cy - ph / 2))
        src_x0, src_y0 = max(0, -x0), max(0, -y0)
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(width, x0 + pw), min(height, y0 + ph)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return
        crop = patch[src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)]
        ink = np.any(crop < 250, axis=2)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1][ink] = crop[ink]

    @staticmethod
    def _fit_lines(lines, font, usable_w, usable_h, polygon: Polygon, cx: float, cy: float, rotated: bool):
        scale = 0.55
        while scale >= 0.28:
            sizes = [cv2.getTextSize(line, font, scale, 1)[0] for line in lines]
            gap = int(round(14 * scale + 4))
            block_w = max(size[0] for size in sizes)
            block_h = gap * (len(lines) - 1) + sizes[0][1]
            foot_w, foot_h = (block_h, block_w) if rotated else (block_w, block_h)
            placed = None
            if block_w <= usable_w and block_h <= usable_h:
                placed = _place(polygon, cx, cy, foot_w, foot_h)
            if placed is not None:
                return lines, scale, sizes, gap, placed[0], placed[1]
            scale = round(scale - 0.04, 2)
        return None
