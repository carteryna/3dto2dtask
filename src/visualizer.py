"""Final annotated overlay for the floor-plan pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry import Polygon


class PipelineVisualizer:
    """Draw regularized rooms, labels, and area readouts on the source render."""

    def save_annotated(
        self,
        image: np.ndarray,
        rooms: Sequence[dict],
        image_name: str,
        output_dir: str | Path,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_name).stem
        path = output_dir / f"annotated_{stem}.png"
        cv2.imwrite(str(path), self.render(image, rooms))
        return path

    def render(self, image: np.ndarray, rooms: Sequence[dict]) -> np.ndarray:
        """Return a BGR image with filled polygons, outlines, and centroid labels."""
        panel = image.copy()
        visible = [room for room in rooms if self._polygon(room) is not None]
        fill = np.zeros_like(panel)
        rings: list[tuple[dict, np.ndarray, tuple[int, int, int]]] = []
        for index, room in enumerate(visible):
            polygon = self._polygon(room)
            ring = self._ring(polygon)
            if ring is None:
                continue
            color = self._color(index, len(visible))
            cv2.fillPoly(fill, [ring], color)
            rings.append((room, ring, color))

        blended = cv2.addWeighted(panel, 0.55, fill, 0.45, 0)
        occupied = np.any(fill > 0, axis=2)
        panel[occupied] = blended[occupied]

        for room, ring, color in rings:
            cv2.polylines(panel, [ring], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(panel, [ring], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
            self._draw_label(panel, room)
        return panel

    @staticmethod
    def _polygon(room: dict) -> Polygon | None:
        polygon = room.get("polygon")
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return None
        return polygon

    @staticmethod
    def _ring(polygon: Polygon) -> np.ndarray | None:
        coords = np.asarray(polygon.exterior.coords, dtype=np.float32)
        if len(coords) < 3:
            return None
        return np.round(coords).astype(np.int32).reshape(-1, 1, 2)

    @staticmethod
    def _color(index: int, count: int) -> tuple[int, int, int]:
        hue = int(round(179 * index / max(count, 1)))
        color = cv2.cvtColor(np.uint8([[[hue, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        return int(color[0]), int(color[1]), int(color[2])

    def _draw_label(self, image: np.ndarray, room: dict) -> None:
        polygon = self._polygon(room)
        if polygon is None:
            return
        centroid = polygon.centroid
        lines = [f"#{room.get('room_id', '?')} {room.get('label', 'Room')}"]
        relative = room.get("relative_area_pct")
        if relative is not None:
            lines.append(f"{float(relative):.1f}%")
        sqft = room.get("estimated_sqft")
        if sqft is not None:
            lines.append(f"{float(sqft):.1f} sqft")
        x = int(round(centroid.x))
        y = int(round(centroid.y)) - 8 * (len(lines) - 1)
        for line in lines:
            origin = (x, y)
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
            y += 16
