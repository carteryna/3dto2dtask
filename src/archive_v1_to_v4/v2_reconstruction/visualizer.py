"""Phase 5: render a synthetic 2D CAD-style blueprint."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class BlueprintRenderer:
    """Draw fitted rooms as black walls on a blank white sheet.

    Each room is an unfilled rectangle. The centroid carries the room label,
    its parsed or heuristic dimensions, and its share of the plan area.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config["blueprint"]

    def render(self, shape: tuple[int, int], rooms: Sequence[dict]) -> np.ndarray:
        """Return a BGR blueprint. ``shape`` is ``(height, width)`` of the flattened image."""
        height, width = int(shape[0]), int(shape[1])
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        thickness = int(self.params["wall_thickness_px"])
        for room in rooms:
            x, y, room_w, room_h = self._box(room)
            cv2.rectangle(canvas, (x, y), (x + room_w, y + room_h), (0, 0, 0), thickness, cv2.LINE_AA)
        for room in rooms:
            self._draw_label(canvas, room)
        return canvas

    def save(self, shape: tuple[int, int], rooms: Sequence[dict], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.render(shape, rooms))
        return path

    def _draw_label(self, canvas: np.ndarray, room: dict) -> None:
        x, y, room_w, room_h = self._box(room)
        if room_w < 36 or room_h < 28:
            return
        lines = [str(room.get("label") or "Room")]
        dimensions = str(room.get("dimensions") or "").strip()
        if dimensions:
            lines.append(dimensions)
        area = room.get("relative_area_pct")
        if area is not None:
            lines.append(f"{float(area):.1f}%")
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(0.55, min(room_w, room_h) / 320.0))
        gap = int(round(16 * scale + 6))
        sizes = [cv2.getTextSize(line, font, scale, 1)[0] for line in lines]
        block_h = gap * len(lines)
        origin_y = y + room_h // 2 - block_h // 2 + sizes[0][1]
        for line, (text_w, text_h) in zip(lines, sizes):
            origin_x = x + room_w // 2 - text_w // 2
            origin_x = int(np.clip(origin_x, x + 4, max(x + 4, x + room_w - text_w - 4)))
            origin_y = int(np.clip(origin_y, y + text_h + 4, max(y + text_h + 4, y + room_h - 4)))
            cv2.putText(canvas, line, (origin_x, origin_y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
            origin_y += gap

    @staticmethod
    def _box(room: dict) -> tuple[int, int, int, int]:
        return int(room["x"]), int(room["y"]), int(room["w"]), int(room["h"])
