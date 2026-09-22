"""Vector regularization of room masks into orthogonal polygons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import yaml
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _largest_polygon(geometry) -> Polygon:
    """Return the largest polygonal piece of a shapely geometry."""
    if geometry is None or geometry.is_empty:
        return Polygon()
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda piece: piece.area)
    if isinstance(geometry, GeometryCollection):
        polygons = [piece for piece in geometry.geoms if isinstance(piece, Polygon) and not piece.is_empty]
        if not polygons:
            return Polygon()
        return max(polygons, key=lambda piece: piece.area)
    return Polygon()


def _angular_distance(angle_deg: float, axis_deg: float) -> float:
    delta = (angle_deg - axis_deg + 180.0) % 360.0 - 180.0
    return abs(delta)


class GeometryRegularizer:
    """Turn noisy room masks into simplified, near-orthogonal polygons.

    Contours are converted to valid shapely polygons, Douglas-Peucker
    simplified, snapped toward Manhattan axes, then clipped so neighboring
    rooms do not share interior pixels.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.geometry = self.config["geometry"]

    def mask_to_polygon(self, mask: np.ndarray) -> Polygon:
        """Convert the outer contour of a binary room mask into a valid polygon."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Polygon()
        contour = max(contours, key=cv2.contourArea)
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            return Polygon()
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.is_empty:
            polygon = polygon.buffer(0)
        return _largest_polygon(polygon)

    def simplify_polygon(self, polygon: Polygon, epsilon_ratio: float) -> Polygon:
        """Douglas-Peucker simplify with epsilon = ratio times the perimeter.

        ``epsilon_ratio`` is ``geometry.douglas_peucker_epsilon`` from the
        pipeline config. The tolerance removes furniture, rug, and door-edge
        jaggies while ``preserve_topology`` keeps the room ring closed.
        """
        if polygon.is_empty or polygon.length <= 0:
            return Polygon()
        epsilon = float(epsilon_ratio) * float(polygon.length)
        simplified = polygon.simplify(epsilon, preserve_topology=True)
        if not simplified.is_valid:
            simplified = simplified.buffer(0)
        return _largest_polygon(simplified)

    def snap_to_manhattan_orthogonality(self, polygon: Polygon, tolerance_deg: float) -> Polygon:
        """Force edges within ``tolerance_deg`` of the axes to be horizontal or vertical.

        Each corner is rebuilt as the intersection of its two adjacent edge
        lines so the ring stays closed. Edges farther than the tolerance,
        including 45-degree door leaves, are left diagonal.
        """
        if polygon.is_empty:
            return Polygon()
        coords = list(polygon.exterior.coords)[:-1]
        if len(coords) < 3:
            return polygon

        lines = [
            self._edge_line(coords[index], coords[(index + 1) % len(coords)], tolerance_deg)
            for index in range(len(coords))
        ]
        rebuilt: list[tuple[float, float]] = []
        for index, original in enumerate(coords):
            previous = lines[(index - 1) % len(coords)]
            current = lines[index]
            intersection = self._intersect_lines(previous, current)
            if intersection is None:
                rebuilt.append((float(original[0]), float(original[1])))
                continue
            previous_point = coords[(index - 1) % len(coords)]
            next_point = coords[(index + 1) % len(coords)]
            reach = max(
                math.hypot(original[0] - previous_point[0], original[1] - previous_point[1]),
                math.hypot(next_point[0] - original[0], next_point[1] - original[1]),
            )
            jump = math.hypot(intersection[0] - original[0], intersection[1] - original[1])
            if jump > reach * 1.25 + 2.0:
                rebuilt.append((float(original[0]), float(original[1])))
            else:
                rebuilt.append(intersection)

        rebuilt = self._dedupe_ring(rebuilt)
        if len(rebuilt) < 3:
            return polygon
        snapped = Polygon(rebuilt)
        if not snapped.is_valid:
            snapped = snapped.buffer(0)
        snapped = _largest_polygon(snapped)
        if snapped.is_empty or snapped.area < 0.4 * polygon.area or snapped.area > 2.5 * polygon.area:
            return polygon
        return snapped

    def resolve_spatial_overlaps(self, polygons: List[Polygon]) -> List[Polygon]:
        """Remove shared interiors so adjacent rooms meet on a single perimeter.

        Larger rooms keep an overlapping sliver. A one-pixel close fills hairline
        gaps left by independent contour snaps, then ``difference`` against the
        union of already accepted rooms clears the overlap.
        """
        prepared: list[tuple[int, Polygon]] = []
        for index, polygon in enumerate(polygons):
            if polygon.is_empty:
                prepared.append((index, Polygon()))
                continue
            closed = _largest_polygon(polygon.buffer(1.0).buffer(-1.0))
            prepared.append((index, closed if not closed.is_empty else _largest_polygon(polygon.buffer(0))))

        order = sorted(range(len(prepared)), key=lambda index: prepared[index][1].area, reverse=True)
        accepted: list[Polygon] = []
        resolved = [Polygon() for _ in prepared]
        for slot in order:
            index, polygon = prepared[slot]
            if polygon.is_empty:
                continue
            if accepted:
                polygon = _largest_polygon(polygon.difference(unary_union(accepted)))
            if polygon.is_empty or polygon.area < 1.0:
                continue
            accepted.append(polygon)
            resolved[index] = polygon
        return resolved

    def _edge_line(
        self,
        start: Sequence[float],
        end: Sequence[float],
        tolerance_deg: float,
    ) -> tuple:
        """Classify an edge as horizontal, vertical, or an unconstrained diagonal."""
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return ("D", (float(start[0]), float(start[1]), float(end[0]), float(end[1])))
        angle = math.degrees(math.atan2(dy, dx))
        distance_horizontal = min(_angular_distance(angle, 0.0), _angular_distance(angle, 180.0))
        distance_vertical = min(_angular_distance(angle, 90.0), _angular_distance(angle, -90.0))
        if min(distance_horizontal, distance_vertical) > tolerance_deg:
            return ("D", (float(start[0]), float(start[1]), float(end[0]), float(end[1])))
        if distance_horizontal <= distance_vertical:
            return ("H", (float(start[1]) + float(end[1])) / 2.0)
        return ("V", (float(start[0]) + float(end[0])) / 2.0)

    @staticmethod
    def _line_coefficients(line: tuple) -> tuple[float, float, float, float]:
        """Return a point on the line and a direction vector."""
        kind, value = line
        if kind == "H":
            return 0.0, float(value), 1.0, 0.0
        if kind == "V":
            return float(value), 0.0, 0.0, 1.0
        x1, y1, x2, y2 = value
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    def _intersect_lines(self, first: tuple, second: tuple) -> tuple[float, float] | None:
        if first[0] == "H" and second[0] == "V":
            return (float(second[1]), float(first[1]))
        if first[0] == "V" and second[0] == "H":
            return (float(first[1]), float(second[1]))
        if first[0] == second[0] and first[0] in {"H", "V"}:
            return None
        x1, y1, dx1, dy1 = self._line_coefficients(first)
        x2, y2, dx2, dy2 = self._line_coefficients(second)
        determinant = dx1 * dy2 - dy1 * dx2
        if abs(determinant) < 1e-8:
            return None
        step = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / determinant
        return (x1 + step * dx1, y1 + step * dy1)

    @staticmethod
    def _dedupe_ring(points: list[tuple[float, float]], minimum_gap: float = 1.5) -> list[tuple[float, float]]:
        if not points:
            return []
        deduped = [points[0]]
        for point in points[1:]:
            if math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) >= minimum_gap:
                deduped.append(point)
        if len(deduped) > 1 and math.hypot(deduped[0][0] - deduped[-1][0], deduped[0][1] - deduped[-1][1]) < minimum_gap:
            deduped.pop()
        return deduped
