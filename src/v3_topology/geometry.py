"""Phase 3: orthogonalize the shared-wall line graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import yaml
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

# A segment is [[x0, y0], [x1, y1]].
Segment = np.ndarray
EdgeRecord = tuple[float, float, int, int]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class GeometryDebug:
    raw_segments: int = 0
    orthogonal_segments: int = 0
    axis_aligned: bool = True
    notes: List[str] = field(default_factory=list)


class ManhattanLineGraph:
    """Turn watershed boundaries into corner-matched orthogonal room polygons.

    Each cell keeps a closed outline. Every edge of that outline is forced
    onto the nearer image axis, and parallel edges within a few pixels share
    one coordinate, so the two sides of a wall become the same line and
    neighbouring rooms use the same corner. A room that sits inside another
    (a bed inside a bedroom) is cut out of the parent, which leaves the faces
    disjoint.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config["v3_geometry"]
        self.debug: Optional[GeometryDebug] = None
        self._contours: list[np.ndarray] = []
        self._rings: list[np.ndarray] = []

    def extract_segments(self, cells: Sequence[np.ndarray]) -> List[Segment]:
        """One straight piece of each cell outline, including the edge a neighbour shares."""
        epsilon = float(self.params["approx_epsilon_px"])
        minimum = float(self.params["min_segment_px"])
        segments: list[Segment] = []
        contours: list[np.ndarray] = []
        for cell in cells:
            binary = np.where(cell > 0, 255, 0).astype(np.uint8)
            found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not found:
                continue
            contour = max(found, key=cv2.contourArea)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            points = approx.reshape(-1, 2).astype(np.float64)
            if len(points) >= 2 and np.allclose(points[0], points[-1]):
                points = points[:-1]
            if len(points) < 3:
                continue
            contours.append(points)
            for index in range(len(points)):
                start = points[index]
                end = points[(index + 1) % len(points)]
                if float(np.hypot(end[0] - start[0], end[1] - start[1])) < minimum:
                    continue
                segments.append(np.stack([start, end]))
        self._contours = contours
        self._rings = []
        self.debug = GeometryDebug(raw_segments=len(segments))
        return segments

    def orthogonalize(self, segments: Sequence[Segment]) -> List[Segment]:
        """Force every kept edge onto 0° or 90° and align shared walls."""
        rings = [self._orthogonal_ring(contour) for contour in self._contours]
        rings = [ring for ring in rings if ring is not None]
        if not rings:
            rings = self._rings_from_segments(segments)
        rings = self._snap_rings(rings)
        self._rings = rings
        orthogonal = self._edges(rings)
        raw = self.debug.raw_segments if self.debug is not None else 0
        self.debug = GeometryDebug(
            raw_segments=raw,
            orthogonal_segments=len(orthogonal),
            axis_aligned=all(self._is_axis_aligned(segment) for segment in orthogonal),
        )
        return orthogonal

    def polygons(self, segments: Sequence[Segment]) -> List[Polygon]:
        """Closed orthogonal faces. A contained room is cut out of its parent."""
        del segments
        minimum = float(self.params["min_polygon_area_px"])
        faces: list[Polygon] = []
        for ring in self._rings:
            polygon = Polygon(np.asarray(ring, dtype=np.float64))
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            for piece in self._parts(polygon):
                if piece.area >= minimum and piece.is_valid:
                    faces.append(piece)
        faces.sort(key=lambda face: face.area)
        kept: list[Polygon] = []
        for face in faces:
            current: list[Polygon] = [face]
            for earlier in kept:
                next_parts: list[Polygon] = []
                for part in current:
                    point = earlier.representative_point()
                    if point is not None and point.within(part) and earlier.area < part.area * 0.98:
                        next_parts.extend(self._parts(part.difference(earlier)))
                    else:
                        next_parts.append(part)
                current = next_parts
            for part in current:
                cleaned = part.buffer(0) if not part.is_valid else part
                for piece in self._parts(cleaned):
                    if piece.area >= minimum and piece.is_valid:
                        kept.append(piece)
        kept = self._resolve_overlaps(kept, minimum)
        kept.sort(key=lambda face: (face.centroid.y, face.centroid.x))
        return kept

    @staticmethod
    def _resolve_overlaps(faces: List[Polygon], minimum: float) -> List[Polygon]:
        """Give a shared sliver to the smaller room so two faces do not cover the same pixels."""
        rooms = list(faces)
        for index in range(len(rooms)):
            for other in range(index + 1, len(rooms)):
                if rooms[index].is_empty or rooms[other].is_empty or not rooms[index].intersects(rooms[other]):
                    continue
                overlap = rooms[index].intersection(rooms[other])
                if overlap.is_empty or overlap.area < 1.0:
                    continue
                if rooms[index].area >= rooms[other].area:
                    rooms[index] = rooms[index].difference(rooms[other])
                else:
                    rooms[other] = rooms[other].difference(rooms[index])
        resolved: list[Polygon] = []
        for room in rooms:
            cleaned = room.buffer(0) if not room.is_valid else room
            for piece in ManhattanLineGraph._parts(cleaned):
                if piece.area >= minimum and piece.is_valid:
                    resolved.append(piece)
        return resolved

    def _orthogonal_ring(self, points: np.ndarray) -> Optional[np.ndarray]:
        """A closed outline whose edges alternate horizontal and vertical."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            return self._box_ring(pts)
        changed = True
        while changed and len(pts) >= 4:
            changed = False
            count = len(pts)
            directions = [self._horizontal(pts[index], pts[(index + 1) % count]) for index in range(count)]
            keep = [index for index in range(count) if directions[index] != directions[(index - 1) % count]]
            if len(keep) < 4:
                break
            if len(keep) != count:
                pts = pts[keep]
                changed = True
        if len(pts) < 4:
            return self._box_ring(np.asarray(points, dtype=np.float64))
        count = len(pts)
        xs = np.full(count, np.nan)
        ys = np.full(count, np.nan)
        for index in range(count):
            start = pts[index]
            end = pts[(index + 1) % count]
            if self._horizontal(start, end):
                y = (start[1] + end[1]) / 2.0
                ys[index] = y
                ys[(index + 1) % count] = y
            else:
                x = (start[0] + end[0]) / 2.0
                xs[index] = x
                xs[(index + 1) % count] = x
        for index in range(count):
            if np.isnan(xs[index]):
                xs[index] = pts[index, 0]
            if np.isnan(ys[index]):
                ys[index] = pts[index, 1]
        return np.stack([xs, ys], axis=1)

    def _snap_rings(self, rings: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Move parallel edges that nearly coincide onto one shared coordinate."""
        tolerance = float(self.params["coord_snap_px"])
        horizontal: list[EdgeRecord] = []
        vertical: list[EdgeRecord] = []
        for ring_index, ring in enumerate(rings):
            count = len(ring)
            for edge_index in range(count):
                start = ring[edge_index]
                end = ring[(edge_index + 1) % count]
                length = float(abs(end[0] - start[0]) + abs(end[1] - start[1]))
                if self._horizontal(start, end):
                    horizontal.append(((start[1] + end[1]) / 2.0, length, ring_index, edge_index))
                else:
                    vertical.append(((start[0] + end[0]) / 2.0, length, ring_index, edge_index))
        y_of = self._grouped_coordinate(horizontal, tolerance)
        x_of = self._grouped_coordinate(vertical, tolerance)
        snapped: list[np.ndarray] = []
        for ring_index, ring in enumerate(rings):
            updated = np.array(ring, dtype=np.float64, copy=True)
            count = len(updated)
            for edge_index in range(count):
                key = (ring_index, edge_index)
                if key in y_of:
                    y = y_of[key]
                    updated[edge_index, 1] = y
                    updated[(edge_index + 1) % count, 1] = y
                if key in x_of:
                    x = x_of[key]
                    updated[edge_index, 0] = x
                    updated[(edge_index + 1) % count, 0] = x
            cleaned = self._drop_duplicates(updated)
            if cleaned is not None:
                snapped.append(cleaned)
        return snapped

    @staticmethod
    def _grouped_coordinate(records: Sequence[EdgeRecord], tolerance: float) -> dict[tuple[int, int], int]:
        if not records:
            return {}
        ordered = sorted(records, key=lambda record: record[0])
        groups: list[list[EdgeRecord]] = [[ordered[0]]]
        for record in ordered[1:]:
            if record[0] - groups[-1][0][0] <= tolerance:
                groups[-1].append(record)
            else:
                groups.append([record])
        assigned: dict[tuple[int, int], int] = {}
        for group in groups:
            weight = sum(record[1] for record in group) or 1.0
            coordinate = int(round(sum(record[0] * record[1] for record in group) / weight))
            for record in group:
                assigned[(record[2], record[3])] = coordinate
        return assigned

    @staticmethod
    def _drop_duplicates(ring: np.ndarray) -> Optional[np.ndarray]:
        if len(ring) == 0:
            return None
        kept = [ring[0]]
        for point in ring[1:]:
            if abs(point[0] - kept[-1][0]) > 0.5 or abs(point[1] - kept[-1][1]) > 0.5:
                kept.append(point)
        if len(kept) >= 2 and abs(kept[0][0] - kept[-1][0]) <= 0.5 and abs(kept[0][1] - kept[-1][1]) <= 0.5:
            kept.pop()
        if len(kept) < 4:
            return None
        return np.asarray(kept, dtype=np.float64)

    @staticmethod
    def _edges(rings: Sequence[np.ndarray]) -> List[Segment]:
        segments: list[Segment] = []
        seen: set[tuple[int, int, int, int]] = set()
        for ring in rings:
            count = len(ring)
            for index in range(count):
                start = ring[index]
                end = ring[(index + 1) % count]
                x0, y0 = int(round(float(start[0]))), int(round(float(start[1])))
                x1, y1 = int(round(float(end[0]))), int(round(float(end[1])))
                if x0 == x1 and y0 == y1:
                    continue
                key = (x0, y0, x1, y1) if (x0, y0) <= (x1, y1) else (x1, y1, x0, y0)
                if key in seen:
                    continue
                seen.add(key)
                segments.append(np.array([[x0, y0], [x1, y1]], dtype=np.float64))
        return segments

    def _rings_from_segments(self, segments: Sequence[Segment]) -> list[np.ndarray]:
        """Fallback when orthogonalize is called without extract_segments."""
        if not segments:
            return []
        points = np.vstack([np.asarray(segment, dtype=np.float64).reshape(2, 2) for segment in segments])
        box = self._box_ring(points)
        return [box] if box is not None else []

    @staticmethod
    def _box_ring(points: np.ndarray) -> Optional[np.ndarray]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return None
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)

    @staticmethod
    def _horizontal(start: np.ndarray, end: np.ndarray) -> bool:
        return abs(float(end[0] - start[0])) >= abs(float(end[1] - start[1]))

    @staticmethod
    def _is_axis_aligned(segment: Segment) -> bool:
        (x0, y0), (x1, y1) = np.asarray(segment, dtype=np.float64).reshape(2, 2)
        return abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6

    @staticmethod
    def _parts(geometry: object) -> List[Polygon]:
        if geometry is None or getattr(geometry, "is_empty", True):
            return []
        if isinstance(geometry, Polygon):
            return [geometry]
        if isinstance(geometry, MultiPolygon):
            return [part for part in geometry.geoms if isinstance(part, Polygon)]
        if isinstance(geometry, GeometryCollection):
            parts: list[Polygon] = []
            for child in geometry.geoms:
                parts.extend(ManhattanLineGraph._parts(child))
            return parts
        return []
