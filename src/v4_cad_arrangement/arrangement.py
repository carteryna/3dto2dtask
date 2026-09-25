"""Phase 3: Planar Wall-Graph Line Arrangement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from src.v4_cad_arrangement.objects import ObjectDetectionResult, ObjectDetector


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class CADWallSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    orientation: str  # "horizontal" | "vertical"
    wall_type: str  # "interior" | "exterior"
    length: float


@dataclass
class PlanarArrangementResult:
    faces: List[Polygon] = field(default_factory=list)
    wall_segments: List[CADWallSegment] = field(default_factory=list)
    exterior_envelope: Polygon = field(default_factory=lambda: Polygon())
    footprint_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    notes: List[str] = field(default_factory=list)


class PlanarWallArranger:
    """Computes a zero-gap, zero-overlap planar line arrangement of rectilinear walls.

    Extracts Manhattan wall centerlines, snaps collinear/parallel segments, extends dangling
    endpoints to meet perpendicular walls or the exterior envelope, and computes the dual
    polygonal face decomposition using Shapely planar graph polygonization.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config.get("v4_arrangement", {})
        self.config_path = path

    def arrange(
        self,
        rectified_image: np.ndarray,
        object_result: Optional[ObjectDetectionResult] = None,
        exterior_envelope: Optional[Polygon] = None,
    ) -> PlanarArrangementResult:
        """Extract wall centerlines and partition the floor plan into planar room faces."""
        h, w = rectified_image.shape[:2]
        notes: List[str] = []

        # 1. Exterior Footprint & Envelope
        footprint = self._building_footprint(rectified_image)
        if exterior_envelope is None or exterior_envelope.is_empty:
            envelope, approx_coords = self._extract_envelope(footprint)
        else:
            envelope = exterior_envelope
            approx_coords = np.array(envelope.exterior.coords)

        # 2. Object Result (Furniture layer decoupling)
        if object_result is None:
            detector = ObjectDetector(self.config_path)
            object_result = detector.detect(rectified_image)

        furn_mask = object_result.furniture_mask
        if furn_mask.shape != (h, w):
            furn_mask = cv2.resize(furn_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Restrict wall search domain to interior excluding decoupled furniture
        wall_domain = cv2.bitwise_and(footprint, cv2.bitwise_not(furn_mask))

        # 3. Detect Raw Manhattan Line Segments
        h_segs, v_segs = self._detect_raw_manhattan_segments(rectified_image, wall_domain)
        notes.append(f"Detected {len(h_segs)} horizontal and {len(v_segs)} vertical raw wall candidates.")

        # 4. Snap Parallel Segments & Merge Collinear Intervals
        snap_tol = float(self.params.get("snap_tolerance_px", 15.0))
        gap_bridge = float(self.params.get("gap_bridge_px", 30.0))
        min_run = float(self.params.get("min_wall_run_px", 45.0))

        h_snapped = self._cluster_and_snap(h_segs, is_horiz=True, snap_tol=snap_tol, gap_bridge=gap_bridge, min_run=min_run)
        v_snapped = self._cluster_and_snap(v_segs, is_horiz=False, snap_tol=snap_tol, gap_bridge=gap_bridge, min_run=min_run)
        notes.append(f"Snapped to {len(h_snapped)} horizontal and {len(v_snapped)} vertical wall lines.")

        # 5. Extend Dangling Endpoints (Ray-shooting to perpendicular walls and envelope)
        ext_tol = float(self.params.get("extension_tolerance_px", 75.0))
        extended_h, extended_v = self._extend_wall_endpoints(
            h_snapped, v_snapped, envelope, ext_tol=ext_tol
        )

        # 6. Build Exterior Envelope Segments
        envelope_lines: List[LineString] = []
        for i in range(len(approx_coords) - 1):
            p1 = (float(approx_coords[i][0]), float(approx_coords[i][1]))
            p2 = (float(approx_coords[i + 1][0]), float(approx_coords[i + 1][1]))
            envelope_lines.append(LineString([p1, p2]))

        # 7. Planar Line Arrangement (Shapely Unary Union & Polygonize)
        all_lines = envelope_lines + extended_h + extended_v
        noded = unary_union(all_lines)
        raw_faces = list(polygonize(noded))

        # 8. Filter Slivers & Keep Valid Room Faces
        min_area = float(self.params.get("min_room_area_px", 3500.0))
        min_dim = float(self.params.get("min_room_dimension_px", 22.0))

        valid_faces: List[Polygon] = []
        for face in raw_faces:
            b = face.bounds
            span_w = b[2] - b[0]
            span_h = b[3] - b[1]
            if min(span_w, span_h) > min_dim and face.area >= min_area:
                if envelope.buffer(5.0).contains(face.representative_point()):
                    valid_faces.append(face)

        notes.append(f"Arrangement generated {len(raw_faces)} total faces; retained {len(valid_faces)} room polygons.")

        # 9. Format CAD Wall Segments
        wall_segments: List[CADWallSegment] = []
        for line in extended_h:
            coords = list(line.coords)
            wall_segments.append(
                CADWallSegment(
                    start=coords[0],
                    end=coords[1],
                    orientation="horizontal",
                    wall_type="interior",
                    length=float(line.length),
                )
            )
        for line in extended_v:
            coords = list(line.coords)
            wall_segments.append(
                CADWallSegment(
                    start=coords[0],
                    end=coords[1],
                    orientation="vertical",
                    wall_type="interior",
                    length=float(line.length),
                )
            )
        for line in envelope_lines:
            coords = list(line.coords)
            dx = abs(coords[1][0] - coords[0][0])
            dy = abs(coords[1][1] - coords[0][1])
            orient = "horizontal" if dx >= dy else "vertical"
            wall_segments.append(
                CADWallSegment(
                    start=coords[0],
                    end=coords[1],
                    orientation=orient,
                    wall_type="exterior",
                    length=float(line.length),
                )
            )

        return PlanarArrangementResult(
            faces=valid_faces,
            wall_segments=wall_segments,
            exterior_envelope=envelope,
            footprint_mask=footprint,
            notes=notes,
        )

    def _detect_raw_manhattan_segments(
        self, image: np.ndarray, domain: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Detect raw line segments strictly aligned to horizontal and vertical axes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, _, _, _ = lsd.detect(gray)

        min_len = float(self.params.get("min_wall_length_px", 35.0))
        h_segs: List[Tuple[float, float, float]] = []  # (x_min, y, x_max)
        v_segs: List[Tuple[float, float, float]] = []  # (x, y_min, y_max)

        if lines is None:
            return h_segs, v_segs

        h, w = image.shape[:2]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            mx = int(round((x1 + x2) / 2.0))
            my = int(round((y1 + y2) / 2.0))

            if mx < 0 or mx >= w or my < 0 or my >= h or domain[my, mx] == 0:
                continue

            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len:
                continue

            angle = float(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
            if angle <= 4.5:
                h_segs.append((float(min(x1, x2)), float((y1 + y2) / 2.0), float(max(x1, x2))))
            elif abs(angle - 90.0) <= 4.5:
                v_segs.append((float((x1 + x2) / 2.0), float(min(y1, y2)), float(max(y1, y2))))

        return h_segs, v_segs

    def _cluster_and_snap(
        self,
        segs: Sequence[Tuple[float, float, float]],
        is_horiz: bool,
        snap_tol: float,
        gap_bridge: float,
        min_run: float,
    ) -> List[Tuple[float, float, float]]:
        """Snap parallel segments within snap_tol and merge collinear intervals."""
        if not segs:
            return []

        coord_idx = 1 if is_horiz else 0
        s_idx, e_idx = (0, 2) if is_horiz else (1, 2)

        sorted_segs = sorted(segs, key=lambda s: s[coord_idx])
        clusters: List[List[Tuple[float, float, float]]] = []

        for s in sorted_segs:
            c = s[coord_idx]
            if not clusters or abs(c - np.mean([item[coord_idx] for item in clusters[-1]])) > snap_tol:
                clusters.append([s])
            else:
                clusters[-1].append(s)

        snapped: List[Tuple[float, float, float]] = []
        for cl in clusters:
            total_len = sum(s[e_idx] - s[s_idx] for s in cl)
            mean_c = sum(s[coord_idx] * (s[e_idx] - s[s_idx]) for s in cl) / max(total_len, 1e-5)

            intervals = sorted([(s[s_idx], s[e_idx]) for s in cl], key=lambda x: x[0])
            merged: List[List[float]] = []
            for start, end in intervals:
                if not merged or start > merged[-1][1] + gap_bridge:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)

            for start, end in merged:
                if end - start >= min_run:
                    snapped.append((round(float(mean_c), 1), round(float(start), 1), round(float(end), 1)))

        return snapped

    def _extend_wall_endpoints(
        self,
        h_snapped: Sequence[Tuple[float, float, float]],
        v_snapped: Sequence[Tuple[float, float, float]],
        envelope: Polygon,
        ext_tol: float,
    ) -> Tuple[List[LineString], List[LineString]]:
        """Extend wall endpoints to intersect perpendicular walls or envelope boundary."""
        min_x, min_y, max_x, max_y = envelope.bounds

        ext_h: List[LineString] = []
        for y, x_min, x_max in h_snapped:
            # Skip if immediately adjacent to exterior envelope bounds
            if abs(y - min_y) < 20.0 or abs(y - max_y) < 20.0:
                continue

            target_left = x_min
            target_right = x_max

            # Extend left towards vertical walls
            for vx, vy1, vy2 in v_snapped:
                if vx < x_min and (x_min - vx) <= ext_tol and (vy1 - 40.0 <= y <= vy2 + 40.0):
                    target_left = min(target_left, vx)

            # Extend left towards envelope
            h_ray = LineString([(min_x - 10.0, y), (max_x + 10.0, y)])
            inter = h_ray.intersection(envelope.boundary)
            if not inter.is_empty:
                pts = [p.x for p in (inter.geoms if hasattr(inter, "geoms") else [inter]) if isinstance(p, Point)]
                left_env = [px for px in pts if px <= x_min]
                if left_env and (x_min - max(left_env)) <= ext_tol:
                    target_left = min(target_left, max(left_env))
                right_env = [px for px in pts if px >= x_max]
                if right_env and (min(right_env) - x_max) <= ext_tol:
                    target_right = max(target_right, min(right_env))

            # Extend right towards vertical walls
            for vx, vy1, vy2 in v_snapped:
                if vx > x_max and (vx - x_max) <= ext_tol and (vy1 - 40.0 <= y <= vy2 + 40.0):
                    target_right = max(target_right, vx)

            ext_h.append(LineString([(target_left, y), (target_right, y)]))

        ext_v: List[LineString] = []
        for x, y_min, y_max in v_snapped:
            # Skip if immediately adjacent to exterior envelope bounds
            if abs(x - min_x) < 20.0 or abs(x - max_x) < 20.0:
                continue

            target_top = y_min
            target_bot = y_max

            # Extend top towards horizontal walls
            for hy, hx1, hx2 in h_snapped:
                if hy < y_min and (y_min - hy) <= ext_tol and (hx1 - 40.0 <= x <= hx2 + 40.0):
                    target_top = min(target_top, hy)

            # Extend top towards envelope
            v_ray = LineString([(x, min_y - 10.0), (x, max_y + 10.0)])
            inter = v_ray.intersection(envelope.boundary)
            if not inter.is_empty:
                pts = [p.y for p in (inter.geoms if hasattr(inter, "geoms") else [inter]) if isinstance(p, Point)]
                top_env = [py for py in pts if py <= y_min]
                if top_env and (y_min - max(top_env)) <= ext_tol:
                    target_top = min(target_top, max(top_env))
                bot_env = [py for py in pts if py >= y_max]
                if bot_env and (min(bot_env) - y_max) <= ext_tol:
                    target_bot = max(target_bot, min(bot_env))

            # Extend bottom towards horizontal walls
            for hy, hx1, hx2 in h_snapped:
                if hy > y_max and (hy - y_max) <= ext_tol and (hx1 - 40.0 <= x <= hx2 + 40.0):
                    target_bot = max(target_bot, hy)

            ext_v.append(LineString([(x, target_top), (x, target_bot)]))

        return ext_h, ext_v

    def _extract_envelope(self, footprint: np.ndarray) -> Tuple[Polygon, np.ndarray]:
        """Extract simplified Manhattan exterior envelope polygon."""
        contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = footprint.shape[:2]
            box_poly = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
            return box_poly, np.array(box_poly.exterior.coords)

        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        eps_ratio = float(self.params.get("envelope_epsilon_ratio", 0.008))
        approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True).reshape(-1, 2)

        # Ensure polygon is closed
        if not np.allclose(approx[0], approx[-1]):
            approx = np.vstack([approx, approx[0]])

        envelope = Polygon(approx)
        if not envelope.is_valid:
            envelope = envelope.buffer(0)
        return envelope, approx

    @staticmethod
    def _building_footprint(image: np.ndarray) -> np.ndarray:
        """Extract filled apartment silhouette excluding surrounding canvas."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        chroma = np.maximum(np.abs(A.astype(np.int16) - 128), np.abs(B.astype(np.int16) - 128))
        canvas = ((L >= 245) & (chroma <= 10)).astype(np.uint8)
        _, labels = cv2.connectedComponents(canvas, connectivity=8)
        border_ids = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        border_ids = border_ids[border_ids != 0]
        silhouette = np.where(np.isin(labels, border_ids), 0, 255).astype(np.uint8)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        footprint = np.zeros_like(silhouette)
        if contours:
            cv2.drawContours(footprint, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
        else:
            footprint[:] = 255
        return footprint
