"""Phase 4: Spatial Containment & Semantic Room Classification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from shapely.geometry import Point, Polygon
from shapely.ops import polylabel, unary_union

from src.v4_cad_arrangement.objects import CADObject, ObjectDetectionResult

_SQFT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|square\s+feet|sqft|sq)\b",
    re.IGNORECASE,
)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class ArchitecturalRoom:
    category: str  # "BEDROOM", "BATHROOM", "LIVING / DINING", "KITCHEN", "BALCONY", "WALK-IN CLOSET", "STORAGE / ENTRY"
    polygon: Polygon
    pixel_area: float
    sqft_area: float
    dimension_ft: Tuple[float, float]  # (width_ft, length_ft)
    dimension_str: str  # e.g. "12'4\" x 10'8\""
    area_fraction: float  # fraction of total unit area
    contained_objects: List[CADObject] = field(default_factory=list)
    label_anchor: Tuple[float, float] = (0.0, 0.0)  # (x, y) optimal text placement


@dataclass
class ClassificationResult:
    rooms: List[ArchitecturalRoom] = field(default_factory=list)
    scale_px_per_ft: float = 0.0
    total_pixel_area: float = 0.0
    total_sqft: float = 0.0
    notes: List[str] = field(default_factory=list)


class RoomClassifier:
    """Performs spatial containment join between planar faces and decoupled CAD objects.

    Assigns architectural room categories (BEDROOM, BATHROOM, LIVING / DINING, KITCHEN,
    BALCONY, WALK-IN CLOSET), computes conformal metric scale (px/ft), and derives nominal
    room dimensions in feet and inches.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config.get("v4_classification", {})

    @staticmethod
    def parse_square_feet(filename: str) -> Optional[float]:
        """Extract declared square footage from filename."""
        match = _SQFT_PATTERN.search(filename)
        if match is not None:
            return float(match.group(1))
        return None

    def classify(
        self,
        faces: Sequence[Polygon],
        object_result: ObjectDetectionResult,
        filename: str,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> ClassificationResult:
        """Classify planar room faces using spatial containment join with CAD objects."""
        notes: List[str] = []
        if not faces:
            return ClassificationResult(notes=["No input room faces to classify."])

        # 1. Metric Scale Calculation
        total_pixel_area = float(sum(face.area for face in faces))
        sqft_declared = self.parse_square_feet(filename)

        if sqft_declared is not None and sqft_declared > 0:
            scale_px_per_ft = float(np.sqrt(total_pixel_area / sqft_declared))
            notes.append(f"Derived scale {scale_px_per_ft:.2f} px/ft from filename ({sqft_declared} sq ft).")
        else:
            scale_px_per_ft = float(self.params.get("fallback_scale_px_per_ft", 20.0))
            sqft_declared = total_pixel_area / (scale_px_per_ft ** 2)
            notes.append(f"Using fallback scale {scale_px_per_ft:.2f} px/ft.")

        # 2. Spatial Containment Join
        face_objects: List[List[CADObject]] = [[] for _ in faces]
        for obj in object_result.objects:
            best_idx = -1
            best_inter = 0.0
            obj_poly = obj.polygon
            for idx, face in enumerate(faces):
                inter = face.intersection(obj_poly).area
                if inter > best_inter:
                    best_inter = inter
                    best_idx = idx
            if best_idx >= 0 and best_inter > 0.15 * obj_poly.area:
                face_objects[best_idx].append(obj)

        # 3. Label Each Face
        raw_labels: List[str] = []
        for idx, face in enumerate(faces):
            objs = face_objects[idx]
            cats = {o.category for o in objs}
            area_sqft = face.area / (scale_px_per_ft ** 2)

            b = face.bounds
            span_w = (b[2] - b[0]) / scale_px_per_ft
            span_h = (b[3] - b[1]) / scale_px_per_ft
            aspect = max(span_w, span_h) / max(min(span_w, span_h), 0.1)

            # Rule 1: Bathroom if contains toilet or bathtub, or contains bathroom seed
            # Note: A real bathroom is compact (typically 30..90 sq ft). If a face swallowed
            # the entire living area (> 140 sq ft), it cannot be a bathroom even if a fixture leaked.
            has_toilet = "toilet" in cats
            has_tub = "bathtub" in cats
            has_bath_seed = any(face.contains(Point(sx, sy)) for sx, sy in object_result.bathroom_seeds)

            if (has_toilet or has_tub or has_bath_seed) and area_sqft <= 120.0:
                raw_labels.append("BATHROOM")
                continue

            # Rule 2: Bedroom if contains primary bed object and reasonable bedroom area (50..220 sq ft)
            has_bed = "bed" in cats
            has_bed_seed = any(face.contains(Point(sx, sy)) for sx, sy in object_result.bedroom_seeds)
            if (has_bed or has_bed_seed) and (55.0 <= area_sqft <= 240.0):
                raw_labels.append("BEDROOM")
                continue

            # Rule 3: Balcony if exterior-facing elongated polygon with low area or high aspect
            if area_sqft <= 75.0 and aspect >= 2.0 and not ({"bed", "sofa", "bathtub", "toilet"} & cats):
                raw_labels.append("BALCONY")
                continue

            # Rule 4: Walk-in Closet if small enclosed room (12..55 sqft)
            if 12.0 <= area_sqft <= 55.0 and not ({"sofa", "dining_table"} & cats):
                raw_labels.append("WALK-IN CLOSET")
                continue

            # Rule 5: Living / Dining Room if contains sofa, dining table, or represents large open area
            if ({"sofa", "dining_table"} & cats) or area_sqft >= 95.0:
                raw_labels.append("LIVING / DINING")
                continue

            # Default for remaining small unclassified spaces
            if area_sqft < 45.0:
                raw_labels.append("CLOSET / STORAGE")
            else:
                raw_labels.append("LIVING / DINING")

        # 4. Merge Adjacent Sub-Cells with the Same Semantic Category
        merged_rooms = self._merge_same_category_cells(faces, raw_labels, face_objects)

        # 5. Build ArchitecturalRoom Records
        rooms: List[ArchitecturalRoom] = []
        for poly, cat, objs in merged_rooms:
            px_area = float(poly.area)
            sqft = px_area / (scale_px_per_ft ** 2)
            frac = px_area / max(total_pixel_area, 1.0)

            # Dimensional span (width x length in feet and inches)
            b = poly.bounds
            w_ft = (b[2] - b[0]) / scale_px_per_ft
            l_ft = (b[3] - b[1]) / scale_px_per_ft
            dim_str = self._format_dimensions(w_ft, l_ft)

            # Anchor text label inside polygon using polylabel (visual center)
            pole = polylabel(poly, tolerance=1.0)
            label_pos = (float(pole.x), float(pole.y))

            rooms.append(
                ArchitecturalRoom(
                    category=cat,
                    polygon=poly,
                    pixel_area=px_area,
                    sqft_area=sqft,
                    dimension_ft=(w_ft, l_ft),
                    dimension_str=dim_str,
                    area_fraction=frac,
                    contained_objects=objs,
                    label_anchor=label_pos,
                )
            )

        notes.append(
            f"Classified {len(rooms)} architectural room(s): "
            + ", ".join(f"{r.category} ({r.sqft_area:.1f} sqft)" for r in rooms)
        )

        return ClassificationResult(
            rooms=rooms,
            scale_px_per_ft=scale_px_per_ft,
            total_pixel_area=total_pixel_area,
            total_sqft=sqft_declared,
            notes=notes,
        )

    @staticmethod
    def _format_dimensions(w_ft: float, l_ft: float) -> str:
        """Format dimensions as architectural feet and inches (e.g. 12'4\" x 10'8\")."""
        dim1 = max(w_ft, l_ft)
        dim2 = min(w_ft, l_ft)

        f1 = int(dim1)
        i1 = int(round((dim1 - f1) * 12))
        if i1 == 12:
            f1 += 1
            i1 = 0

        f2 = int(dim2)
        i2 = int(round((dim2 - f2) * 12))
        if i2 == 12:
            f2 += 1
            i2 = 0

        return f"{f1}'{i1}\" x {f2}'{i2}\""

    @staticmethod
    def _merge_same_category_cells(
        faces: Sequence[Polygon],
        labels: Sequence[str],
        face_objects: Sequence[List[CADObject]],
    ) -> List[Tuple[Polygon, str, List[CADObject]]]:
        """Merge adjacent cells sharing the same category into a unified room polygon."""
        n = len(faces)
        # Group by category, but keep non-adjacent components distinct
        visited = [False] * n
        merged: List[Tuple[Polygon, str, List[CADObject]]] = []

        for i in range(n):
            if visited[i]:
                continue
            cat = labels[i]
            # Flood fill adjacent cells of the same category
            comp_indices = [i]
            visited[i] = True
            queue = [i]
            while queue:
                curr = queue.pop(0)
                for j in range(n):
                    if not visited[j] and labels[j] == cat:
                        # Check shared edge
                        inter = faces[curr].intersection(faces[j])
                        if inter.length > 5.0:
                            visited[j] = True
                            comp_indices.append(j)
                            queue.append(j)

            comp_polys = [faces[idx] for idx in comp_indices]
            comp_objs: List[CADObject] = []
            for idx in comp_indices:
                comp_objs.extend(face_objects[idx])

            if len(comp_polys) == 1:
                merged.append((comp_polys[0], cat, comp_objs))
            else:
                union_poly = unary_union(comp_polys)
                if isinstance(union_poly, Polygon):
                    merged.append((union_poly, cat, comp_objs))
                else:
                    # MultiPolygon
                    for sub_poly in union_poly.geoms:
                        if isinstance(sub_poly, Polygon) and not sub_poly.is_empty:
                            merged.append((sub_poly, cat, comp_objs))

        return merged
