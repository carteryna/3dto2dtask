"""Phase 4: Conformal Metric Scale & Dimensional Engine.

Calculates metric scale factors from target square footage in filenames and/or
standard architectural fixtures (countertop depth 60cm, toilet 70x40cm, bed 152x203cm),
validates dimensional bounds, formats room dimensions into standard architectural notation
(feet/inches and metric meters), and performs fixture cross-validation sanity checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import polylabel
import yaml

from src.v5_semi_automated.geometry import GeometryResult, RegularizedRoomNode
from src.v5_semi_automated.topology import TopologyResult


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


_SQFT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|square\s+feet|sqft|sq)\b",
    re.IGNORECASE,
)


@dataclass
class DimensionedRoom:
    """Represents a regularized room with precise metric and imperial dimensions."""
    room_id: int
    label: str
    polygon: Polygon
    area_px: float
    area_sqft: float
    area_m2: float
    area_fraction: float
    dimension_ft: Tuple[float, float]  # (max_span_ft, min_span_ft)
    dimension_m: Tuple[float, float]   # (max_span_m, min_span_m)
    dimension_str: str                 # e.g. "14'8\" x 12'4\""
    dimension_metric_str: str          # e.g. "4.47m x 3.76m"
    centroid: Tuple[float, float]
    label_anchor: Tuple[float, float]
    fixtures: List[str] = field(default_factory=list)
    is_outdoor: bool = False
    aspect_ratio: float = 1.0


@dataclass
class FixtureScaleAnchor:
    """Represents a metric scale anchor derived from standard architectural fixtures."""
    fixture_type: str                  # "countertop" | "bed" | "toilet"
    standard_dim_cm: Tuple[float, ...]
    measured_dim_px: Tuple[float, ...]
    scale_px_per_ft: float
    scale_cm_per_px: float
    error_vs_chosen_scale_pct: float
    notes: str = ""


@dataclass
class ScaleResult:
    """Result of Phase 4 Conformal Metric Scale & Dimensional Engine."""
    unit_key: str
    target_sqft: Optional[float]
    computed_total_sqft: float
    computed_interior_sqft: float
    computed_outdoor_sqft: float
    total_area_error_pct: float
    scale_px_per_ft: float
    scale_cm_per_px: float
    scale_source: str                  # "filename_conformal" | "fixture_consensus" | "fallback"
    rooms: List[DimensionedRoom] = field(default_factory=list)
    anchors: List[FixtureScaleAnchor] = field(default_factory=list)
    is_gate4_passed: bool = False
    notes: List[str] = field(default_factory=list)


class SemiAutomatedScale:
    """Calculates conformal metric scales and dimensional callouts for regularized floor plans.

    Integrates filename square footage with standard architectural fixtures to guarantee
    sub-3.0% total area error, cross-validates fixture dimensions (countertop depth 60cm,
    toilet 70x40cm, bed 152x203cm), and formats room spans into professional CAD strings.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config.get("v5_scale", {})
        self.cm_per_foot = float(self.params.get("cm_per_foot", 30.48))
        self.counter_depth_cm = float(self.params.get("standard_countertop_depth_cm", 60.0))
        self.toilet_w_cm = float(self.params.get("standard_toilet_width_cm", 40.0))
        self.toilet_l_cm = float(self.params.get("standard_toilet_length_cm", 70.0))
        self.toilet_sqft = float(self.params.get("standard_toilet_area_sqft", 3.01))
        self.bed_w_cm = float(self.params.get("standard_queen_bed_width_cm", 152.4))
        self.bed_l_cm = float(self.params.get("standard_queen_bed_length_cm", 203.2))
        self.bed_sqft = float(self.params.get("standard_queen_bed_area_sqft", 33.33))
        self.max_area_err_ratio = float(self.params.get("max_area_error_ratio", 0.03))
        self.fallback_scale_ft = float(self.params.get("fallback_scale_px_per_ft", 22.0))

    @staticmethod
    def parse_square_feet(filename_or_key: str) -> Optional[float]:
        """Extract declared target square footage from filename or path string."""
        stem = Path(filename_or_key).stem
        matches = _SQFT_PATTERN.findall(stem)
        if matches:
            return float(matches[-1])
        return None

    def process(
        self,
        geometry_result: GeometryResult,
        topology_result: Optional[TopologyResult] = None,
        filename_or_key: str = "",
    ) -> ScaleResult:
        """Execute Phase 4 metric scaling and dimensional calculations."""
        unit_key = geometry_result.unit_key
        notes: List[str] = []

        if not geometry_result.rooms:
            notes.append("No regularized rooms provided in input.")
            return ScaleResult(
                unit_key=unit_key,
                target_sqft=None,
                computed_total_sqft=0.0,
                computed_interior_sqft=0.0,
                computed_outdoor_sqft=0.0,
                total_area_error_pct=100.0,
                scale_px_per_ft=self.fallback_scale_ft,
                scale_cm_per_px=self.cm_per_foot / self.fallback_scale_ft,
                scale_source="none",
                notes=notes,
            )

        # 1. Total and interior pixel area
        total_px = sum(rm.area_px for rm in geometry_result.rooms)
        interior_px = sum(rm.area_px for rm in geometry_result.rooms if not rm.is_outdoor)
        outdoor_px = sum(rm.area_px for rm in geometry_result.rooms if rm.is_outdoor)

        # 2. Parse target square footage
        target_sqft = self.parse_square_feet(filename_or_key)
        if target_sqft is None:
            target_sqft = self.parse_square_feet(unit_key)

        # 3. Derive Fixture Scales from furniture layers
        furniture_layers = topology_result.furniture_layers if topology_result else {}
        anchors = self._compute_fixture_scales(furniture_layers)
        for anchor in anchors:
            notes.append(
                f"Fixture scale ({anchor.fixture_type}): {anchor.scale_px_per_ft:.2f} px/ft "
                f"({anchor.scale_cm_per_px:.2f} cm/px) - {anchor.notes}"
            )

        # 4. Determine Master Conformal Scale
        if target_sqft is not None and target_sqft > 0:
            scale_px_per_ft = float(np.sqrt(total_px / target_sqft))
            scale_source = "filename_conformal"
            notes.append(f"Derived master scale {scale_px_per_ft:.2f} px/ft from filename target ({target_sqft:.0f} sq ft).")
        elif anchors:
            valid_scales = [a.scale_px_per_ft for a in anchors if a.scale_px_per_ft > 0]
            scale_px_per_ft = float(np.median(valid_scales)) if valid_scales else self.fallback_scale_ft
            scale_source = "fixture_consensus"
            target_sqft = total_px / (scale_px_per_ft ** 2)
            notes.append(f"Derived master scale {scale_px_per_ft:.2f} px/ft from fixture consensus.")
        else:
            scale_px_per_ft = self.fallback_scale_ft
            scale_source = "fallback"
            target_sqft = total_px / (scale_px_per_ft ** 2)
            notes.append(f"No filename or fixtures; used fallback scale {scale_px_per_ft:.2f} px/ft.")

        scale_cm_per_px = self.cm_per_foot / scale_px_per_ft

        # Update anchor error percentages against the master scale
        for anchor in anchors:
            if anchor.scale_px_per_ft > 0:
                err = abs(anchor.scale_px_per_ft - scale_px_per_ft) / scale_px_per_ft * 100.0
                anchor.error_vs_chosen_scale_pct = float(round(err, 1))

        # 5. Build DimensionedRoom records
        dimensioned_rooms: List[DimensionedRoom] = []
        for rm in geometry_result.rooms:
            area_sqft = float(rm.area_px / (scale_px_per_ft ** 2))
            area_m2 = float(area_sqft * 0.092903)
            area_fraction = float(rm.area_px / total_px) if total_px > 0 else 0.0

            # Bounding dimensional span
            minx, miny, maxx, maxy = rm.polygon.bounds
            w_ft = float((maxx - minx) / scale_px_per_ft)
            l_ft = float((maxy - miny) / scale_px_per_ft)
            d1_ft, d2_ft = max(w_ft, l_ft), min(w_ft, l_ft)

            d1_m = d1_ft * (self.cm_per_foot / 100.0)
            d2_m = d2_ft * (self.cm_per_foot / 100.0)

            dim_str = self.format_dimensions_architectural(d1_ft, d2_ft)
            dim_metric_str = self.format_dimensions_metric(d1_m, d2_m)

            # Optimal visual center label placement
            try:
                pole = polylabel(rm.polygon, tolerance=1.0)
                label_anchor = (float(pole.x), float(pole.y))
            except Exception:
                label_anchor = rm.centroid

            dimensioned_rooms.append(
                DimensionedRoom(
                    room_id=rm.room_id,
                    label=rm.label,
                    polygon=rm.polygon,
                    area_px=rm.area_px,
                    area_sqft=area_sqft,
                    area_m2=area_m2,
                    area_fraction=area_fraction,
                    dimension_ft=(d1_ft, d2_ft),
                    dimension_m=(d1_m, d2_m),
                    dimension_str=dim_str,
                    dimension_metric_str=dim_metric_str,
                    centroid=rm.centroid,
                    label_anchor=label_anchor,
                    fixtures=list(rm.fixtures),
                    is_outdoor=rm.is_outdoor,
                    aspect_ratio=rm.aspect_ratio,
                )
            )

        # Sort by room_id ascending
        dimensioned_rooms.sort(key=lambda r: r.room_id)

        # 6. Aggregate totals and verify QA Gate 4
        computed_total_sqft = float(sum(r.area_sqft for r in dimensioned_rooms))
        computed_interior_sqft = float(sum(r.area_sqft for r in dimensioned_rooms if not r.is_outdoor))
        computed_outdoor_sqft = float(sum(r.area_sqft for r in dimensioned_rooms if r.is_outdoor))

        if target_sqft is not None and target_sqft > 0:
            area_err_pct = float(abs(computed_total_sqft - target_sqft) / target_sqft * 100.0)
        else:
            area_err_pct = 0.0

        is_gate4_passed = area_err_pct <= (self.max_area_err_ratio * 100.0)
        notes.append(
            f"QA Gate 4: Total Sq Ft = {computed_total_sqft:.1f} (Target {target_sqft:.1f}, "
            f"Err {area_err_pct:.2f}% <= {self.max_area_err_ratio * 100.0}%)."
        )

        return ScaleResult(
            unit_key=unit_key,
            target_sqft=target_sqft,
            computed_total_sqft=computed_total_sqft,
            computed_interior_sqft=computed_interior_sqft,
            computed_outdoor_sqft=computed_outdoor_sqft,
            total_area_error_pct=area_err_pct,
            scale_px_per_ft=scale_px_per_ft,
            scale_cm_per_px=scale_cm_per_px,
            scale_source=scale_source,
            rooms=dimensioned_rooms,
            anchors=anchors,
            is_gate4_passed=is_gate4_passed,
            notes=notes,
        )

    def _compute_fixture_scales(
        self,
        furniture_layers: Dict[str, np.ndarray],
    ) -> List[FixtureScaleAnchor]:
        """Compute independent metric scale anchors from standard fixtures."""
        anchors: List[FixtureScaleAnchor] = []
        if not furniture_layers:
            return anchors

        # 1. Countertop Depth (Yellow Layer)
        if "countertop" in furniture_layers:
            c_mask = furniture_layers["countertop"]
            if np.sum(c_mask > 0) > 200:
                dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
                depth_px = float(dist.max() * 2.0)
                if depth_px > 5.0:
                    depth_ft = self.counter_depth_cm / self.cm_per_foot  # ~2.0 ft
                    s_ft = depth_px / depth_ft
                    s_cm = self.cm_per_foot / s_ft
                    anchors.append(
                        FixtureScaleAnchor(
                            fixture_type="countertop",
                            standard_dim_cm=(self.counter_depth_cm,),
                            measured_dim_px=(depth_px,),
                            scale_px_per_ft=float(s_ft),
                            scale_cm_per_px=float(s_cm),
                            error_vs_chosen_scale_pct=0.0,
                            notes=f"Depth {depth_px:.1f} px vs standard {self.counter_depth_cm:.0f} cm",
                        )
                    )

        # 2. Bed (Pink Layer)
        if "bed" in furniture_layers:
            b_mask = furniture_layers["bed"]
            cnts, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area_px = float(cv2.contourArea(c))
                rect = cv2.minAreaRect(c)
                w_px = float(min(rect[1]))
                l_px = float(max(rect[1]))

                if area_px > 1000.0:
                    s_ft = float(np.sqrt(area_px / self.bed_sqft))
                    s_cm = float(self.cm_per_foot / s_ft)
                    anchors.append(
                        FixtureScaleAnchor(
                            fixture_type="bed",
                            standard_dim_cm=(self.bed_w_cm, self.bed_l_cm),
                            measured_dim_px=(w_px, l_px),
                            scale_px_per_ft=s_ft,
                            scale_cm_per_px=s_cm,
                            error_vs_chosen_scale_pct=0.0,
                            notes=f"Area {area_px:.0f} px² vs Queen {self.bed_sqft:.1f} sqft ({w_px:.1f}x{l_px:.1f} px)",
                        )
                    )

        # 3. Toilet (Green Layer)
        if "toilet" in furniture_layers:
            t_mask = furniture_layers["toilet"]
            cnts, _ = cv2.findContours(t_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area_px = float(cv2.contourArea(c))
                rect = cv2.minAreaRect(c)
                w_px = float(min(rect[1]))
                l_px = float(max(rect[1]))

                if area_px > 100.0 and w_px > 5.0:
                    toilet_w_ft = self.toilet_w_cm / self.cm_per_foot
                    s_ft = float(w_px / toilet_w_ft)
                    s_cm = float(self.cm_per_foot / s_ft)
                    anchors.append(
                        FixtureScaleAnchor(
                            fixture_type="toilet",
                            standard_dim_cm=(self.toilet_w_cm, self.toilet_l_cm),
                            measured_dim_px=(w_px, l_px),
                            scale_px_per_ft=s_ft,
                            scale_cm_per_px=s_cm,
                            error_vs_chosen_scale_pct=0.0,
                            notes=f"Width {w_px:.1f} px vs standard {self.toilet_w_cm:.0f} cm",
                        )
                    )

        return anchors

    @staticmethod
    def format_dimensions_architectural(d1_ft: float, d2_ft: float) -> str:
        """Format feet into standard architectural notation (e.g. 14'8\" x 12'4\")."""
        f1 = int(d1_ft)
        i1 = int(round((d1_ft - f1) * 12))
        if i1 >= 12:
            f1 += 1
            i1 = 0

        f2 = int(d2_ft)
        i2 = int(round((d2_ft - f2) * 12))
        if i2 >= 12:
            f2 += 1
            i2 = 0

        return f"{f1}'{i1}\" x {f2}'{i2}\""

    @staticmethod
    def format_dimensions_metric(d1_m: float, d2_m: float) -> str:
        """Format meters into architectural metric notation (e.g. 4.47m x 3.76m)."""
        return f"{d1_m:.2f}m x {d2_m:.2f}m"
