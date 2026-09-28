"""Phase 5: Architectural Vector Exporters (RFC 7946 GeoJSON & 2D SVG).

Exports regularized 2D CAD floor plans into structured, standard vector formats:
- RFC 7946 compliant GeoJSON FeatureCollection (right-hand rule, strict validity).
- Layered Scalable Vector Graphics (SVG) blueprint with embedded CSS styling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.geometry.polygon import orient

from src.v5_semi_automated.scale import DimensionedRoom, ScaleResult


class SemiAutomatedExporter:
    """Exports regularized architectural CAD layouts to GeoJSON and SVG."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        pass

    def export_geojson(
        self,
        scale_result: ScaleResult,
        wall_polygon: Polygon | MultiPolygon,
        total_footprint: Polygon,
        output_path: str | Path,
        coord_units: str = "feet",
    ) -> Dict:
        """Export RFC 7946 compliant GeoJSON FeatureCollection.

        Args:
            scale_result: Phase 4 dimensional rooms and scale data.
            wall_polygon: Solid walls Shapely geometry.
            total_footprint: Total unit exterior boundary polygon.
            output_path: Path to write .geojson file.
            coord_units: 'feet' (scaled to architectural coordinates) or 'pixels'.

        Returns:
            GeoJSON FeatureCollection dictionary.
        """
        scale = scale_result.scale_px_per_ft if coord_units == "feet" and scale_result.scale_px_per_ft > 0 else 1.0

        def transform_poly(poly: Polygon) -> Polygon:
            """Ensure RFC 7946 right-hand rule and scale coordinates."""
            if poly.is_empty:
                return poly
            # Orient counter-clockwise exterior, clockwise holes
            oriented = orient(poly, sign=1.0)
            ext_coords = [(round(x / scale, 3), round(y / scale, 3)) for x, y in oriented.exterior.coords]
            inter_coords = [
                [(round(x / scale, 3), round(y / scale, 3)) for x, y in hole.coords]
                for hole in oriented.interiors
            ]
            return Polygon(ext_coords, inter_coords)

        features: List[Dict] = []

        # 1. Total Footprint Feature
        footprint_tf = transform_poly(total_footprint)
        features.append({
            "type": "Feature",
            "id": "footprint",
            "properties": {
                "feature_type": "building_envelope",
                "unit_key": scale_result.unit_key,
                "gross_area_sqft": round(float(total_footprint.area / (scale_result.scale_px_per_ft ** 2)), 2),
                "target_sqft": scale_result.target_sqft,
                "units": coord_units,
            },
            "geometry": mapping(footprint_tf),
        })

        # 2. Wall Features
        wall_geoms = [wall_polygon] if isinstance(wall_polygon, Polygon) else wall_polygon.geoms
        for idx, wg in enumerate(wall_geoms, start=1):
            if wg.is_empty:
                continue
            wall_tf = transform_poly(wg)
            features.append({
                "type": "Feature",
                "id": f"wall_{idx}",
                "properties": {
                    "feature_type": "wall",
                    "wall_type": "structural_partition",
                    "area_sqft": round(float(wg.area / (scale_result.scale_px_per_ft ** 2)), 2),
                    "units": coord_units,
                },
                "geometry": mapping(wall_tf),
            })

        # 3. Room Features
        for rm in scale_result.rooms:
            rm_tf = transform_poly(rm.polygon)
            features.append({
                "type": "Feature",
                "id": f"room_{rm.room_id}",
                "properties": {
                    "feature_type": "room",
                    "room_id": rm.room_id,
                    "label": rm.label,
                    "area_sqft": round(rm.area_sqft, 2),
                    "area_m2": round(rm.area_m2, 2),
                    "dimension_ft": rm.dimension_str,
                    "dimension_metric": rm.dimension_metric_str,
                    "span_length_ft": round(rm.dimension_ft[0], 2),
                    "span_width_ft": round(rm.dimension_ft[1], 2),
                    "span_length_m": round(rm.dimension_m[0], 2),
                    "span_width_m": round(rm.dimension_m[1], 2),
                    "is_outdoor": rm.is_outdoor,
                    "fixtures": rm.fixtures,
                    "units": coord_units,
                },
                "geometry": mapping(rm_tf),
            })

        geojson_doc = {
            "type": "FeatureCollection",
            "name": f"{scale_result.unit_key}_architectural_cad_floorplan",
            "crs": {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:OGC:1.3:CRS84 ({coord_units})"},
            },
            "features": features,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(geojson_doc, handle, indent=2)

        return geojson_doc

    def export_svg(
        self,
        scale_result: ScaleResult,
        wall_polygon: Polygon | MultiPolygon,
        total_footprint: Polygon,
        render_shape: Tuple[int, int],
        output_path: str | Path,
        unit_label: Optional[str] = None,
        header_height: int = 90,
    ) -> str:
        """Export crisp, layered SVG architectural blueprint.

        Args:
            scale_result: Phase 4 dimensional rooms and scale data.
            wall_polygon: Solid walls Shapely geometry.
            total_footprint: Total unit exterior boundary polygon.
            render_shape: (height, width) canvas size.
            output_path: Path to write .svg file.
            unit_label: Formatted display title.
            header_height: Height in pixels of top architectural banner.

        Returns:
            Complete SVG XML string.
        """
        h, w = render_shape[:2]
        display_name = unit_label if unit_label else scale_result.unit_key.replace("_", " ").title()
        header_h = header_height

        svg_lines: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            '  <defs>',
            '    <style>',
            '      .bg-paper { fill: #fdfdfd; }',
            '      .grid-line { stroke: #ededed; stroke-width: 1px; }',
            '      .wall-solid { fill: #2c2c2c; stroke: #141414; stroke-width: 2px; fill-rule: evenodd; }',
            '      .room-indoor { fill: #ffffff; stroke: #404040; stroke-width: 1px; }',
            '      .room-outdoor { fill: #f4faff; stroke: #404040; stroke-width: 1px; }',
            '      .badge-box { fill: #ffffff; stroke: #b0b0b0; stroke-width: 1px; rx: 2px; }',
            '      .badge-title { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 8.5px; font-weight: bold; fill: #111111; text-anchor: middle; }',
            '      .badge-area { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 7.5px; fill: #505050; text-anchor: middle; }',
            '      .badge-dim { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 7.5px; font-weight: 600; fill: #9e2a00; text-anchor: middle; }',
            '      .decking-line { stroke: #d0e4f0; stroke-width: 1.5px; }',
            '    </style>',
            '  </defs>',
            '  <!-- Sheet Background -->',
            f'  <rect width="{w}" height="{h}" class="bg-paper" />',
            '  <!-- Drafting Grid -->',
            '  <g id="drafting-grid">',
        ]

        # 1. Drafting Grid below header
        grid_step = 40
        for x in range(0, w, grid_step):
            svg_lines.append(f'    <line x1="{x}" y1="{header_h}" x2="{x}" y2="{h}" class="grid-line" />')
        for y in range(header_h, h, grid_step):
            svg_lines.append(f'    <line x1="0" y1="{y}" x2="{w}" y2="{y}" class="grid-line" />')
        svg_lines.append('  </g>')

        # 2. Solid Walls
        def poly_to_path(poly: Polygon) -> str:
            if poly.is_empty:
                return ""
            parts = []
            ext_str = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in poly.exterior.coords) + " Z"
            parts.append(ext_str)
            for interior in poly.interiors:
                int_str = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in interior.coords) + " Z"
                parts.append(int_str)
            return " ".join(parts)

        svg_lines.append('  <!-- Solid Structural Walls (Zero Unclaimed Gaps) -->')
        svg_lines.append('  <g id="structural-walls">')
        wall_geoms = [wall_polygon] if isinstance(wall_polygon, Polygon) else wall_polygon.geoms
        for idx, wg in enumerate(wall_geoms, start=1):
            path_d = poly_to_path(wg)
            if path_d:
                svg_lines.append(f'    <path id="wall_{idx}" d="{path_d}" class="wall-solid" />')
        svg_lines.append('  </g>')

        # 3. Rooms
        svg_lines.append('  <!-- Architectural Rooms -->')
        svg_lines.append('  <g id="architectural-rooms">')
        for rm in scale_result.rooms:
            path_d = poly_to_path(rm.polygon)
            cls = "room-outdoor" if rm.is_outdoor else "room-indoor"
            svg_lines.append(f'    <path id="room_{rm.room_id}" d="{path_d}" class="{cls}" />')
            # Balcony decking slats
            if rm.is_outdoor or "balcony" in rm.fixtures:
                minx, miny, maxx, maxy = rm.polygon.bounds
                for y in range(int(miny) + 6, int(maxy) - 4, 8):
                    svg_lines.append(f'    <line x1="{minx + 4:.1f}" y1="{y:.1f}" x2="{maxx - 4:.1f}" y2="{y:.1f}" class="decking-line" />')
        svg_lines.append('  </g>')

        # 4. Dimension Badges
        svg_lines.append('  <!-- Room Dimension Callout Badges -->')
        svg_lines.append('  <g id="room-labels">')
        for rm in scale_result.rooms:
            ax, ay = rm.label_anchor[0], rm.label_anchor[1]
            is_compact = rm.area_sqft < 35.0
            if is_compact:
                tag1 = f"Room {rm.room_id}"
                if rm.fixtures:
                    tag1 += f" ({rm.fixtures[0]})"
                tag2 = f"{rm.area_sqft:.0f} sf | {rm.dimension_str}"
                bw = max(len(tag1), len(tag2)) * 5.2 + 10
                bh = 22
                bx = ax - bw / 2.0
                by = ay - bh / 2.0
                svg_lines.append(f'    <g id="badge_{rm.room_id}">')
                svg_lines.append(f'      <rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" class="badge-box" />')
                svg_lines.append(f'      <text x="{ax:.1f}" y="{by + 9:.1f}" class="badge-title">{tag1}</text>')
                svg_lines.append(f'      <text x="{ax:.1f}" y="{by + 18:.1f}" class="badge-dim">{tag2}</text>')
                svg_lines.append('    </g>')
            else:
                tag1 = f"Room {rm.room_id}"
                if rm.fixtures:
                    tag1 += f" ({rm.fixtures[0]})"
                elif rm.is_outdoor:
                    tag1 += " (Balcony)"
                tag2 = f"{rm.area_sqft:.0f} SQ FT ({rm.area_m2:.1f} m²)"
                tag3 = f"{rm.dimension_str}"
                bw = max(len(tag1), len(tag2), len(tag3)) * 5.2 + 12
                bh = 32
                bx = ax - bw / 2.0
                by = ay - bh / 2.0
                svg_lines.append(f'    <g id="badge_{rm.room_id}">')
                svg_lines.append(f'      <rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" class="badge-box" />')
                svg_lines.append(f'      <text x="{ax:.1f}" y="{by + 10:.1f}" class="badge-title">{tag1}</text>')
                svg_lines.append(f'      <text x="{ax:.1f}" y="{by + 20:.1f}" class="badge-area">{tag2}</text>')
                svg_lines.append(f'      <text x="{ax:.1f}" y="{by + 29:.1f}" class="badge-dim">{tag3}</text>')
                svg_lines.append('    </g>')
        svg_lines.append('  </g>')

        # 5. Architectural CAD Header (strictly above plan)
        net_sqft = sum(rm.area_sqft for rm in scale_result.rooms if not rm.is_outdoor and "balcony" not in rm.fixtures)
        balc_sqft = sum(rm.area_sqft for rm in scale_result.rooms if rm.is_outdoor or "balcony" in rm.fixtures)
        gross_sqft = float(total_footprint.area / (scale_result.scale_px_per_ft ** 2)) if scale_result.scale_px_per_ft > 0 else 0.0

        arrow_x = 35 if w >= 900 else 26
        arrow_y = header_h // 2
        r = 16

        svg_lines.extend([
            '  <!-- Architectural Header Banner (strictly above plan) -->',
            '  <g id="blueprint-header">',
            f'    <rect x="0" y="0" width="{w}" height="{header_h}" fill="#222222" />',
            f'    <line x1="0" y1="{header_h - 2}" x2="{w}" y2="{header_h - 2}" stroke="#373737" stroke-width="1px" />',
            f'    <line x1="0" y1="{header_h - 1}" x2="{w}" y2="{header_h - 1}" stroke="#D04E18" stroke-width="2px" />',
            '    <!-- North Indicator -->',
            f'    <circle cx="{arrow_x}" cy="{arrow_y}" r="{r}" fill="none" stroke="#787878" stroke-width="1px" />',
            f'    <polygon points="{arrow_x},{arrow_y - r + 2} {arrow_x},{arrow_y + r - 2} {arrow_x - 5},{arrow_y}" fill="#e6e6e6" />',
            f'    <polygon points="{arrow_x},{arrow_y - r + 2} {arrow_x},{arrow_y + r - 2} {arrow_x + 5},{arrow_y}" fill="none" stroke="#e6e6e6" stroke-width="1px" />',
            f'    <text x="{arrow_x}" y="{arrow_y - r - 2}" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="8px" font-weight="bold" fill="#f0f0f0" text-anchor="middle">N</text>',
        ])

        if w >= 900:
            tx = 68
            svg_lines.extend([
                f'    <text x="{tx}" y="26" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="9px" font-weight="bold" fill="#aaaaaa" letter-spacing="1px">ARCHITECTURAL 2D BLUEPRINT</text>',
                f'    <text x="{tx}" y="52" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="14px" font-weight="bold" fill="#ffffff">{display_name.upper()[:45]}</text>',
                f'    <text x="{tx}" y="73" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="9.5px" font-weight="bold" fill="#50a0ff">ZERO WALL GAPS (MANHATTAN CAD)  |  SCALE: 1\' = {scale_result.scale_px_per_ft:.2f} px</text>',
            ])
            # Four metric cards
            metrics = [
                ("GROSS", f"{gross_sqft:.1f} SQ FT"),
                ("BALCONY", f"{balc_sqft:.1f} SQ FT"),
                ("NET INT", f"{net_sqft:.1f} SQ FT"),
                ("TARGET", f"{scale_result.target_sqft:.0f} SQ FT"),
            ]
            rx = w - 20
            for label, val in metrics:
                bw = 100
                bx = rx - bw
                cx = bx + bw / 2
                svg_lines.extend([
                    f'    <rect x="{bx}" y="16" width="{bw}" height="{header_h - 32}" rx="3" fill="#303030" stroke="#4b4b4b" stroke-width="1px" />',
                    f'    <text x="{cx}" y="33" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="8px" fill="#a0a0a0" text-anchor="middle">{label}</text>',
                    f'    <text x="{cx}" y="55" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="10.5px" font-weight="bold" fill="#ffffff" text-anchor="middle">{val}</text>',
                ])
                rx = bx - 10
        else:
            tx = 52
            stats_str = f"TARGET: {scale_result.target_sqft:.0f} sf  |  NET: {net_sqft:.1f} sf  |  BALC: {balc_sqft:.1f} sf  |  GROSS: {gross_sqft:.1f} sf"
            svg_lines.extend([
                f'    <text x="{tx}" y="22" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="8px" font-weight="bold" fill="#aaaaaa" letter-spacing="0.5px">ARCHITECTURAL 2D BLUEPRINT</text>',
                f'    <text x="{tx}" y="42" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12px" font-weight="bold" fill="#ffffff">{display_name.upper()[:34]}</text>',
                f'    <text x="{tx}" y="60" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="8.5px" fill="#d7d7d7">{stats_str}</text>',
                f'    <text x="{tx}" y="76" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="8.5px" font-weight="bold" fill="#50a0ff">SCALE: 1\' = {scale_result.scale_px_per_ft:.2f} px  |  ZERO WALL GAPS CAD</text>',
            ])

        svg_lines.extend([
            '  </g>',
            '</svg>',
        ])

        svg_content = "\n".join(svg_lines)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write(svg_content)

        return svg_content
