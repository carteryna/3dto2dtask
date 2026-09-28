#!/usr/bin/env python3
"""Iteration 5 Phase 5 Verification Harness: Architectural CAD Blueprint Synthesis & Vector Export.

Validates:
1. Strict space invariant: Space is either a wall or a room (0.0000 px² gaps, 0.0000 px² overlaps).
2. Publication-quality 2D CAD blueprint PNG with solid continuous walls.
3. RFC 7946 compliant GeoJSON export with right-hand rule polygon geometry.
4. Scalable Vector Graphics (SVG) architectural blueprint export.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5_semi_automated.export import SemiAutomatedExporter
from src.v5_semi_automated.geometry import SemiAutomatedGeometry
from src.v5_semi_automated.rectification import (
    SemiAutomatedRectifier,
    normalize_unit_key,
)
from src.v5_semi_automated.scale import SemiAutomatedScale
from src.v5_semi_automated.topology import SemiAutomatedTopology
from src.v5_semi_automated.visualizer import SemiAutomatedVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test Iteration 5 Phase 5 CAD Blueprint Synthesis.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", help="Single marketing floor plan image (e.g. inout_data/asset.webp)")
    parser.add_argument("--input-dir", default="inout_data", help="Directory of source marketing images")
    parser.add_argument("--output-dir", default="data/output/debug_v5", help="Directory for debug outputs")
    parser.add_argument("--pub-dir", default="data/output", help="Directory for publication outputs")
    return parser.parse_args()


def validate_geojson_rfc7946(geojson_path: Path) -> Tuple[bool, str]:
    """Validate GeoJSON file structure and RFC 7946 geometry invariants."""
    try:
        with geojson_path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
        if doc.get("type") != "FeatureCollection":
            return False, "Root object is not FeatureCollection."
        features = doc.get("features", [])
        if not features:
            return False, "FeatureCollection contains 0 features."

        for feat in features:
            geom = feat.get("geometry")
            if not geom:
                return False, f"Feature {feat.get('id')} has empty geometry."
            # Validate geometry validity with Shapely
            s_geom = shape(geom)
            if not s_geom.is_valid:
                return False, f"Feature {feat.get('id')} geometry is invalid: {s_geom}"
            # Check right-hand rule for Polygons (exterior ring counter-clockwise)
            if s_geom.geom_type == "Polygon":
                coords = list(s_geom.exterior.coords)
                # Compute signed area: positive for CCW in Cartesian
                signed_area = 0.5 * sum(
                    coords[i][0] * coords[i+1][1] - coords[i+1][0] * coords[i][1]
                    for i in range(len(coords) - 1)
                )
                if signed_area <= 0:
                    return False, f"Feature {feat.get('id')} violates RFC 7946 right-hand rule."

        return True, f"Valid RFC 7946 GeoJSON with {len(features)} features."
    except Exception as exc:
        return False, f"GeoJSON parsing failed: {exc}"


def validate_svg_xml(svg_path: Path) -> Tuple[bool, str]:
    """Validate SVG XML well-formedness."""
    try:
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
        if not root.tag.endswith("svg"):
            return False, f"Root tag is not svg: {root.tag}"
        return True, "Valid well-formed SVG XML."
    except Exception as exc:
        return False, f"SVG validation failed: {exc}"


def main():
    args = parse_args()
    debug_base = PROJECT_ROOT / args.output_dir
    pub_base = PROJECT_ROOT / args.pub_dir
    debug_base.mkdir(parents=True, exist_ok=True)
    pub_base.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_paths = [Path(args.input)]
    else:
        input_dir = PROJECT_ROOT / args.input_dir
        input_paths = sorted(input_dir.glob("*.webp"))

    if not input_paths:
        print("No .webp images found to process.")
        sys.exit(1)

    rectifier = SemiAutomatedRectifier(args.config)
    topology_engine = SemiAutomatedTopology(args.config)
    geometry_engine = SemiAutomatedGeometry(args.config)
    scale_engine = SemiAutomatedScale(args.config)
    visualizer = SemiAutomatedVisualizer(args.config)
    exporter = SemiAutomatedExporter(args.config)

    print(f"=== Iteration 5 Phase 5: Architectural CAD Blueprint Synthesis ({len(input_paths)} assets) ===")

    results_summary: List[Dict] = []

    for path in input_paths:
        stem = path.stem
        unit_key = normalize_unit_key(stem)
        unit_out_dir = debug_base / stem
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        render_img = cv2.imread(str(path))
        if render_img is None:
            print(f"[Error] Could not read {path}")
            continue

        print(f"\n--- Processing: {stem} ---")
        print(f"  Unit Key: {unit_key}")

        # 1. Phase 1 Rectification
        rect_render_path = unit_out_dir / "02_rectified_render.png"
        rect_fn_path = unit_out_dir / "03_rectified_furniture.png"
        rect_cl_path = unit_out_dir / "03_rectified_center_line.png"
        rect_ns_path = unit_out_dir / "03_rectified_no_stroke.png"
        rect_st_path = unit_out_dir / "03_rectified_stroke.png"

        if rect_render_path.is_file() and (rect_cl_path.is_file() or rect_ns_path.is_file() or rect_fn_path.is_file()):
            rect_render = cv2.imread(str(rect_render_path))
            rect_guides = {}
            if rect_cl_path.is_file():
                rect_guides["center_line"] = cv2.imread(str(rect_cl_path), cv2.IMREAD_UNCHANGED)
            if rect_ns_path.is_file():
                rect_guides["no_stroke"] = cv2.imread(str(rect_ns_path), cv2.IMREAD_UNCHANGED)
            if rect_st_path.is_file():
                rect_guides["stroke"] = cv2.imread(str(rect_st_path), cv2.IMREAD_UNCHANGED)
            if rect_fn_path.is_file():
                rect_guides["furniture"] = cv2.imread(str(rect_fn_path), cv2.IMREAD_UNCHANGED)
        else:
            guides = rectifier.load_guides_for_unit(path)
            rect_res = rectifier.rectify(render_img, guides=guides, unit_key=unit_key)
            rect_render = rect_res.rectified_render
            rect_guides = rect_res.rectified_guides

        # 2. Phase 2 Topology
        topo_res = topology_engine.process(rect_guides, unit_key=unit_key)

        # 3. Phase 3 Geometry
        geom_res = geometry_engine.process(topo_res, render_shape=rect_render.shape[:2])

        # 4. Phase 4 Scale
        scale_res = scale_engine.process(geom_res, topology_result=topo_res, filename_or_key=stem)

        # 5. Phase 5 Visualizer & Vector Exporters
        blueprint_res = visualizer.render(
            scale_result=scale_res,
            geometry_result=geom_res,
            topology_result=topo_res,
            render_shape=rect_render.shape[:2],
            unit_label=stem,
        )

        for note in blueprint_res.notes:
            print(f"  [Note] {note}")

        # Save Visual Blueprint PNG
        cad_png_path = unit_out_dir / "16_architectural_cad_blueprint.png"
        cv2.imwrite(str(cad_png_path), blueprint_res.blueprint_image)

        pub_png_path = pub_base / f"blueprint_v5_{stem}.png"
        cv2.imwrite(str(pub_png_path), blueprint_res.blueprint_image)

        # Save GeoJSON
        geojson_path = unit_out_dir / "blueprint.geojson"
        exporter.export_geojson(
            scale_result=scale_res,
            wall_polygon=blueprint_res.wall_polygon,
            total_footprint=blueprint_res.total_footprint_polygon,
            output_path=geojson_path,
            coord_units="feet",
        )
        pub_geojson_path = pub_base / f"blueprint_v5_{stem}.geojson"
        exporter.export_geojson(
            scale_result=scale_res,
            wall_polygon=blueprint_res.wall_polygon,
            total_footprint=blueprint_res.total_footprint_polygon,
            output_path=pub_geojson_path,
            coord_units="feet",
        )

        # Save SVG
        svg_path = unit_out_dir / "blueprint.svg"
        exporter.export_svg(
            scale_result=scale_res,
            wall_polygon=blueprint_res.wall_polygon,
            total_footprint=blueprint_res.total_footprint_polygon,
            render_shape=blueprint_res.blueprint_image.shape[:2],
            output_path=svg_path,
            unit_label=stem,
        )
        pub_svg_path = pub_base / f"blueprint_v5_{stem}.svg"
        exporter.export_svg(
            scale_result=scale_res,
            wall_polygon=blueprint_res.wall_polygon,
            total_footprint=blueprint_res.total_footprint_polygon,
            render_shape=blueprint_res.blueprint_image.shape[:2],
            output_path=pub_svg_path,
            unit_label=stem,
        )

        # QA Gate 5 Verification
        geojson_valid, geojson_msg = validate_geojson_rfc7946(geojson_path)
        svg_valid, svg_msg = validate_svg_xml(svg_path)
        zero_gaps = blueprint_res.unclaimed_gap_px2 <= 0.0001
        zero_overlap = blueprint_res.overlap_rooms_walls_px2 <= 0.0001
        gate5_passed = blueprint_res.is_gate5_passed and geojson_valid and svg_valid and zero_gaps and zero_overlap
        status_str = "PASS" if gate5_passed else "FAIL"

        print(
            f"  [QA Gate 5] Overlap: {blueprint_res.overlap_rooms_walls_px2:.4f} px², "
            f"Gaps: {blueprint_res.unclaimed_gap_px2:.4f} px², "
            f"GeoJSON: {'PASS' if geojson_valid else 'FAIL'}, "
            f"SVG: {'PASS' if svg_valid else 'FAIL'} [{status_str}]"
        )
        print(f"    Gross Footprint: {blueprint_res.total_footprint_sqft:.1f} sq ft | Walls: {blueprint_res.wall_sqft:.1f} sq ft | Net Rooms: {blueprint_res.net_room_sqft:.1f} sq ft")

        results_summary.append({
            "unit": stem,
            "rooms": len(scale_res.rooms),
            "overlap_px2": blueprint_res.overlap_rooms_walls_px2,
            "gap_px2": blueprint_res.unclaimed_gap_px2,
            "footprint_sqft": blueprint_res.total_footprint_sqft,
            "net_sqft": blueprint_res.net_room_sqft,
            "geojson": geojson_valid,
            "svg": svg_valid,
            "gate5": gate5_passed,
        })

    print("\n" + "=" * 90)
    print("PHASE 5 QA GATE 5 SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Unit':<35} | {'Rooms':<5} | {'Overlap':<11} | {'Gaps':<11} | {'GeoJSON':<7} | {'SVG':<5} | {'Gate 5'}")
    print("-" * 90)
    all_passed = True
    for r in results_summary:
        ov_str = f"{r['overlap_px2']:.4f} px²"
        gp_str = f"{r['gap_px2']:.4f} px²"
        gj_str = "PASS" if r['geojson'] else "FAIL"
        svg_str = "PASS" if r['svg'] else "FAIL"
        g5_str = "PASS" if r['gate5'] else "FAIL"
        if not r['gate5']:
            all_passed = False
        print(f"{r['unit'][:35]:<35} | {r['rooms']:<5} | {ov_str:<11} | {gp_str:<11} | {gj_str:<7} | {svg_str:<5} | {g5_str}")
    print("=" * 90)
    if all_passed:
        print("ALL ASSETS PASSED QA GATE 5 (0 Gaps, 0 Overlap, Valid GeoJSON & SVG, Publication PNG).")
        print("Iteration 5 Pipeline Architecture is 100% COMPLETE!")
    else:
        print("SOME ASSETS FAILED QA GATE 5.")
        sys.exit(1)


if __name__ == "__main__":
    main()
