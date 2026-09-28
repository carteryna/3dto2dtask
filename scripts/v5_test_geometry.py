"""Visual and quantitative verification script for Iteration 5 Phase 3: Manhattan Planar Graph Regularization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v5_semi_automated.rectification import (
    SemiAutomatedRectifier,
    normalize_unit_key,
)
from src.v5_semi_automated.topology import SemiAutomatedTopology
from src.v5_semi_automated.geometry import (
    RegularizedRoomNode,
    SemiAutomatedGeometry,
)

# 12 distinct, aesthetically pleasing pastel room colors (BGR)
ROOM_COLORS: List[Tuple[int, int, int]] = [
    (240, 200, 160),  # Soft Light Blue
    (180, 230, 180),  # Soft Mint
    (200, 180, 240),  # Soft Rose / Lavender
    (170, 230, 240),  # Soft Yellow
    (230, 190, 220),  # Soft Purple
    (200, 230, 200),  # Soft Sage
    (240, 220, 180),  # Soft Sky
    (190, 210, 240),  # Soft Peach
    (220, 220, 170),  # Soft Aqua
    (210, 180, 210),  # Soft Mauve
    (180, 240, 220),  # Soft Lemon
    (230, 210, 240),  # Soft Lilac
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Iteration 5 Phase 3 Manhattan Planar Regularization.")
    parser.add_argument("--input", default=None, help="Path to single floor-plan image (.webp).")
    parser.add_argument("--input_dir", default="inout_data", help="Directory of floor-plan images.")
    parser.add_argument("--output_dir", default="data/output/debug_v5", help="Directory for debug outputs.")
    parser.add_argument("--config", default=None, help="Path to configs/config.yaml.")
    return parser.parse_args()


def render_manhattan_polygons(
    bg_image: np.ndarray,
    rooms: List[RegularizedRoomNode],
    alpha: float = 0.55,
) -> np.ndarray:
    """Render pastel colored fills for regularized Manhattan polygons with solid borders and callouts."""
    out = bg_image.copy()
    h, w = bg_image.shape[:2]
    color_layer = np.zeros_like(bg_image)

    for i, rm in enumerate(rooms):
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        coords = np.array(rm.polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(color_layer, [coords], color)

    mask_any = np.any(color_layer > 0, axis=-1)
    out[mask_any] = cv2.addWeighted(out, 1.0 - alpha, color_layer, alpha, 0)[mask_any]

    # Draw solid black boundary lines and room badges
    for i, rm in enumerate(rooms):
        coords = np.array(rm.polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(out, [coords], isClosed=True, color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)

        cx, cy = int(round(rm.centroid[0])), int(round(rm.centroid[1]))
        tag = f"R{rm.room_id}"
        if rm.fixtures:
            tag += f" ({','.join(rm.fixtures)})"
        elif rm.is_outdoor:
            tag += " (Balcony)"

        sub_tag = f"{int(round(rm.area_px)):,} px"

        # Text badge background
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thick = 1
        (t_w, t_h), _ = cv2.getTextSize(tag, font, scale, thick)
        (s_w, s_h), _ = cv2.getTextSize(sub_tag, font, scale - 0.05, thick)

        badge_w = max(t_w, s_w) + 12
        badge_h = t_h + s_h + 12
        bx1 = max(0, cx - badge_w // 2)
        by1 = max(0, cy - badge_h // 2)
        bx2 = min(w - 1, bx1 + badge_w)
        by2 = min(h - 1, by1 + badge_h)

        cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (50, 50, 50), 1, cv2.LINE_AA)

        text_x1 = bx1 + (badge_w - t_w) // 2
        text_y1 = by1 + t_h + 3
        text_x2 = bx1 + (badge_w - s_w) // 2
        text_y2 = text_y1 + s_h + 4

        cv2.putText(out, tag, (text_x1, text_y1), font, scale, (10, 10, 10), thick, cv2.LINE_AA)
        cv2.putText(out, sub_tag, (text_x2, text_y2), font, scale - 0.05, (100, 100, 100), thick, cv2.LINE_AA)

    return out


def render_shared_wall_wireframe(
    render_shape: Tuple[int, int],
    rooms: List[RegularizedRoomNode],
) -> np.ndarray:
    """Render 1D vector wireframe of Manhattan shared walls."""
    h, w = render_shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 248  # Off-white CAD canvas

    # Draw grid lines for architectural aesthetics
    for x in range(0, w, 50):
        cv2.line(canvas, (x, 0), (x, h), (235, 235, 235), 1)
    for y in range(0, h, 50):
        cv2.line(canvas, (0, y), (w, y), (235, 235, 235), 1)

    # Draw room edges
    for rm in rooms:
        coords = np.array(rm.polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(canvas, [coords], isClosed=True, color=(40, 40, 40), thickness=2, lineType=cv2.LINE_AA)

        # Plot corner nodes
        for pt in coords[:-1]:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, (0, 100, 255), -1, cv2.LINE_AA)

    return canvas


def render_manhattan_fixtures_overlay(
    bg_image: np.ndarray,
    rooms: List[RegularizedRoomNode],
    furniture_layers: Dict[str, np.ndarray],
) -> np.ndarray:
    """Render Manhattan polygons overlaid with colored fixture fills."""
    out = bg_image.copy()

    # Draw room polygons
    for i, rm in enumerate(rooms):
        coords = np.array(rm.polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(out, [coords], isClosed=True, color=(60, 60, 60), thickness=2, lineType=cv2.LINE_AA)

    # Overlay fixtures
    colors = {
        "bed": (203, 100, 255),       # Pink
        "countertop": (0, 230, 255),   # Yellow
        "toilet": (0, 255, 120),       # Green
        "balcony": (255, 150, 0),      # Blue
    }
    color_layer = np.zeros_like(bg_image)
    for k, col in colors.items():
        m = furniture_layers.get(k)
        if m is not None and np.sum(m > 0) > 0:
            color_layer[m > 0] = col

    mask_any = np.any(color_layer > 0, axis=-1)
    out[mask_any] = cv2.addWeighted(out, 0.35, color_layer, 0.65, 0)[mask_any]
    return out


def main():
    args = parse_args()
    output_base = PROJECT_ROOT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

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

    print(f"=== Iteration 5 Phase 3: Manhattan Planar Regularization ({len(input_paths)} assets) ===")

    results_summary = []

    for path in input_paths:
        stem = path.stem
        unit_key = normalize_unit_key(stem)
        unit_out_dir = output_base / stem
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

        # 3. Phase 3 Manhattan Regularization
        geom_res = geometry_engine.process(topo_res, render_shape=rect_render.shape[:2])

        for note in geom_res.notes:
            print(f"  [Note] {note}")

        # Gate 3 QA Verification
        gate3_passed = geom_res.is_gate3_passed
        status_str = "PASS" if gate3_passed else "FAIL"
        print(f"  [QA Gate 3] Rooms: {len(geom_res.rooms)}/{len(topo_res.rooms)}, Overlap: {geom_res.max_overlap_px2:.4f} px², Ortho Dev: {geom_res.max_ortho_dev_deg:.2f}° [{status_str}]")

        # 1. Save Manhattan Room Polygons Overlay
        poly_vis = render_manhattan_polygons(rect_render, geom_res.rooms, alpha=0.55)
        cv2.imwrite(str(unit_out_dir / "10_manhattan_room_polygons.png"), poly_vis)

        # 2. Save 1D Shared Wall Wireframe
        wire_vis = render_shared_wall_wireframe(rect_render.shape, geom_res.rooms)
        cv2.imwrite(str(unit_out_dir / "11_manhattan_shared_wall_wireframe.png"), wire_vis)

        # 3. Save Manhattan Fixtures Overlay
        fix_vis = render_manhattan_fixtures_overlay(rect_render, geom_res.rooms, topo_res.furniture_layers)
        cv2.imwrite(str(unit_out_dir / "12_manhattan_fixtures_overlay.png"), fix_vis)

        results_summary.append({
            "unit": stem,
            "rooms_count": len(geom_res.rooms),
            "expected_rooms": len(topo_res.rooms),
            "max_overlap": geom_res.max_overlap_px2,
            "max_ortho_dev": geom_res.max_ortho_dev_deg,
            "passed": gate3_passed,
        })

    print("\n" + "=" * 80)
    print("PHASE 3 QA GATE 3 SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Unit':<35} | {'Rooms':<10} | {'Max Overlap':<15} | {'Max Ortho Dev':<15} | {'Gate 3'}")
    print("-" * 80)
    all_passed = True
    for r in results_summary:
        gate_res = "PASS" if r['passed'] else "FAIL"
        if not r['passed']:
            all_passed = False
        print(f"{r['unit'][:35]:<35} | {r['rooms_count']:>2}/{r['expected_rooms']:<2}     | {r['max_overlap']:>10.4f} px² | {r['max_ortho_dev']:>10.2f}°    | {gate_res}")
    print("=" * 80)
    if all_passed:
        print("ALL ASSETS PASSED QA GATE 3 (0.0000 px² Overlap, Ortho Dev <= 0.5°). Ready for Phase 4!")
    else:
        print("SOME ASSETS FAILED QA GATE 3. Review notes above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
