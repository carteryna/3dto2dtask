"""Visual and quantitative verification script for Iteration 5 Phase 4: Conformal Metric Scale & Dimensional Engine."""

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
from src.v5_semi_automated.geometry import SemiAutomatedGeometry
from src.v5_semi_automated.scale import (
    DimensionedRoom,
    ScaleResult,
    SemiAutomatedScale,
)
from src.v5_semi_automated.visualizer import SemiAutomatedVisualizer

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
    parser = argparse.ArgumentParser(description="Test Iteration 5 Phase 4 Conformal Metric Scale Engine.")
    parser.add_argument("--input", default=None, help="Path to single floor-plan image (.webp).")
    parser.add_argument("--input_dir", default="inout_data", help="Directory of floor-plan images.")
    parser.add_argument("--output_dir", default="data/output/debug_v5", help="Directory for debug outputs.")
    parser.add_argument("--config", default=None, help="Path to configs/config.yaml.")
    return parser.parse_args()


def render_dimensioned_overlay(
    bg_image: np.ndarray,
    rooms: List[DimensionedRoom],
    alpha: float = 0.50,
) -> np.ndarray:
    """Render pastel colored fills with architectural dimension badges onto the background."""
    out = bg_image.copy()
    h, w = bg_image.shape[:2]
    color_layer = np.zeros_like(bg_image)

    for i, rm in enumerate(rooms):
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        coords = np.array(rm.polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(color_layer, [coords], color)

    mask_any = np.any(color_layer > 0, axis=-1)
    out[mask_any] = cv2.addWeighted(out, 1.0 - alpha, color_layer, alpha, 0)[mask_any]

    # Draw solid black boundary lines and architectural badges
    for i, rm in enumerate(rooms):
        coords = np.array(rm.polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(out, [coords], isClosed=True, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)

        # Badge position (visual center via polylabel)
        ax, ay = int(round(rm.label_anchor[0])), int(round(rm.label_anchor[1]))

        line1 = f"Room {rm.room_id}"
        if rm.fixtures:
            line1 += f" ({','.join(rm.fixtures)})"
        elif rm.is_outdoor:
            line1 += " (Balcony)"

        line2 = f"{rm.area_sqft:.1f} sq ft ({rm.area_m2:.1f} m2)"
        line3 = rm.dimension_str

        # Render 3-line CAD callout badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        thick = 1

        (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
        (w2, h2), _ = cv2.getTextSize(line2, font, scale - 0.04, thick)
        (w3, h3), _ = cv2.getTextSize(line3, font, scale, thick)

        badge_w = max(w1, w2, w3) + 14
        badge_h = h1 + h2 + h3 + 18
        bx1 = max(0, ax - badge_w // 2)
        by1 = max(0, ay - badge_h // 2)
        bx2 = min(w - 1, bx1 + badge_w)
        by2 = min(h - 1, by1 + badge_h)

        cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (60, 60, 60), 1, cv2.LINE_AA)

        ty1 = by1 + h1 + 4
        ty2 = ty1 + h2 + 4
        ty3 = ty2 + h3 + 4

        cv2.putText(out, line1, (bx1 + (badge_w - w1) // 2, ty1), font, scale, (10, 10, 10), thick, cv2.LINE_AA)
        cv2.putText(out, line2, (bx1 + (badge_w - w2) // 2, ty2), font, scale - 0.04, (80, 80, 80), thick, cv2.LINE_AA)
        cv2.putText(out, line3, (bx1 + (badge_w - w3) // 2, ty3), font, scale, (180, 50, 0), thick, cv2.LINE_AA)

    return out


def render_fixture_verification(
    bg_image: np.ndarray,
    scale_result: ScaleResult,
    furniture_layers: dict[str, np.ndarray],
) -> np.ndarray:
    """Render fixture overlays displaying measured vs standard architectural dimensions."""
    out = bg_image.copy()
    h, w = bg_image.shape[:2]

    # Color overlay for fixtures
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
    out[mask_any] = cv2.addWeighted(out, 0.4, color_layer, 0.6, 0)[mask_any]

    # Plot measurement callouts for each anchor
    s_cm_per_px = scale_result.scale_cm_per_px
    y_text = 30
    cv2.putText(
        out,
        f"Master Scale: {scale_result.scale_px_per_ft:.2f} px/ft ({s_cm_per_px:.2f} cm/px) [{scale_result.scale_source}]",
        (20, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Master Scale: {scale_result.scale_px_per_ft:.2f} px/ft ({s_cm_per_px:.2f} cm/px) [{scale_result.scale_source}]",
        (20, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 50, 200),
        1,
        cv2.LINE_AA,
    )

    y_text += 25
    for anchor in scale_result.anchors:
        txt = f"[{anchor.fixture_type.upper()}] Scale: {anchor.scale_px_per_ft:.2f} px/ft (Error vs Master: {anchor.error_vs_chosen_scale_pct:.1f}%)"
        cv2.putText(out, txt, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        y_text += 20

    return out


def render_dimensioned_blueprint(
    render_shape: Tuple[int, int],
    rooms: List[DimensionedRoom],
    total_sqft: float,
    scale_px_per_ft: float,
) -> np.ndarray:
    """Render clean, architectural 2D blueprint wireframe with room dimensions."""
    h, w = render_shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 250  # Architectural parchment white

    # Draw drafting grid lines
    for x in range(0, w, 40):
        cv2.line(canvas, (x, 0), (x, h), (238, 238, 238), 1)
    for y in range(0, h, 40):
        cv2.line(canvas, (0, y), (w, y), (238, 238, 238), 1)

    # Room fills and walls
    for i, rm in enumerate(rooms):
        coords = np.array(rm.polygon.exterior.coords, dtype=np.int32)
        # Soft architectural fill
        fill_col = (245, 240, 230) if not rm.is_outdoor else (235, 245, 250)
        cv2.fillPoly(canvas, [coords], fill_col)
        # Bold dual-line wall boundary
        cv2.polylines(canvas, [coords], isClosed=True, color=(40, 40, 40), thickness=3, lineType=cv2.LINE_AA)

        # Plot room dimension annotations
        ax, ay = int(round(rm.label_anchor[0])), int(round(rm.label_anchor[1]))
        tag1 = f"Room {rm.room_id}"
        if rm.fixtures:
            tag1 += f" ({','.join(rm.fixtures)})"
        elif rm.is_outdoor:
            tag1 += " (Balcony)"

        tag2 = f"{rm.area_sqft:.1f} SQ FT"
        tag3 = rm.dimension_str

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.36
        thick = 1

        (w1, h1), _ = cv2.getTextSize(tag1, font, scale, thick)
        (w2, h2), _ = cv2.getTextSize(tag2, font, scale - 0.05, thick)
        (w3, h3), _ = cv2.getTextSize(tag3, font, scale, thick)

        bw = max(w1, w2, w3) + 12
        bh = h1 + h2 + h3 + 14
        bx = max(0, ax - bw // 2)
        by = max(0, ay - bh // 2)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (255, 255, 255), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (150, 150, 150), 1, cv2.LINE_AA)

        cv2.putText(canvas, tag1, (bx + (bw - w1) // 2, by + h1 + 3), font, scale, (30, 30, 30), thick, cv2.LINE_AA)
        cv2.putText(canvas, tag2, (bx + (bw - w2) // 2, by + h1 + h2 + 6), font, scale - 0.05, (100, 100, 100), thick, cv2.LINE_AA)
        cv2.putText(canvas, tag3, (bx + (bw - w3) // 2, by + h1 + h2 + h3 + 9), font, scale, (150, 30, 0), thick, cv2.LINE_AA)

    # Title block in bottom-right corner
    tb_w, tb_h = 240, 60
    tb_x, tb_y = w - tb_w - 20, h - tb_h - 20
    cv2.rectangle(canvas, (tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h), (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"TOTAL AREA: {total_sqft:.0f} SQ. FT.", (tb_x + 12, tb_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"SCALE: 1' = {scale_px_per_ft:.1f}px (CONFORMAL)", (tb_x + 12, tb_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    return canvas


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
    scale_engine = SemiAutomatedScale(args.config)
    visualizer = SemiAutomatedVisualizer(args.config)

    print(f"=== Iteration 5 Phase 4: Conformal Metric Scale Engine ({len(input_paths)} assets) ===")

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

        # 3. Phase 3 Geometry
        geom_res = geometry_engine.process(topo_res, render_shape=rect_render.shape[:2])

        # 4. Phase 4 Metric Scale
        scale_res = scale_engine.process(geom_res, topology_result=topo_res, filename_or_key=stem)

        for note in scale_res.notes:
            print(f"  [Note] {note}")

        # Gate 4 QA Verification
        gate4_passed = scale_res.is_gate4_passed
        status_str = "PASS" if gate4_passed else "FAIL"
        print(
            f"  [QA Gate 4] Target: {scale_res.target_sqft:.0f} sqft, "
            f"Computed: {scale_res.computed_total_sqft:.1f} sqft, "
            f"Error: {scale_res.total_area_error_pct:.2f}% [{status_str}]"
        )

        for rm in scale_res.rooms:
            tag = f"Room {rm.room_id}"
            if rm.fixtures:
                tag += f" ({','.join(rm.fixtures)})"
            elif rm.is_outdoor:
                tag += " (Balcony)"
            print(f"    {tag:<28} | {rm.area_sqft:>6.1f} sq ft ({rm.area_m2:>5.1f} m2) | {rm.dimension_str:<15} ({rm.dimension_metric_str})")

        # 1. Save Dimensioned CAD Overlay
        dim_vis = render_dimensioned_overlay(rect_render, scale_res.rooms, alpha=0.50)
        cv2.imwrite(str(unit_out_dir / "13_dimensioned_cad_overlay.png"), dim_vis)

        # 2. Save Fixture Dimension Verification
        fix_vis = render_fixture_verification(rect_render, scale_res, topo_res.furniture_layers)
        cv2.imwrite(str(unit_out_dir / "14_fixture_dimension_verification.png"), fix_vis)

        # 3. Save Architectural 2D Blueprint Wireframe (Solid Continuous Walls, Zero Gaps)
        bp_res = visualizer.render(
            scale_result=scale_res,
            geometry_result=geom_res,
            topology_result=topo_res,
            render_shape=rect_render.shape[:2],
            unit_label=stem,
        )
        cv2.imwrite(str(unit_out_dir / "15_dimensioned_blueprint_wireframe.png"), bp_res.blueprint_image)

        results_summary.append({
            "unit": stem,
            "target_sqft": scale_res.target_sqft,
            "computed_sqft": scale_res.computed_total_sqft,
            "error_pct": scale_res.total_area_error_pct,
            "scale_px_per_ft": scale_res.scale_px_per_ft,
            "scale_cm_per_px": scale_res.scale_cm_per_px,
            "rooms_count": len(scale_res.rooms),
            "passed": gate4_passed,
        })

    print("\n" + "=" * 85)
    print("PHASE 4 QA GATE 4 SUMMARY TABLE")
    print("=" * 85)
    print(f"{'Unit':<32} | {'Target':<10} | {'Computed':<12} | {'Error':<10} | {'Scale (px/ft)':<14} | {'Gate 4'}")
    print("-" * 85)
    all_passed = True
    for r in results_summary:
        gate_res = "PASS" if r['passed'] else "FAIL"
        if not r['passed']:
            all_passed = False
        t_str = f"{r['target_sqft']:.0f} sqft" if r['target_sqft'] else "N/A"
        c_str = f"{r['computed_sqft']:.1f} sqft"
        e_str = f"{r['error_pct']:.2f}%"
        s_str = f"{r['scale_px_per_ft']:.2f} px/ft"
        print(f"{r['unit'][:32]:<32} | {t_str:<10} | {c_str:<12} | {e_str:<10} | {s_str:<14} | {gate_res}")
    print("=" * 85)
    if all_passed:
        print("ALL ASSETS PASSED QA GATE 4 (Area Error <= 3.0%, Fixture Consistency Validated). Ready for Phase 5!")
    else:
        print("SOME ASSETS FAILED QA GATE 4. Review notes above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
