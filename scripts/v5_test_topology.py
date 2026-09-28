"""Visual and quantitative verification script for Iteration 5 Phase 2: Topological Extraction & Door Bridging."""

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
from src.v5_semi_automated.topology import (
    SemiAutomatedTopology,
    TopologicalRoomNode,
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
    parser = argparse.ArgumentParser(description="Test Iteration 5 Phase 2 Topological Extraction & Door Bridging.")
    parser.add_argument("--input", default=None, help="Path to single floor-plan image (.webp).")
    parser.add_argument("--input_dir", default="inout_data", help="Directory of floor-plan images.")
    parser.add_argument("--output_dir", default="data/output/debug_v5", help="Directory for debug outputs.")
    parser.add_argument("--config", default=None, help="Path to configs/config.yaml.")
    return parser.parse_args()


def render_rooms_overlay(
    bg_image: np.ndarray,
    rooms: List[TopologicalRoomNode],
    alpha: float = 0.55,
) -> np.ndarray:
    """Render pastel colored fills for each room with solid black borders onto the background."""
    h, w = bg_image.shape[:2]
    overlay = bg_image.copy()
    color_layer = np.zeros_like(bg_image)

    for i, room in enumerate(rooms):
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        mask_bool = room.mask > 0
        color_layer[mask_bool] = color

    # Blend
    cv2.addWeighted(color_layer, alpha, overlay, 1.0 - alpha, 0, overlay)

    # Draw solid room perimeter contours
    for i, room in enumerate(rooms):
        if len(room.contour) > 0:
            cv2.drawContours(overlay, [room.contour], -1, (40, 40, 40), 2, cv2.LINE_AA)

        # Draw room label & area callout
        cx, cy = int(round(room.centroid[0])), int(round(room.centroid[1]))
        if room.label == f"Room {room.room_id}":
            label_text = room.label
        else:
            label_text = f"#{room.room_id} {room.label}"
        area_text = f"{int(room.area_px):,} px"

        # Text banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thick = 1
        (tw1, th1), _ = cv2.getTextSize(label_text, font, scale, thick)
        (tw2, th2), _ = cv2.getTextSize(area_text, font, scale - 0.05, thick)
        tw = max(tw1, tw2)
        th_total = th1 + th2 + 8

        x0, y0 = cx - tw // 2 - 4, cy - th_total // 2 - 2
        cv2.rectangle(overlay, (x0, y0), (x0 + tw + 8, y0 + th_total + 6), (255, 255, 255), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + tw + 8, y0 + th_total + 6), (70, 70, 70), 1)

        cv2.putText(overlay, label_text, (x0 + 4, y0 + th1 + 2), font, scale, (20, 20, 20), thick, cv2.LINE_AA)
        cv2.putText(overlay, area_text, (x0 + 4, y0 + th1 + th2 + 6), font, scale - 0.05, (100, 100, 100), thick, cv2.LINE_AA)

    return overlay


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

    print(f"=== Iteration 5 Phase 2: Topological Extraction & Door Bridging ({len(input_paths)} assets) ===")

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

        # Check if Phase 1 rectified assets exist, else compute them
        rect_render_path = unit_out_dir / "02_rectified_render.png"
        rect_cl_path = unit_out_dir / "03_rectified_center_line.png"
        rect_ns_path = unit_out_dir / "03_rectified_no_stroke.png"
        rect_st_path = unit_out_dir / "03_rectified_stroke.png"

        if rect_render_path.is_file() and (rect_cl_path.is_file() or rect_ns_path.is_file()):
            rect_render = cv2.imread(str(rect_render_path))
            rect_guides = {}
            if rect_cl_path.is_file():
                rect_guides["center_line"] = cv2.imread(str(rect_cl_path), cv2.IMREAD_UNCHANGED)
            if rect_ns_path.is_file():
                rect_guides["no_stroke"] = cv2.imread(str(rect_ns_path), cv2.IMREAD_UNCHANGED)
            if rect_st_path.is_file():
                rect_guides["stroke"] = cv2.imread(str(rect_st_path), cv2.IMREAD_UNCHANGED)
        else:
            guides = rectifier.load_guides_for_unit(path)
            rect_res = rectifier.rectify(render_img, guides=guides, unit_key=unit_key)
            rect_render = rect_res.rectified_render
            rect_guides = rect_res.rectified_guides

        # Execute Phase 2 Topology
        topo_res = topology_engine.process(rect_guides, unit_key=unit_key)

        for note in topo_res.notes:
            print(f"  [Note] {note}")

        # Gate 2 QA Verification
        # 1. Check mutual overlap between all pairs of room masks (must be 0 px)
        max_overlap_px = 0
        n_rooms = len(topo_res.rooms)
        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                inter = np.sum(cv2.bitwise_and(topo_res.rooms[i].mask, topo_res.rooms[j].mask) > 0)
                if inter > max_overlap_px:
                    max_overlap_px = inter

        # 2. Check minimum room count expectation (Heritage >= 6, Highlandlux >= 9, Limestone >= 9)
        expected_min_rooms = 6 if "heritage" in unit_key else 9
        passed_room_count = n_rooms >= expected_min_rooms
        passed_zero_overlap = max_overlap_px == 0
        gate2_passed = passed_room_count and passed_zero_overlap

        status_str = "PASS" if gate2_passed else "FAIL"
        print(f"  [QA Gate 2] Closed Rooms: {n_rooms} (min {expected_min_rooms}), Max Overlap: {max_overlap_px} px [{status_str}]")

        # 1. Save Closed Topological Wall Mask
        cv2.imwrite(str(unit_out_dir / "06_topological_wall_mask.png"), topo_res.wall_mask)

        # 2. Save Bridged Centerline Mask
        # Draw on black canvas with high visibility (electric cyan)
        vis_centerline = np.zeros_like(rect_render)
        vis_centerline[topo_res.centerline_mask > 0] = (255, 230, 0)  # Bright cyan
        cv2.imwrite(str(unit_out_dir / "07_bridged_centerline_mask.png"), vis_centerline)

        # 3. Save Topological Room Overlay with pastel fills and callouts
        rooms_vis = render_rooms_overlay(rect_render, topo_res.rooms, alpha=0.55)
        cv2.imwrite(str(unit_out_dir / "08_topological_rooms_overlay.png"), rooms_vis)

        # 4. Save Room Seeds / Centroids visualization
        seeds_vis = rect_render.copy()
        for room in topo_res.rooms:
            cx, cy = int(round(room.centroid[0])), int(round(room.centroid[1]))
            cv2.circle(seeds_vis, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(seeds_vis, (cx, cy), 8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(seeds_vis, f"R{room.room_id}", (cx + 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)
        cv2.imwrite(str(unit_out_dir / "09_room_seeds_centroids.png"), seeds_vis)

        results_summary.append({
            "unit": stem,
            "rooms_count": n_rooms,
            "expected_min": expected_min_rooms,
            "max_overlap": max_overlap_px,
            "passed": gate2_passed,
        })

    print("\n" + "=" * 80)
    print("PHASE 2 QA GATE 2 SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Unit':<35} | {'Closed Rooms':<15} | {'Max Overlap':<15} | {'Gate 2'}")
    print("-" * 80)
    all_passed = True
    for r in results_summary:
        gate_res = "PASS" if r['passed'] else "FAIL"
        if not r['passed']:
            all_passed = False
        print(f"{r['unit'][:35]:<35} | {r['rooms_count']:>2} (min {r['expected_min']})        | {r['max_overlap']:>10} px   | {gate_res}")
    print("=" * 80)
    if all_passed:
        print("ALL ASSETS PASSED QA GATE 2 (100% Closed Room Cycles, 0 Overlap). Ready for Phase 3!")
    else:
        print("SOME ASSETS FAILED QA GATE 2. Review notes above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
