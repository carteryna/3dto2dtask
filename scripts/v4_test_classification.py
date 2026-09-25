"""Visual verification script for Iteration 4 Phase 4: Spatial Containment & Semantic Classification."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v4_cad_arrangement.arrangement import PlanarWallArranger
from src.v4_cad_arrangement.calibration import ManhattanCalibrator
from src.v4_cad_arrangement.classification import RoomClassifier
from src.v4_cad_arrangement.objects import ObjectDetector

ROOM_COLORS = {
    "BEDROOM": (220, 190, 240),        # Soft Lavender / Purple
    "BATHROOM": (240, 220, 160),       # Soft Sky Blue
    "LIVING / DINING": (180, 230, 180), # Soft Mint Green
    "KITCHEN": (170, 220, 240),        # Soft Gold / Amber
    "BALCONY": (200, 230, 200),        # Soft Sage Green
    "WALK-IN CLOSET": (240, 200, 210),  # Soft Rose
    "CLOSET / STORAGE": (210, 210, 210),# Light Grey
}


def draw_centered_text(
    img: np.ndarray,
    lines: list[str],
    center: tuple[int, int],
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (20, 20, 20),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    line_spacing = 18

    # Measure text bounding box
    max_w = 0
    total_h = len(lines) * line_spacing
    sizes = []
    for line in lines:
        (tw, th), base = cv2.getTextSize(line, font, scale, thick)
        sizes.append((tw, th))
        if tw > max_w:
            max_w = tw

    cx, cy = center
    x0 = cx - max_w // 2 - 6
    y0 = cy - total_h // 2 - 4
    x1 = cx + max_w // 2 + 6
    y1 = cy + total_h // 2 + 6

    # Draw rounded-ish background badge
    cv2.rectangle(img, (x0, y0), (x1, y1), bg_color, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), 1)

    cur_y = cy - (total_h // 2) + 14
    for i, line in enumerate(lines):
        tw, th = sizes[i]
        tx = cx - tw // 2
        cv2.putText(img, line, (tx, cur_y), font, scale, text_color, thick, cv2.LINE_AA)
        cur_y += line_spacing


def test_classification():
    data_dir = PROJECT_ROOT / "inout_data"
    output_dir = PROJECT_ROOT / "data" / "output" / "debug_v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(data_dir.glob("*.webp"))
    if not input_paths:
        print("No .webp files found in inout_data/")
        return

    calibrator = ManhattanCalibrator()
    detector = ObjectDetector()
    arranger = PlanarWallArranger()
    classifier = RoomClassifier()

    print(f"=== Testing Iteration 4 Phase 4: Spatial Containment & Semantic Classification ({len(input_paths)} images) ===")

    for path in input_paths:
        stem = path.stem
        out_stem_dir = output_dir / stem
        out_stem_dir.mkdir(parents=True, exist_ok=True)

        rectified_path = out_stem_dir / "03_rectified_floor.png"
        if rectified_path.is_file():
            rectified_img = cv2.imread(str(rectified_path))
        else:
            orig_img = cv2.imread(str(path))
            calib_res = calibrator.calibrate(orig_img)
            rectified_img = calib_res.rectified_image
            cv2.imwrite(str(rectified_path), rectified_img)

        # Run Phase 2 Object Detection
        obj_res = detector.detect(rectified_img)

        # Run Phase 3 Planar Wall Arrangement
        arr_res = arranger.arrange(rectified_img, object_result=obj_res)

        # Run Phase 4 Semantic Room Classification
        class_res = classifier.classify(
            faces=arr_res.faces,
            object_result=obj_res,
            filename=stem,
            image_shape=rectified_img.shape[:2],
        )

        print(f"\n--- {stem} ---")
        for note in class_res.notes:
            print(f"  [Note] {note}")

        print(f"  Scale: {class_res.scale_px_per_ft:.2f} px/ft | Total sq ft: {class_res.total_sqft:.1f}")
        print(f"  Architectural Rooms ({len(class_res.rooms)}):")
        for idx, room in enumerate(class_res.rooms):
            print(
                f"    [{idx:02d}] {room.category:<16} | "
                f"{room.sqft_area:>5.1f} sq ft ({room.area_fraction*100:>4.1f}%) | "
                f"{room.dimension_str:<15} | "
                f"Objs: {[o.category for o in room.contained_objects]}"
            )

        # -------------------------------------------------------------
        # Render 08_semantic_rooms.png
        # -------------------------------------------------------------
        vis_rooms = rectified_img.copy()
        room_overlay = np.zeros_like(vis_rooms)

        for room in class_res.rooms:
            color = ROOM_COLORS.get(room.category, (200, 200, 200))
            coords = np.array(room.polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(room_overlay, [coords], color)
            cv2.polylines(vis_rooms, [coords], isClosed=True, color=(40, 40, 40), thickness=2, lineType=cv2.LINE_AA)

        vis_rooms = cv2.addWeighted(vis_rooms, 0.55, room_overlay, 0.45, 0)

        # Draw labels
        for room in class_res.rooms:
            rx, ry = int(round(room.label_anchor[0])), int(round(room.label_anchor[1]))
            badge_color = ROOM_COLORS.get(room.category, (240, 240, 240))
            lines = [
                room.category,
                room.dimension_str,
                f"{room.sqft_area:.1f} sq ft ({room.area_fraction*100:.1f}%)",
            ]
            draw_centered_text(vis_rooms, lines, (rx, ry), badge_color)

        p8 = out_stem_dir / "08_semantic_rooms.png"
        cv2.imwrite(str(p8), vis_rooms)
        print(f"  Saved {p8.name}")


if __name__ == "__main__":
    test_classification()
