"""Visual verification script for Iteration 4 Phase 2: Object Decoupling & Fixture Anchors."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v4_cad_arrangement.calibration import ManhattanCalibrator
from src.v4_cad_arrangement.objects import ObjectDetector


def draw_label(img: np.ndarray, text: str, pos: tuple[int, int], bg_color: tuple[int, int, int], text_color: tuple[int, int, int] = (255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, text_color, thick, cv2.LINE_AA)


def test_objects():
    data_dir = PROJECT_ROOT / "inout_data"
    output_dir = PROJECT_ROOT / "data" / "output" / "debug_v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(data_dir.glob("*.webp"))
    if not input_paths:
        print("No .webp files found in inout_data/")
        return

    calibrator = ManhattanCalibrator()
    detector = ObjectDetector()

    print(f"=== Testing Iteration 4 Phase 2: Object Decoupling & Fixture Anchors ({len(input_paths)} images) ===")

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

        print(f"\n--- {stem} ---")
        for note in obj_res.notes:
            print(f"  [Note] {note}")
        print(f"  Detected {len(obj_res.objects)} objects:")
        for obj in obj_res.objects:
            b = [round(float(v), 1) for v in obj.bbox]
            print(f"    - {obj.category.upper()}: bbox={b}, conf={obj.confidence:.2f}")
        print(f"  Bathroom seeds: {obj_res.bathroom_seeds}")
        print(f"  Bedroom seeds:  {obj_res.bedroom_seeds}")
        print(f"  Living seeds:   {obj_res.living_seeds}")

        # Render 04_detected_furniture.png
        vis_furn = rectified_img.copy()
        furn_overlay = np.zeros_like(vis_furn)

        cat_colors = {
            "bed": (180, 80, 200),        # Purple
            "sofa": (50, 180, 50),        # Green
            "dining_table": (30, 150, 240), # Orange
            "nightstand": (140, 140, 240), # Light Orange
            "chair": (180, 180, 80),      # Cyan-ish
            "rug": (0, 220, 220),         # Yellow
        }

        has_furn = False
        for obj in obj_res.objects:
            if obj.category not in cat_colors:
                continue
            has_furn = True
            col = cat_colors[obj.category]
            b = [int(round(v)) for v in obj.bbox]
            cv2.rectangle(vis_furn, (b[0], b[1]), (b[2], b[3]), col, 2)
            cv2.rectangle(furn_overlay, (b[0], b[1]), (b[2], b[3]), col, -1)
            draw_label(vis_furn, f"{obj.category.upper()} ({obj.confidence:.2f})", (b[0], b[1] - 4), col)

        if has_furn:
            vis_furn = cv2.addWeighted(vis_furn, 0.75, furn_overlay, 0.25, 0)
        p4 = out_stem_dir / "04_detected_furniture.png"
        cv2.imwrite(str(p4), vis_furn)
        print(f"  Saved {p4.name}")

        # Render 05_fixture_anchors.png
        vis_fix = rectified_img.copy()
        fix_overlay = np.zeros_like(vis_fix)

        fix_colors = {
            "bathtub": (220, 100, 20),   # Deep Blue
            "toilet": (0, 0, 220),        # Red
            "fixture": (128, 128, 0),     # Teal
        }

        has_fix = False
        for obj in obj_res.objects:
            if obj.category not in fix_colors:
                continue
            has_fix = True
            col = fix_colors[obj.category]
            b = [int(round(v)) for v in obj.bbox]
            cv2.rectangle(vis_fix, (b[0], b[1]), (b[2], b[3]), col, 2)
            cv2.rectangle(fix_overlay, (b[0], b[1]), (b[2], b[3]), col, -1)
            draw_label(vis_fix, f"{obj.category.upper()} ({obj.confidence:.2f})", (b[0], b[1] - 4), col)

        if has_fix:
            vis_fix = cv2.addWeighted(vis_fix, 0.70, fix_overlay, 0.30, 0)

        # Draw bathroom seeds
        for sx, sy in obj_res.bathroom_seeds:
            csx, csy = int(round(sx)), int(round(sy))
            cv2.circle(vis_fix, (csx, csy), 6, (0, 0, 255), -1)
            cv2.circle(vis_fix, (csx, csy), 10, (255, 255, 255), 2)
            draw_label(vis_fix, f"BATH SEED ({csx},{csy})", (csx + 12, csy), (0, 0, 200))

        p5 = out_stem_dir / "05_fixture_anchors.png"
        cv2.imwrite(str(p5), vis_fix)
        print(f"  Saved {p5.name}")


if __name__ == "__main__":
    test_objects()
