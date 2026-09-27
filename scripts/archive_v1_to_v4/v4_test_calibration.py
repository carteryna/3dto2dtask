#!/usr/bin/env python3
"""Visual QA for Iteration 4 Phase 1: Metric Camera Calibration & Ground Rectification.

Run inside the container:

    python scripts/v4_test_calibration.py

Tests Manhattan camera calibration on each reference 3D render in inout_data/.
Saves debug panels under data/output/debug_v4/<stem>/:
  - 01_lsd_lines.png: Detected structural horizontal and vertical line segments.
  - 02_vanishing_points.png: Lines extended toward the vanishing point with VP coordinates.
  - 03_rectified_floor.png: Rectified orthographic top-down view with residual skew metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4_cad_arrangement.calibration import ManhattanCalibrator

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v4"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def draw_lines(canvas: np.ndarray, lines: list[np.ndarray], color: tuple[int, int, int], thickness: int = 2) -> None:
    for line in lines:
        x1, y1, x2, y2 = [int(round(v)) for v in line]
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_vanishing_extensions(
    image: np.ndarray,
    v_lines: list[np.ndarray],
    vp: tuple[float, float],
    step: int = 4,
) -> np.ndarray:
    """Draw a subset of vertical lines extending upward towards the vanishing point."""
    h, w = image.shape[:2]
    canvas = image.copy()
    vx, vy = vp
    # Draw faint rays toward VP
    for i, line in enumerate(v_lines):
        if i % step != 0:
            continue
        x1, y1, x2, y2 = line
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Extend from (mx, my) towards (vx, vy) up to top edge y=0
        if abs(my - vy) > 1e-3:
            t = (0 - my) / (vy - my)
            top_x = int(round(mx + t * (vx - mx)))
            cv2.line(canvas, (int(round(mx)), int(round(my))), (top_x, 0), (255, 120, 0), 1, cv2.LINE_AA)
        cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), 2, cv2.LINE_AA)

    # Annotation badge
    badge = f"VP_v: ({vx:.1f}, {vy:.1f})"
    cv2.rectangle(canvas, (10, 10), (320, 48), (0, 0, 0), -1)
    cv2.putText(canvas, badge, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")

    calibrator = ManhattanCalibrator(ROOT / "configs" / "config.yaml")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Executing Phase 1 Camera Calibration & Rectification on {len(images)} render(s)...")

    results_table = []

    for image_path in images:
        stem = image_path.stem
        out_stem_dir = DEBUG_DIR / stem
        out_stem_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        res = calibrator.calibrate_and_rectify(image)

        # 01_lsd_lines.png: Overlay green (horizontal) and blue (vertical) lines
        p1_canvas = image.copy()
        draw_lines(p1_canvas, res.horizontal_lines, color=(0, 220, 0), thickness=2)
        draw_lines(p1_canvas, res.vertical_lines, color=(220, 50, 50), thickness=2)
        cv2.putText(
            p1_canvas,
            f"H-lines: {len(res.horizontal_lines)} (green) | V-lines: {len(res.vertical_lines)} (blue)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            p1_canvas,
            f"H-lines: {len(res.horizontal_lines)} (green) | V-lines: {len(res.vertical_lines)} (blue)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_stem_dir / "01_lsd_lines.png"), p1_canvas)

        # 02_vanishing_points.png: Lines extended toward the vanishing point
        p2_canvas = draw_vanishing_extensions(image, res.vertical_lines, res.vp_vertical)
        cv2.imwrite(str(out_stem_dir / "02_vanishing_points.png"), p2_canvas)

        # 03_rectified_floor.png: The rectified image with stats
        p3_canvas = res.rectified_image.copy()
        stat_text = (
            f"Skew before: {res.skew_before_deg:.2f} deg -> after: {res.skew_after_deg:.2f} deg "
            f"({'Orthographic' if res.is_orthographic else 'Rectified'})"
        )
        cv2.putText(p3_canvas, stat_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(p3_canvas, stat_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_stem_dir / "03_rectified_floor.png"), p3_canvas)

        results_table.append(
            {
                "stem": stem,
                "vp_v": res.vp_vertical,
                "vp_h": res.vp_horizontal,
                "skew_before": res.skew_before_deg,
                "skew_after": res.skew_after_deg,
                "is_orthographic": res.is_orthographic,
                "out_shape": res.rectified_image.shape,
            }
        )

        print(
            f"[{stem}]\n"
            f"  VP_vertical: ({res.vp_vertical[0]:.1f}, {res.vp_vertical[1]:.1f})\n"
            f"  VP_horizontal: ({res.vp_horizontal[0]:.1f}, {res.vp_horizontal[1]:.1f})\n"
            f"  Skew reduction: {res.skew_before_deg:.2f}° -> {res.skew_after_deg:.2f}°\n"
            f"  Output shape: {res.rectified_image.shape[:2]} (from {image.shape[:2]})\n"
            f"  Wrote QA panels to {out_stem_dir}"
        )

    print("\nPhase 1 Calibration & Ground Rectification complete.")


if __name__ == "__main__":
    main()
