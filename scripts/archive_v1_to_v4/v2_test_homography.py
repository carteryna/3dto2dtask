#!/usr/bin/env python3
"""Visual QA for Iteration 2 Phase 1 perspective correction.

Run inside the container:

    python scripts/v2_test_homography.py

Writes per-image panels under ``data/output/debug_v2/<image_stem>/`` and
prints two skew measures before and after warping: the length-weighted lean
of every footprint outline segment from the nearest axis, and the lean of the
four detected exterior-wall sides.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2_reconstruction.homography import PerspectiveCorrector

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v2"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def side_skew_deg(quad: np.ndarray) -> float:
    """Mean deviation of the quad's top/bottom from horizontal and left/right from vertical."""
    deviations = []
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        angle = abs(math.degrees(math.atan2(float(end[1] - start[1]), float(end[0] - start[0])))) % 180.0
        if index % 2 == 0:
            deviations.append(min(angle, 180.0 - angle))
        else:
            deviations.append(abs(angle - 90.0))
    return float(sum(deviations) / len(deviations))


def draw_quad(image: np.ndarray, quad: np.ndarray, contour: np.ndarray | None) -> np.ndarray:
    panel = image.copy()
    if contour is not None:
        cv2.drawContours(panel, [contour], -1, (40, 160, 240), 1, cv2.LINE_AA)
    points = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(panel, [points], True, (40, 40, 220), 2, cv2.LINE_AA)
    for label, corner in zip(("TL", "TR", "BR", "BL"), quad):
        center = (int(round(corner[0])), int(round(corner[1])))
        cv2.circle(panel, center, 6, (40, 40, 220), -1, cv2.LINE_AA)
        cv2.putText(panel, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 220), 2)
    return panel


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    corrector = PerspectiveCorrector(ROOT / "configs" / "config.yaml")
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        result = corrector.correct(image)

        after_contour = corrector.detect_footprint(result.flattened)
        after_quad = corrector.detect_boundary_quad(result.flattened, after_contour)
        before = corrector.outline_skew_deg(result.footprint_contour)
        after = corrector.outline_skew_deg(after_contour)
        quad_before = side_skew_deg(result.source_quad)
        quad_after = side_skew_deg(after_quad)

        output_dir = DEBUG_DIR / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "01_boundary_quad.png"), draw_quad(image, result.source_quad, result.footprint_contour))
        cv2.imwrite(str(output_dir / "02_flattened.png"), result.flattened)
        cv2.imwrite(str(output_dir / "03_flattened_check.png"), draw_quad(result.flattened, after_quad, after_contour))

        height, width = result.flattened.shape[:2]
        print(
            f"{image_path.name}: applied={result.applied} "
            f"outline_skew {before:.2f} -> {after:.2f} deg, "
            f"wall_quad_skew {quad_before:.2f} -> {quad_after:.2f} deg, "
            f"flattened={width}x{height} -> {output_dir}"
        )
    print(f"Homography panels written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
