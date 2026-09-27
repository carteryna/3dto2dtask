#!/usr/bin/env python3
"""Visual QA for Iteration 2 Phase 4 scale estimation.

Run inside the container:

    python scripts/v2_test_scale.py

Each render is flattened, segmented, and fitted. OCR sets the scale when a
dimension callout matches a room. Otherwise the median architectural anchor
is used. ``10_scale_anchors.png`` draws door gaps and bathtubs; the counter
depth is a distance, so it is printed rather than boxed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2_reconstruction.geometry import RectilinearFitter
from src.v2_reconstruction.homography import PerspectiveCorrector
from src.v2_reconstruction.scale import ScaleEstimator, rooms_from_rects
from src.v2_reconstruction.segmentation import FloorSeamSegmenter

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v2"
COLORS = {
    "doorway": (40, 40, 220),
    "bathtub": (220, 120, 20),
    "countertop": (20, 160, 20),
}


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def draw_anchors(image: np.ndarray, anchors: list) -> np.ndarray:
    panel = image.copy()
    for anchor in anchors:
        if anchor.kind == "countertop":
            continue
        x, y, width, height = anchor.bbox
        color = COLORS.get(anchor.kind, (0, 0, 255))
        cv2.rectangle(panel, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
        label = f"{anchor.kind[0]} {anchor.pixels:.0f}px"
        cv2.putText(panel, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return panel


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    config = ROOT / "configs" / "config.yaml"
    corrector = PerspectiveCorrector(config)
    segmenter = FloorSeamSegmenter(config)
    fitter = RectilinearFitter(config)
    estimator = ScaleEstimator(config)
    tolerance = int(fitter.params["grid_snap_tolerance_px"])
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        flattened = corrector.correct(image).flattened
        floors = segmenter.segment(flattened)
        rects = fitter.snap_to_global_grid(
            fitter.split_open_plan_areas(fitter.fit_bounding_rectangles(floors), flattened),
            tolerance,
        )
        scale = estimator.estimate(flattened, rooms_from_rects(rects))
        result = estimator.result
        assert result is not None

        output_dir = DEBUG_DIR / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "10_scale_anchors.png"), draw_anchors(flattened, result.anchors))

        counts: dict[str, int] = {}
        for anchor in result.anchors:
            counts[anchor.kind] = counts.get(anchor.kind, 0) + 1
        ratios = [anchor.pixels_per_cm for anchor in result.anchors]
        anchor_median = float(np.median(ratios)) if ratios else None
        scale_text = f"{scale:.4f} px/cm" if scale is not None else "none"
        anchor_text = f"{anchor_median:.4f}" if anchor_median is not None else "none"
        ocr_text = f"{result.ocr_pixels_per_cm:.4f}" if result.ocr_pixels_per_cm is not None else "none"
        print(
            f"{image_path.name}: source={result.source} scale={scale_text} "
            f"ocr={ocr_text} px/cm callouts={result.callout_count} "
            f"anchor_median={anchor_text} px/cm anchors={counts}"
        )
        for anchor in result.anchors:
            print(
                f"  {anchor.kind:11s} {anchor.pixels:6.1f} px / {anchor.centimeters:.0f} cm "
                f"= {anchor.pixels_per_cm:.4f} px/cm"
            )
    print(f"Scale panels written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
