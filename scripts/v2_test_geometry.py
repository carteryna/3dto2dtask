#!/usr/bin/env python3
"""Visual QA for Iteration 2 Phase 3 rectilinear room fitting.

Run inside the container:

    python scripts/v2_test_geometry.py

Each render is flattened and segmented, then fitted. Panels are written to
``data/output/debug_v2/<image_stem>/``:

- ``08_raw_rectangles.png``: orthogonal rectangles before grid snapping.
- ``09_grid_snapped_rectangles.png``: the same rooms on the shared wall grid.
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
from src.v2_reconstruction.segmentation import FloorSeamSegmenter

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v2"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def palette(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(max(count, 1)):
        hue = int(180 * index / max(count, 1))
        hsv = np.uint8([[[hue, 200, 230]]])
        colors.append(tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))
    return colors


def draw_rects(image: np.ndarray, rects: list[dict], alpha: float = 0.35) -> np.ndarray:
    panel = image.copy()
    colors = palette(len(rects))
    for rect, color in zip(rects, colors):
        x, y, width, height = int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])
        overlay = panel.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, thickness=-1)
        panel = cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0)
        cv2.rectangle(panel, (x, y), (x + width, y + height), color, thickness=2, lineType=cv2.LINE_AA)
    return panel


def shared_edges(rects: list[dict], tolerance: int) -> int:
    """Count pairs of rectangles that share a horizontal or vertical boundary."""
    count = 0
    for i, first in enumerate(rects):
        for second in rects[i + 1 :]:
            ax1, ay1 = int(first["x"]), int(first["y"])
            ax2, ay2 = ax1 + int(first["w"]), ay1 + int(first["h"])
            bx1, by1 = int(second["x"]), int(second["y"])
            bx2, by2 = bx1 + int(second["w"]), by1 + int(second["h"])
            vertical = abs(ax2 - bx1) <= tolerance or abs(bx2 - ax1) <= tolerance
            horizontal = abs(ay2 - by1) <= tolerance or abs(by2 - ay1) <= tolerance
            x_overlap = min(ax2, bx2) - max(ax1, bx1) > tolerance
            y_overlap = min(ay2, by2) - max(ay1, by1) > tolerance
            if (vertical and y_overlap) or (horizontal and x_overlap):
                count += 1
    return count


def overlap_px(rects: list[dict], same_source: bool) -> int:
    total = 0
    for i, first in enumerate(rects):
        for second in rects[i + 1 :]:
            shared = int(first["source_index"]) == int(second["source_index"])
            if shared != same_source:
                continue
            x1 = max(int(first["x"]), int(second["x"]))
            y1 = max(int(first["y"]), int(second["y"]))
            x2 = min(int(first["x"] + first["w"]), int(second["x"] + second["w"]))
            y2 = min(int(first["y"] + first["h"]), int(second["y"] + second["h"]))
            if x2 > x1 and y2 > y1:
                total += (x2 - x1) * (y2 - y1)
    return total


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    config = ROOT / "configs" / "config.yaml"
    corrector = PerspectiveCorrector(config)
    segmenter = FloorSeamSegmenter(config)
    fitter = RectilinearFitter(config)
    tolerance = int(fitter.params["grid_snap_tolerance_px"])
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        flattened = corrector.correct(image).flattened
        floors = segmenter.segment(flattened)
        raw = fitter.fit_bounding_rectangles(floors)
        split = fitter.split_open_plan_areas(raw, flattened)
        snapped = fitter.snap_to_global_grid(split, tolerance)

        output_dir = DEBUG_DIR / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "08_raw_rectangles.png"), draw_rects(flattened, split))
        cv2.imwrite(str(output_dir / "09_grid_snapped_rectangles.png"), draw_rects(flattened, snapped))

        print(
            f"{image_path.name}: masks={len(floors)} raw={len(raw)} "
            f"after_split={len(split)} snapped={len(snapped)} "
            f"shared_edges={shared_edges(snapped, 1)} "
            f"cross_overlap_px={overlap_px(snapped, False)} "
            f"l_overlap_px={overlap_px(snapped, True)} "
            f"-> {output_dir}"
        )
    print(f"Geometry panels written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
