#!/usr/bin/env python3
"""Visual QA for Iteration 2 Phase 2 floor-seam segmentation.

Run inside the container:

    python scripts/v2_test_segmentation.py

Each render is flattened with Phase 1, then segmented. Panels are written to
``data/output/debug_v2/<image_stem>/``:

- ``04_wall_ridges.png``: top-hat wall tops (red) and the closed barrier (grey).
- ``05_room_cells.png``: wall-top cells before seam refinement.
- ``06_floor_sources.png``: floor outlines (green from FastSAM, orange from the
  wall-top cell fallback); cell pixels trimmed away as wall face or fixture are magenta.
- ``07_floor_masks.png``: final floor masks with their outlines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def overlay_masks(image: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    panel = image.copy()
    colors = palette(len(masks))
    for mask, color in zip(masks, colors):
        tint = np.zeros_like(panel)
        tint[:] = color
        region = mask > 0
        panel[region] = (panel[region] * (1 - alpha) + tint[region] * alpha).astype(np.uint8)
    for mask, color in zip(masks, colors):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)
    return panel


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    corrector = PerspectiveCorrector(ROOT / "configs" / "config.yaml")
    segmenter = FloorSeamSegmenter(ROOT / "configs" / "config.yaml")
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        flattened = corrector.correct(image).flattened
        floors = segmenter.segment(flattened)
        debug = segmenter.debug
        assert debug is not None

        output_dir = DEBUG_DIR / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        ridges_panel = flattened.copy()
        ridges_panel[debug.barrier > 0] = (ridges_panel[debug.barrier > 0] * 0.3 + 90).astype(np.uint8)
        ridges_panel[debug.ridges > 0] = (40, 40, 220)
        cv2.imwrite(str(output_dir / "04_wall_ridges.png"), ridges_panel)
        cv2.imwrite(str(output_dir / "05_room_cells.png"), overlay_masks(flattened, debug.cells))

        sources_panel = flattened.copy()
        for cell, floor, source in zip(debug.cells, floors, debug.sources):
            trimmed = (cell > 0) & (floor == 0)
            sources_panel[trimmed] = (sources_panel[trimmed] * 0.4 + np.array([200, 40, 200]) * 0.6).astype(np.uint8)
            color = (60, 180, 60) if source == "sam" else (0, 140, 255)
            contours, _ = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(sources_panel, contours, -1, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / "06_floor_sources.png"), sources_panel)
        cv2.imwrite(str(output_dir / "07_floor_masks.png"), overlay_masks(flattened, floors))

        cell_area = sum(int(np.count_nonzero(cell)) for cell in debug.cells)
        floor_area = sum(int(np.count_nonzero(floor)) for floor in floors)
        before = segmenter.seam_alignment(debug.cells, debug.seam_edges)
        after = segmenter.seam_alignment(floors, debug.seam_edges)
        trimmed_pct = 100.0 * (cell_area - floor_area) / cell_area if cell_area else 0.0
        sam_rooms = debug.sources.count("sam")
        print(
            f"{image_path.name}: rooms={len(floors)} (sam={sam_rooms}, cell={len(floors) - sam_rooms}) "
            f"proposals={debug.proposal_count} trimmed={trimmed_pct:.1f}% "
            f"seam_alignment {before:.3f} -> {after:.3f}"
        )
        for note in debug.notes:
            print(f"  note: {note}")
    print(f"Segmentation panels written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
