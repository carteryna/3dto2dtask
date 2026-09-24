#!/usr/bin/env python3
"""Visual QA for Iteration 3 Phase 1 floor segmentation.

Run inside the container:

    python scripts/v3_test_segmentation.py

The original render is not warped. Panels are written to
``data/output/debug_v3/<image_stem>/``:

- ``01_wall_tops.png``: thin shell-coloured ribbons rejected as wall tops.
- ``02_floor_masks.png``: kept floor regions and their centroids.
- ``03_walkable_footprint.png``: the union of those regions, with furniture
  holes filled where they do not cross a wall top.
- ``04_unclaimed.png``: interior pixels that were not a FastSAM floor, wall
  top, or furniture mask. Beige carpet is recovered from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v3_topology.segmentation import FloorSegmenter

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v3"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def colorize(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    panel = image.copy()
    rng = np.random.default_rng(3)
    for index, mask in enumerate(masks, start=1):
        color = rng.integers(40, 220, size=3)
        panel[mask > 0] = (0.45 * panel[mask > 0] + 0.55 * color).astype(np.uint8)
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
        cv2.circle(panel, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(panel, str(index), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    return panel


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    segmenter = FloorSegmenter(ROOT / "configs" / "config.yaml")
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        masks = segmenter.segment(image)
        footprint = segmenter.walkable_footprint(masks)
        debug = segmenter.debug
        assert debug is not None

        folder = DEBUG_DIR / image_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        walls = image.copy()
        walls[debug.wall_tops > 0] = (40, 40, 220)
        cv2.imwrite(str(folder / "01_wall_tops.png"), walls)
        cv2.imwrite(str(folder / "02_floor_masks.png"), colorize(image, masks))
        walkable = image.copy()
        walkable[footprint > 0] = (0.45 * walkable[footprint > 0] + 0.55 * np.array([40, 170, 60])).astype(np.uint8)
        cv2.imwrite(str(folder / "03_walkable_footprint.png"), walkable)
        missed = image.copy()
        missed[debug.unclaimed > 0] = (0.45 * missed[debug.unclaimed > 0] + 0.55 * np.array([40, 140, 220])).astype(np.uint8)
        cv2.imwrite(str(folder / "04_unclaimed.png"), missed)

        interior = int(np.count_nonzero(debug.footprint))
        covered = int(np.count_nonzero((footprint > 0) & (debug.footprint > 0)))
        leaked = int(np.count_nonzero((footprint > 0) & (debug.wall_tops > 0)))
        areas = [int(np.count_nonzero(mask)) for mask in masks]
        overlap = 0
        for index, mask in enumerate(masks):
            for other in masks[index + 1 :]:
                overlap += int(np.count_nonzero((mask > 0) & (other > 0)))
        print(
            f"{image_path.name}: proposals={debug.proposal_count} "
            f"objects={debug.object_count} containers={debug.container_count} "
            f"recovered={debug.recovered_count} floors={len(masks)} "
            f"areas={areas} coverage={covered / interior:.3f} wall_leak_px={leaked} overlap_px={overlap}"
        )
        for note in debug.notes:
            print(f"  {note}")


if __name__ == "__main__":
    main()
