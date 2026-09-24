#!/usr/bin/env python3
"""Visual QA for Iteration 3 Phase 2 topological partitioning.

Run inside the container:

    python scripts/v3_test_topology.py

Phase 1 builds the walkable footprint. Phase 2 paints each floor mask as a
watershed marker and floods the remaining footprint. Panels are written to
``data/output/debug_v3/<image_stem>/``:

- ``05_seeds.png``: walkable footprint with one centroid per Phase 1 mask.
- ``06_room_cells.png``: watershed cells. Adjacent cells share an edge.
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
from src.v3_topology.topology import TopologyPartitioner

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v3"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def colorize(image: np.ndarray, masks: list[np.ndarray], seeds: list[tuple[int, int]]) -> np.ndarray:
    panel = image.copy()
    rng = np.random.default_rng(5)
    for index, mask in enumerate(masks, start=1):
        color = rng.integers(40, 220, size=3)
        panel[mask > 0] = (0.45 * panel[mask > 0] + 0.55 * color).astype(np.uint8)
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
        cv2.putText(
            panel,
            str(index),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for x, y in seeds:
        cv2.circle(panel, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 4, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    segmenter = FloorSegmenter(ROOT / "configs" / "config.yaml")
    partitioner = TopologyPartitioner(ROOT / "configs" / "config.yaml")
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        masks = segmenter.segment(image)
        footprint = segmenter.walkable_footprint(masks)
        room_seeds = partitioner.seeds(masks)
        cells = partitioner.partition(footprint, room_seeds, masks)
        debug = partitioner.debug
        assert debug is not None

        folder = DEBUG_DIR / image_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        seed_panel = image.copy()
        seed_panel[footprint > 0] = (0.55 * seed_panel[footprint > 0] + 0.45 * np.array([80, 180, 80])).astype(np.uint8)
        for index, (x, y) in enumerate(debug.seeds, start=1):
            cv2.circle(seed_panel, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(seed_panel, str(index), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.imwrite(str(folder / "05_seeds.png"), seed_panel)
        cv2.imwrite(str(folder / "06_room_cells.png"), colorize(image, cells, debug.seeds))

        union = np.zeros(footprint.shape, dtype=np.uint8)
        overlap = 0
        for cell in cells:
            overlap += int(np.count_nonzero((union > 0) & (cell > 0)))
            union[cell > 0] = 255
        uncovered = int(np.count_nonzero((footprint > 0) & (union == 0)))
        outside = sum(int(np.count_nonzero((cell > 0) & (footprint == 0))) for cell in cells)
        areas = [int(np.count_nonzero(cell)) for cell in cells]
        print(
            f"{image_path.name}: seeds={len(debug.seeds)} dropped={debug.dropped_seeds} "
            f"unseeded={debug.unseeded_components} cells={len(cells)} areas={areas} "
            f"overlap_px={overlap} uncovered_px={uncovered} outside_px={outside}"
        )


if __name__ == "__main__":
    main()
