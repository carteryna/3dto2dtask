#!/usr/bin/env python3
"""Visual QA for Phase 2 wall extraction and door-gap closing.

Run inside the container:

    python scripts/test_preprocessing.py

Reads renders from ``inout_data/`` and writes a four-panel debug set per image
under ``data/output/debug/<image_stem>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v1_baseline.preprocessing import WallExtractor

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def render_seed_panel(distance: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Colorize the floor distance map and mark room centroids."""
    normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
    panel = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
    for center_x, center_y in seeds:
        point = (int(round(float(center_x))), int(round(float(center_y))))
        cv2.circle(panel, point, 7, (0, 255, 0), thickness=-1)
        cv2.circle(panel, point, 9, (255, 255, 255), thickness=1)
    return panel


def process_image(extractor: WallExtractor, image_path: Path) -> None:
    denoised = extractor.load_and_denoise(str(image_path))
    wall_mask = extractor.extract_wall_mask(denoised)
    closed_mask = extractor.close_door_gaps(wall_mask)
    seeds, distance = extractor.generate_room_seeds(closed_mask)

    output_dir = DEBUG_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "01_denoised.png"), denoised)
    cv2.imwrite(str(output_dir / "02_raw_wall_mask.png"), wall_mask)
    cv2.imwrite(str(output_dir / "03_closed_door_wall_mask.png"), closed_mask)
    cv2.imwrite(str(output_dir / "04_distance_transform_seeds.png"), render_seed_panel(distance, seeds))

    print(
        f"{image_path.name}: seeds={len(seeds)} "
        f"wall_px={int(np.count_nonzero(wall_mask))} "
        f"closed_px={int(np.count_nonzero(closed_mask))} "
        f"-> {output_dir}"
    )


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")

    extractor = WallExtractor(ROOT / "configs" / "config.yaml")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        process_image(extractor, image_path)
    print(f"Debug masks written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
