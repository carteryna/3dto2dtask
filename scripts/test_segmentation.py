#!/usr/bin/env python3
"""Visual QA for Phase 3 hybrid room segmentation.

Run inside the container:

    python scripts/test_segmentation.py

Reuses the Phase 2 wall extractor on ``inout_data/`` and writes room overlays
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

from src.preprocessing import WallExtractor
from src.segmentation import RoomSegmenter

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


def colorize_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Paint each room mask with a distinct HSV hue."""
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if not masks:
        return canvas
    for index, mask in enumerate(masks):
        hue = int(round(179 * index / max(len(masks), 1)))
        color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        canvas[mask > 0] = color
    return canvas


def overlay_rooms(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Blend color-coded rooms onto the source render and draw ids."""
    color = colorize_masks(masks, image.shape[:2])
    blended = cv2.addWeighted(image, 0.45, color, 0.55, 0)
    occupied = np.any(color > 0, axis=2)
    panel = image.copy()
    panel[occupied] = blended[occupied]
    for index, mask in enumerate(masks, start=1):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1)
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
        cv2.putText(panel, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def process_image(extractor: WallExtractor, segmenter: RoomSegmenter, image_path: Path) -> None:
    denoised = extractor.load_and_denoise(str(image_path))
    wall_mask = extractor.extract_wall_mask(denoised)
    closed_mask = extractor.close_door_gaps(wall_mask)
    seeds, _distance = extractor.generate_room_seeds(closed_mask)
    filtered = segmenter.segment(denoised, seeds, closed_mask)

    output_dir = DEBUG_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "05_raw_sam_proposals.png"), colorize_masks(segmenter.raw_sam_masks, denoised.shape[:2]))
    cv2.imwrite(str(output_dir / "06_filtered_room_masks.png"), colorize_masks(filtered, denoised.shape[:2]))
    cv2.imwrite(str(output_dir / "07_color_coded_segmented_rooms.png"), overlay_rooms(denoised, filtered))

    print(
        f"{image_path.name}: seeds={len(seeds)} rooms={len(filtered)} "
        f"watershed_fallbacks={segmenter.fallback_count} -> {output_dir}"
    )


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")

    config_path = ROOT / "configs" / "config.yaml"
    extractor = WallExtractor(config_path)
    segmenter = RoomSegmenter(config_path)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        process_image(extractor, segmenter, image_path)
    print(f"Room overlays written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
