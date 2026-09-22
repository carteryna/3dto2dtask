#!/usr/bin/env python3
"""Visual QA for Phase 5 OCR scale parsing.

Run inside the container:

    python scripts/test_ocr.py

Rebuilds Phase 4 polygons for every render in ``inout_data/`` and writes OCR
debug panels under ``data/output/debug/<image_stem>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry import GeometryRegularizer
from src.ocr import OCRScaleParser
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


def room_color(index: int, count: int) -> tuple[int, int, int]:
    hue = int(round(179 * index / max(count, 1)))
    color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.45) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_detections(image: np.ndarray, text_boxes: list[dict]) -> np.ndarray:
    panel = image.copy()
    for box in text_boxes:
        points = np.round(np.asarray(box["bbox"], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        if "width_ft" in box or "area_sqft" in box:
            color = (40, 180, 40)
        elif OCRScaleParser._is_room_label(str(box.get("text", ""))):
            color = (200, 120, 20)
        else:
            color = (160, 160, 160)
        cv2.polylines(panel, [points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        anchor = (int(points[0, 0, 0]), max(16, int(points[0, 0, 1]) - 6))
        label = f"{box['text']} ({float(box['confidence']):.2f})"
        draw_text(panel, label, anchor, 0.4)
    return panel


def draw_fused(image: np.ndarray, rooms: list[dict]) -> np.ndarray:
    panel = image.copy()
    visible = [room for room in rooms if isinstance(room["polygon"], Polygon) and not room["polygon"].is_empty]
    fill = np.zeros_like(panel)
    for index, room in enumerate(visible):
        coords = np.round(np.asarray(room["polygon"].exterior.coords, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(fill, [coords.reshape(-1, 1, 2)], room_color(index, len(visible)))
    blended = cv2.addWeighted(panel, 0.55, fill, 0.45, 0)
    occupied = np.any(fill > 0, axis=2)
    panel[occupied] = blended[occupied]

    for index, room in enumerate(visible):
        polygon = room["polygon"]
        coords = np.round(np.asarray(polygon.exterior.coords, dtype=np.float32)).astype(np.int32)
        cv2.polylines(panel, [coords.reshape(-1, 1, 2)], True, (255, 255, 255), 2, cv2.LINE_AA)
        lines = []
        if room["labels"]:
            lines.append(", ".join(room["labels"]))
        if room["dimensions"]:
            lines.append(str(room["dimensions"][0].get("text", "")))
        if room.get("nominal_area_sqft"):
            lines.append(f"{float(room['nominal_area_sqft']):.1f} sqft nominal")
        elif room.get("estimated_area_sqft") and not room.get("fallback_mode"):
            lines.append(f"{float(room['estimated_area_sqft']):.1f} sqft est.")
        lines.append(f"{float(room['relative_area_pct']):.1f}%")
        centroid = polygon.centroid
        x = int(round(centroid.x))
        y = int(round(centroid.y)) - 8 * (len(lines) - 1)
        for line in lines:
            draw_text(panel, line, (x, y))
            y += 16
    return panel


def regularized_polygons(extractor, segmenter, regularizer, image: np.ndarray) -> list[Polygon]:
    wall_mask = extractor.extract_wall_mask(image)
    closed_mask = extractor.close_door_gaps(wall_mask)
    seeds, _distance = extractor.generate_room_seeds(closed_mask)
    masks = segmenter.segment(image, seeds, closed_mask)
    epsilon_ratio = float(regularizer.geometry["douglas_peucker_epsilon"])
    tolerance_deg = float(regularizer.geometry["ortho_snap_tolerance_deg"])
    snapped = []
    for mask in masks:
        polygon = regularizer.mask_to_polygon(mask)
        simplified = regularizer.simplify_polygon(polygon, epsilon_ratio)
        snapped.append(regularizer.snap_to_manhattan_orthogonality(simplified, tolerance_deg))
    return regularizer.resolve_spatial_overlaps(snapped)


def process_image(extractor, segmenter, regularizer, parser: OCRScaleParser, image_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    denoised = cv2.bilateralFilter(
        image,
        d=int(extractor.preprocessing["bilateral_d"]),
        sigmaColor=float(extractor.preprocessing["bilateral_sigma_color"]),
        sigmaSpace=float(extractor.preprocessing["bilateral_sigma_space"]),
    )
    polygons = regularized_polygons(extractor, segmenter, regularizer, denoised)
    text_boxes = parser.extract_text_boxes(image)
    parsed, scale_ratio = parser.parse_dimensions(text_boxes)
    rooms = parser.associate_text_with_polygons(polygons, text_boxes)

    output_dir = DEBUG_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "10_ocr_detected_text_boxes.png"), draw_detections(image, text_boxes))
    cv2.imwrite(str(output_dir / "11_fused_spatial_context.png"), draw_fused(image, rooms))

    labels = [label for room in rooms for label in room["labels"]]
    print(
        f"{image_path.name}: texts={len(text_boxes)} dimensions={len(parsed)} "
        f"scale_ratio={None if scale_ratio is None else round(scale_ratio, 3)} "
        f"fallback={parser.fallback_mode} labels={labels} -> {output_dir}"
    )


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")

    config_path = ROOT / "configs" / "config.yaml"
    extractor = WallExtractor(config_path)
    segmenter = RoomSegmenter(config_path)
    regularizer = GeometryRegularizer(config_path)
    parser = OCRScaleParser(config_path)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        process_image(extractor, segmenter, regularizer, parser, image_path)
    print(f"OCR overlays written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
