#!/usr/bin/env python3
"""Visual QA for Phase 4 polygon regularization.

Run inside the container:

    python scripts/test_geometry.py

Rebuilds Phase 3 room masks for every render in ``inout_data/`` and writes
vector debug panels under ``data/output/debug/<image_stem>/``.
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

from src.v1_baseline.geometry import GeometryRegularizer
from src.v1_baseline.preprocessing import WallExtractor
from src.v1_baseline.segmentation import RoomSegmenter

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


def ring_points(polygon: Polygon) -> np.ndarray | None:
    if polygon.is_empty:
        return None
    coords = np.asarray(polygon.exterior.coords, dtype=np.float32)
    if len(coords) < 2:
        return None
    return np.round(coords).astype(np.int32).reshape(-1, 1, 2)


def draw_ring(image: np.ndarray, polygon: Polygon, color: tuple[int, int, int], thickness: int) -> None:
    points = ring_points(polygon)
    if points is None:
        return
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_comparison(image: np.ndarray, raw_polygons: list[Polygon], simplified: list[Polygon]) -> np.ndarray:
    """Raw mask contours in red, Douglas-Peucker vectors in green."""
    panel = image.copy()
    shade = np.zeros_like(panel)
    panel = cv2.addWeighted(panel, 0.75, shade, 0.25, 0)
    for polygon in raw_polygons:
        draw_ring(panel, polygon, (40, 40, 220), 1)
    for polygon in simplified:
        draw_ring(panel, polygon, (40, 200, 40), 2)
    cv2.putText(panel, "red: raw contour", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 220), 2, cv2.LINE_AA)
    cv2.putText(panel, "green: simplified", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 200, 40), 2, cv2.LINE_AA)
    return panel


def draw_regularized(image: np.ndarray, polygons: list[Polygon]) -> np.ndarray:
    """Ortho polygons with vertices, bounding boxes, and pixel-area labels."""
    panel = image.copy()
    visible = [polygon for polygon in polygons if not polygon.is_empty]
    total_area = sum(polygon.area for polygon in visible) or 1.0
    fill = np.zeros_like(panel)
    for index, polygon in enumerate(visible):
        points = ring_points(polygon)
        if points is None:
            continue
        cv2.fillPoly(fill, [points], room_color(index, len(visible)))
    blended = cv2.addWeighted(panel, 0.55, fill, 0.45, 0)
    occupied = np.any(fill > 0, axis=2)
    panel[occupied] = blended[occupied]

    for index, polygon in enumerate(visible):
        color = room_color(index, len(visible))
        points = ring_points(polygon)
        if points is None:
            continue
        cv2.polylines(panel, [points], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for vertex in points.reshape(-1, 2):
            cv2.circle(panel, (int(vertex[0]), int(vertex[1])), 3, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(panel, (int(vertex[0]), int(vertex[1])), 4, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        min_x, min_y, max_x, max_y = polygon.bounds
        cv2.rectangle(
            panel,
            (int(round(min_x)), int(round(min_y))),
            (int(round(max_x)), int(round(max_y))),
            color,
            1,
            lineType=cv2.LINE_AA,
        )
        centroid = polygon.centroid
        label = f"{polygon.area:.0f}px {100.0 * polygon.area / total_area:.1f}%"
        origin = (int(round(centroid.x)), int(round(centroid.y)))
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def process_image(
    extractor: WallExtractor,
    segmenter: RoomSegmenter,
    regularizer: GeometryRegularizer,
    image_path: Path,
) -> None:
    denoised = extractor.load_and_denoise(str(image_path))
    wall_mask = extractor.extract_wall_mask(denoised)
    closed_mask = extractor.close_door_gaps(wall_mask)
    seeds, _distance = extractor.generate_room_seeds(closed_mask)
    masks = segmenter.segment(denoised, seeds, closed_mask)

    epsilon_ratio = float(regularizer.geometry["douglas_peucker_epsilon"])
    tolerance_deg = float(regularizer.geometry["ortho_snap_tolerance_deg"])
    raw_polygons: list[Polygon] = []
    simplified: list[Polygon] = []
    snapped: list[Polygon] = []
    for mask in masks:
        polygon = regularizer.mask_to_polygon(mask)
        raw_polygons.append(polygon)
        simple = regularizer.simplify_polygon(polygon, epsilon_ratio)
        simplified.append(simple)
        snapped.append(regularizer.snap_to_manhattan_orthogonality(simple, tolerance_deg))
    regularized = regularizer.resolve_spatial_overlaps(snapped)

    output_dir = DEBUG_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "08_raw_vs_simplified_contours.png"), draw_comparison(denoised, raw_polygons, simplified))
    cv2.imwrite(str(output_dir / "09_ortho_regularized_polygons.png"), draw_regularized(denoised, regularized))

    kept = sum(not polygon.is_empty for polygon in regularized)
    print(
        f"{image_path.name}: masks={len(masks)} polygons={kept} "
        f"epsilon_ratio={epsilon_ratio} tolerance_deg={tolerance_deg} -> {output_dir}"
    )


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")

    config_path = ROOT / "configs" / "config.yaml"
    extractor = WallExtractor(config_path)
    segmenter = RoomSegmenter(config_path)
    regularizer = GeometryRegularizer(config_path)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        process_image(extractor, segmenter, regularizer, image_path)
    print(f"Vector overlays written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
