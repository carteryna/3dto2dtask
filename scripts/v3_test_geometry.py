#!/usr/bin/env python3
"""Visual QA for Iteration 3 Phase 3 Manhattan line graph.

Run inside the container:

    python scripts/v3_test_geometry.py

Phases 1 and 2 build the watershed cells. Phase 3 turns their boundaries
into orthogonal lines and room polygons. Panels are written to
``data/output/debug_v3/<image_stem>/``:

- ``07_boundary_segments.png``: cell outlines before orthogonalization.
- ``08_orthogonal_lines.png``: the horizontal and vertical wall graph.
- ``09_room_polygons.png``: faces of that graph, with shared corners.
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

from src.v3_topology.geometry import ManhattanLineGraph
from src.v3_topology.segmentation import FloorSegmenter
from src.v3_topology.topology import TopologyPartitioner

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v3"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def draw_segments(canvas: np.ndarray, segments: list[np.ndarray], color: tuple[int, int, int], thickness: int) -> None:
    for segment in segments:
        points = np.asarray(segment, dtype=np.int32).reshape(-1, 2)
        cv2.line(canvas, tuple(points[0]), tuple(points[1]), color, thickness, cv2.LINE_AA)


def draw_polygons(canvas: np.ndarray, polygons: list[Polygon]) -> None:
    rng = np.random.default_rng(7)
    for index, polygon in enumerate(polygons, start=1):
        color = tuple(int(channel) for channel in rng.integers(40, 210, size=3))
        exterior = np.array(polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(canvas, [exterior], color)
        for interior in polygon.interiors:
            hole = np.array(interior.coords, dtype=np.int32)
            cv2.fillPoly(canvas, [hole], (255, 255, 255))
        cv2.polylines(canvas, [exterior], True, (20, 20, 20), 2, cv2.LINE_AA)
        center = polygon.representative_point()
        cv2.putText(
            canvas,
            str(index),
            (int(center.x) + 4, int(center.y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def overlap_area(polygons: list[Polygon]) -> float:
    total = 0.0
    for index, polygon in enumerate(polygons):
        for other in polygons[index + 1 :]:
            if polygon.intersects(other):
                total += float(polygon.intersection(other).area)
    return total


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    segmenter = FloorSegmenter(ROOT / "configs" / "config.yaml")
    partitioner = TopologyPartitioner(ROOT / "configs" / "config.yaml")
    graph = ManhattanLineGraph(ROOT / "configs" / "config.yaml")
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        masks = segmenter.segment(image)
        footprint = segmenter.walkable_footprint(masks)
        room_seeds = partitioner.seeds(masks)
        cells = partitioner.partition(footprint, room_seeds, masks)
        raw = graph.extract_segments(cells)
        orthogonal = graph.orthogonalize(raw)
        rooms = graph.polygons(orthogonal)
        debug = graph.debug
        assert debug is not None

        folder = DEBUG_DIR / image_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        raw_panel = image.copy()
        draw_segments(raw_panel, raw, (40, 40, 210), 1)
        cv2.imwrite(str(folder / "07_boundary_segments.png"), raw_panel)
        line_panel = np.full_like(image, 255)
        draw_segments(line_panel, orthogonal, (20, 20, 20), 2)
        cv2.imwrite(str(folder / "08_orthogonal_lines.png"), line_panel)
        polygon_panel = np.full_like(image, 255)
        draw_polygons(polygon_panel, rooms)
        cv2.imwrite(str(folder / "09_room_polygons.png"), polygon_panel)

        footprint_area = int(np.count_nonzero(footprint))
        polygon_area = sum(float(room.area) for room in rooms)
        print(
            f"{image_path.name}: raw={debug.raw_segments} orthogonal={debug.orthogonal_segments} "
            f"axis_aligned={debug.axis_aligned} polygons={len(rooms)} cells={len(cells)} "
            f"overlap_px={overlap_area(rooms):.0f} "
            f"area_ratio={polygon_area / footprint_area if footprint_area else 0:.3f}"
        )


if __name__ == "__main__":
    main()
