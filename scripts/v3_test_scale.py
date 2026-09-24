#!/usr/bin/env python3
"""Visual QA for Iteration 3 Phase 4 global area scale.

Run inside the container:

    python scripts/v3_test_scale.py

Phases 1 to 3 build the room polygons. Phase 4 reads the square footage from
the filename and converts the combined pixel area into pixels per foot. A
second pass renames each file so it has no size, which exercises the YOLOv8n
toilet fallback. Gap widths are not used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v3_topology.geometry import ManhattanLineGraph
from src.v3_topology.scale import AreaScaleEstimator
from src.v3_topology.segmentation import FloorSegmenter
from src.v3_topology.topology import TopologyPartitioner

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
INPUT_DIR = ROOT / "inout_data"


def list_inputs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    images = list_inputs(INPUT_DIR)
    if not images:
        raise SystemExit(f"No floor-plan images found in {INPUT_DIR}")
    segmenter = FloorSegmenter(ROOT / "configs" / "config.yaml")
    partitioner = TopologyPartitioner(ROOT / "configs" / "config.yaml")
    graph = ManhattanLineGraph(ROOT / "configs" / "config.yaml")
    estimator = AreaScaleEstimator(ROOT / "configs" / "config.yaml")
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        masks = segmenter.segment(image)
        footprint = segmenter.walkable_footprint(masks)
        seeds = partitioner.seeds(masks)
        cells = partitioner.partition(footprint, seeds, masks)
        polygons = graph.polygons(graph.orthogonalize(graph.extract_segments(cells)))
        scale = estimator.estimate(polygons, image_path.name, image)
        result = estimator.result
        assert result is not None
        rebuilt = result.pixel_area / (scale ** 2) if scale else None
        print(
            f"{image_path.name}: sqft={result.sqft} px={result.pixel_area:.0f} "
            f"scale={None if scale is None else round(scale, 3)} px/ft "
            f"source={result.source} rebuilt_sqft={None if rebuilt is None else round(rebuilt, 2)}"
        )
        fallback = estimator.estimate(polygons, "unit.webp", image)
        fallback_result = estimator.result
        assert fallback_result is not None
        anchor_text = ", ".join(
            f"{anchor.kind}:{anchor.pixel_span:.0f}px->{anchor.scale_px_per_ft:.2f}"
            for anchor in fallback_result.anchors
        ) or "none"
        print(f"  fallback without a size: scale={None if fallback is None else round(fallback, 3)} anchors={anchor_text}")
        for note in fallback_result.notes:
            print(f"  {note}")


if __name__ == "__main__":
    main()
