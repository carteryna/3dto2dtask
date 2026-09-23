#!/usr/bin/env python3
"""CLI for the 3D floor-plan to 2D room-layout pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from src.geometry import GeometryRegularizer
from src.metrics import (
    compute_boundary_iou,
    compute_iou,
    compute_orthogonality_deviation,
    compute_relative_area_error,
)
from src.ocr import OCRScaleParser
from src.preprocessing import WallExtractor
from src.segmentation import RoomSegmenter
from src.visualizer import PipelineVisualizer

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a 3D floor-plan render into regularized room polygons.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to one floor-plan image.")
    source.add_argument("--input_dir", help="Directory of floor-plan images to process as a batch.")
    parser.add_argument("--output_dir", default="data/output", help="Directory for result.json and annotated overlays.")
    parser.add_argument(
        "--method",
        choices=("hybrid", "watershed"),
        default="hybrid",
        help="hybrid runs FastSAM with a watershed fallback; watershed uses the classical fallback only.",
    )
    parser.add_argument("--config", default=None, help="Path to configs/config.yaml.")
    return parser.parse_args()


def list_images(input_path: str | None, input_dir: str | None) -> list[Path]:
    if input_path:
        path = Path(input_path)
        if not path.is_file():
            raise SystemExit(f"Input image not found: {path}")
        return [path]
    folder = Path(input_dir)
    images = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No floor-plan images found in {folder}")
    return images


def regularize_masks(regularizer: GeometryRegularizer, masks: list[np.ndarray]) -> list[Polygon]:
    epsilon_ratio = float(regularizer.geometry["douglas_peucker_epsilon"])
    tolerance_deg = float(regularizer.geometry["ortho_snap_tolerance_deg"])
    snapped: list[Polygon] = []
    for mask in masks:
        polygon = regularizer.mask_to_polygon(mask)
        simplified = regularizer.simplify_polygon(polygon, epsilon_ratio)
        snapped.append(regularizer.snap_to_manhattan_orthogonality(simplified, tolerance_deg))
    return regularizer.resolve_spatial_overlaps(snapped)


def raw_polygons(regularizer: GeometryRegularizer, masks: list[np.ndarray]) -> list[Polygon]:
    return [regularizer.mask_to_polygon(mask) for mask in masks]


def pipeline_method_name(method: str) -> str:
    if method == "watershed":
        return "classical_watershed_shapely"
    return "hybrid_fastsam_shapely"


def room_label(record: dict, room_id: int) -> str:
    labels = [str(label).strip() for label in record.get("labels", []) if str(label).strip()]
    if labels:
        return ", ".join(labels)
    return f"Room {room_id}"


def room_sqft(record: dict) -> float | None:
    nominal = record.get("nominal_area_sqft")
    if nominal:
        return round(float(nominal), 1)
    estimated = record.get("estimated_area_sqft")
    if estimated:
        return round(float(estimated), 1)
    return None


def polygon_vertices(polygon: Polygon) -> list[list[int]]:
    coords = list(polygon.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    vertices: list[list[int]] = []
    for x, y in coords:
        point = [int(round(float(x))), int(round(float(y)))]
        if not vertices or point != vertices[-1]:
            vertices.append(point)
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    return vertices


def polygon_bbox(polygon: Polygon) -> list[int]:
    min_x, min_y, max_x, max_y = polygon.bounds
    return [int(round(min_x)), int(round(min_y)), int(round(max_x)), int(round(max_y))]


def build_payload(
    image_name: str,
    image: np.ndarray,
    method: str,
    parser: OCRScaleParser,
    records: list[dict],
    confidences: list[float],
) -> tuple[dict, list[dict]]:
    height, width = image.shape[:2]
    export_rooms = []
    visual_rooms = []
    for index, record in enumerate(records):
        polygon = record.get("polygon")
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue
        room_id = len(export_rooms) + 1
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        sqft = room_sqft(record)
        label = room_label(record, room_id)
        export_rooms.append(
            {
                "room_id": room_id,
                "label": label,
                "confidence": round(confidence, 2),
                "pixel_area": int(round(float(polygon.area))),
                "relative_area_pct": round(float(record.get("relative_area_pct") or 0.0), 2),
                "estimated_sqft": sqft,
                "polygon_vertices": polygon_vertices(polygon),
                "bbox": polygon_bbox(polygon),
            }
        )
        visual_rooms.append(
            {
                "polygon": polygon,
                "room_id": room_id,
                "label": label,
                "relative_area_pct": record.get("relative_area_pct"),
                "estimated_sqft": sqft,
            }
        )
    payload = {
        "metadata": {
            "image_name": image_name,
            "dimensions": {"width": int(width), "height": int(height)},
            "pipeline_method": pipeline_method_name(method),
            "scale_px_per_ft": None if parser.scale_ratio is None else round(float(parser.scale_ratio), 3),
            "fallback_mode": bool(parser.fallback_mode),
            "total_detected_rooms": len(export_rooms),
        },
        "rooms": export_rooms,
    }
    return payload, visual_rooms


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def agreement_metrics(prediction: list[np.ndarray], reference: list[np.ndarray]) -> dict:
    """Score hybrid masks against classical watershed masks, matched by best IoU."""
    ious: list[float] = []
    boundary_ious: list[float] = []
    area_errors: list[float] = []
    used: set[int] = set()
    for pred in prediction:
        best_index = None
        best_score = -1.0
        for index, reference_mask in enumerate(reference):
            if index in used:
                continue
            score = compute_iou(pred, reference_mask)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            continue
        used.add(best_index)
        reference_mask = reference[best_index]
        ious.append(best_score)
        boundary_ious.append(compute_boundary_iou(pred, reference_mask))
        area_errors.append(
            compute_relative_area_error(float(np.count_nonzero(pred)), float(np.count_nonzero(reference_mask)))
        )
    return {
        "mean_iou": mean(ious),
        "mean_boundary_iou": mean(boundary_ious),
        "mean_rae_pct": mean(area_errors),
        "matched_rooms": len(ious),
    }


def mean_orthogonality(polygons: list[Polygon]) -> float | None:
    scores = [compute_orthogonality_deviation(polygon) for polygon in polygons if not polygon.is_empty]
    return mean(scores)


def process_image(
    image_path: Path,
    method: str,
    extractor: WallExtractor,
    segmenter: RoomSegmenter,
    regularizer: GeometryRegularizer,
    parser: OCRScaleParser,
    visualizer: PipelineVisualizer,
    output_dir: Path,
) -> tuple[dict, dict]:
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    denoised = extractor.load_and_denoise(str(image_path))
    wall_mask = extractor.extract_wall_mask(denoised)
    closed_mask = extractor.close_door_gaps(wall_mask)
    seeds, _distance = extractor.generate_room_seeds(closed_mask)

    segmenter.segmentation["model_type"] = "fastsam" if method == "hybrid" else "watershed_fallback"
    room_masks = segmenter.segment(denoised, seeds, closed_mask)
    confidences = list(segmenter.mask_confidences)

    segmenter.segmentation["model_type"] = "watershed_fallback"
    watershed_masks = segmenter.segment(denoised, seeds, closed_mask)
    segmenter.segmentation["model_type"] = "fastsam" if method == "hybrid" else "watershed_fallback"

    polygons = regularize_masks(regularizer, room_masks)
    watershed_contours = raw_polygons(regularizer, watershed_masks)
    text_boxes = parser.extract_text_boxes(original)
    parser.parse_dimensions(text_boxes)
    records = parser.associate_text_with_polygons(polygons, text_boxes)
    for record, confidence in zip(records, confidences):
        record["confidence"] = confidence

    payload, visual_rooms = build_payload(image_path.name, original, method, parser, records, confidences)
    annotated = visualizer.save_annotated(original, visual_rooms, image_path.name, output_dir)
    metrics = {
        "image_name": image_path.name,
        "agreement_with_watershed": agreement_metrics(room_masks, watershed_masks),
        "hybrid_orthogonality_deg": mean_orthogonality(polygons),
        "watershed_orthogonality_deg": mean_orthogonality(watershed_contours),
    }
    print(
        f"{image_path.name}: rooms={payload['metadata']['total_detected_rooms']} "
        f"fallback={payload['metadata']['fallback_mode']} "
        f"scale={payload['metadata']['scale_px_per_ft']} -> {annotated}"
    )
    return payload, metrics


def aggregate_metrics(rows: list[dict]) -> dict:
    def collect(key: str) -> list[float]:
        values = []
        for row in rows:
            if key in row and row[key] is not None:
                values.append(float(row[key]))
            nested = row.get("agreement_with_watershed", {})
            if key in nested and nested[key] is not None:
                values.append(float(nested[key]))
        return values

    agreement = [row["agreement_with_watershed"] for row in rows]
    return {
        "mean_iou": mean([item["mean_iou"] for item in agreement if item["mean_iou"] is not None]),
        "mean_boundary_iou": mean(
            [item["mean_boundary_iou"] for item in agreement if item["mean_boundary_iou"] is not None]
        ),
        "mean_rae_pct": mean([item["mean_rae_pct"] for item in agreement if item["mean_rae_pct"] is not None]),
        "hybrid_orthogonality_deg": mean(collect("hybrid_orthogonality_deg")),
        "watershed_orthogonality_deg": mean(collect("watershed_orthogonality_deg")),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config
    extractor = WallExtractor(config_path)
    segmenter = RoomSegmenter(config_path)
    regularizer = GeometryRegularizer(config_path)
    parser = OCRScaleParser(config_path)
    visualizer = PipelineVisualizer()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    metric_rows = []
    for image_path in list_images(args.input, args.input_dir):
        payload, metrics = process_image(
            image_path, args.method, extractor, segmenter, regularizer, parser, visualizer, output_dir
        )
        payloads.append(payload)
        metric_rows.append(metrics)

    result_path = output_dir / "result.json"
    document = payloads[0] if len(payloads) == 1 else payloads
    result_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    summary = aggregate_metrics(metric_rows)
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps({"per_image": metric_rows, "aggregate": summary}, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")
    print(
        "Aggregate vs watershed: "
        f"IoU={summary['mean_iou']} boundary_IoU={summary['mean_boundary_iou']} "
        f"RAE={summary['mean_rae_pct']} "
        f"ortho_hybrid={summary['hybrid_orthogonality_deg']} "
        f"ortho_watershed={summary['watershed_orthogonality_deg']}"
    )


if __name__ == "__main__":
    main()
