#!/usr/bin/env python3
"""CLI for the 3D floor-plan to 2D room-layout pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from src.metrics import (
    compute_boundary_iou,
    compute_iou,
    compute_orthogonality_deviation,
    compute_relative_area_error,
)
from src.v1_baseline.geometry import GeometryRegularizer
from src.v1_baseline.ocr import OCRScaleParser
from src.v1_baseline.preprocessing import WallExtractor
from src.v1_baseline.segmentation import RoomSegmenter
from src.v2_reconstruction.geometry import RectilinearFitter
from src.v2_reconstruction.homography import PerspectiveCorrector
from src.v2_reconstruction.scale import ScaleEstimator, rooms_from_rects
from src.v2_reconstruction.segmentation import FloorSeamSegmenter
from src.v2_reconstruction.visualizer import BlueprintRenderer
from src.v3_topology.geometry import ManhattanLineGraph
from src.v3_topology.scale import AreaScaleEstimator
from src.v3_topology.segmentation import FloorSegmenter
from src.v3_topology.topology import TopologyPartitioner
from src.v3_topology.visualizer import TopologyBlueprint
from src.visualizer import PipelineVisualizer

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a 3D floor-plan render into regularized room polygons.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to one floor-plan image.")
    source.add_argument("--input_dir", help="Directory of floor-plan images to process as a batch.")
    parser.add_argument("--output_dir", default="data/output", help="Directory for result.json and annotated overlays.")
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v3"),
        default="v3",
        help="v1 is the exploratory baseline; v2 is the rectilinear reconstruction; v3 is the shared-edge topology engine.",
    )
    parser.add_argument(
        "--method",
        choices=("hybrid", "watershed"),
        default="hybrid",
        help="v1 only: hybrid runs FastSAM with a watershed fallback; watershed uses the classical fallback only.",
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


def run_v1(args: argparse.Namespace) -> None:
    """Iteration 1 exploratory baseline: perspective-space room extraction."""
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


def _feet_inches(cm: float) -> str:
    total_inches = float(cm) / 2.54
    feet = int(total_inches // 12)
    inches = int(round(total_inches - feet * 12))
    if inches == 12:
        feet += 1
        inches = 0
    return f"{feet}'{inches}\""


def _dimension_text(record: dict, rect: dict, pixels_per_cm: float | None) -> str:
    pixel_w = float(rect["w"])
    pixel_h = float(rect["h"])
    pixel_aspect = pixel_w / pixel_h if pixel_h else 0.0
    for dimension in record.get("dimensions") or []:
        width_ft = float(dimension.get("width_ft") or 0.0)
        height_ft = float(dimension.get("height_ft") or 0.0)
        if width_ft <= 0 or height_ft <= 0 or pixel_aspect <= 0:
            continue
        direct = abs(pixel_aspect - (width_ft / height_ft))
        swapped = abs(pixel_aspect - (height_ft / width_ft))
        # A callout pasted onto a wing of the wrong shape is left off the drawing.
        if min(direct, swapped) > 0.4:
            continue
        if swapped < direct:
            width_ft, height_ft = height_ft, width_ft
        return f"{_feet_inches(width_ft * 30.48)} x {_feet_inches(height_ft * 30.48)}"
    if pixels_per_cm is None or pixels_per_cm <= 0:
        return ""
    return f"{_feet_inches(pixel_w / pixels_per_cm)} x {_feet_inches(pixel_h / pixels_per_cm)}"


def build_v2_rooms(rects: list[dict], records: list[dict], pixels_per_cm: float | None, sqft_per_sqcm: float) -> list[dict]:
    areas = [max(int(rect["w"]), 0) * max(int(rect["h"]), 0) for rect in rects]
    total = sum(areas) or 1
    rooms = []
    for index, rect in enumerate(rects):
        record = records[index] if index < len(records) else {}
        labels = [str(label).strip() for label in record.get("labels", []) if str(label).strip()]
        pixel_area = areas[index]
        area_sqft = None
        width_cm = None
        height_cm = None
        if pixels_per_cm and pixels_per_cm > 0:
            width_cm = float(rect["w"]) / pixels_per_cm
            height_cm = float(rect["h"]) / pixels_per_cm
            area_sqft = width_cm * height_cm * sqft_per_sqcm
        nominal = None
        for dimension in record.get("dimensions") or []:
            if dimension.get("area_sqft"):
                nominal = float(dimension["area_sqft"])
                break
        rooms.append(
            {
                "x": int(rect["x"]),
                "y": int(rect["y"]),
                "w": int(rect["w"]),
                "h": int(rect["h"]),
                "label": ", ".join(labels) if labels else f"Room {index + 1}",
                "dimensions": _dimension_text(record, rect, pixels_per_cm),
                "relative_area_pct": 100.0 * pixel_area / total,
                "pixel_area": pixel_area,
                "width_cm": None if width_cm is None else round(width_cm, 1),
                "height_cm": None if height_cm is None else round(height_cm, 1),
                "area_sqft": None if area_sqft is None else round(area_sqft, 1),
                "nominal_sqft": None if nominal is None else round(nominal, 1),
            }
        )
    return rooms


def v2_payload(image_name: str, shape: tuple[int, int], rooms: list[dict], scale: float | None, source: str) -> dict:
    height, width = int(shape[0]), int(shape[1])
    export = []
    for index, room in enumerate(rooms, start=1):
        x, y, room_w, room_h = int(room["x"]), int(room["y"]), int(room["w"]), int(room["h"])
        export.append(
            {
                "room_id": index,
                "label": room["label"],
                "dimensions": room["dimensions"],
                "pixel_area": int(room["pixel_area"]),
                "relative_area_pct": round(float(room["relative_area_pct"]), 2),
                "width_cm": room["width_cm"],
                "height_cm": room["height_cm"],
                "area_sqft": room["area_sqft"],
                "nominal_sqft": room["nominal_sqft"],
                "polygon_vertices": [[x, y], [x + room_w, y], [x + room_w, y + room_h], [x, y + room_h]],
                "bbox": [x, y, x + room_w, y + room_h],
            }
        )
    return {
        "metadata": {
            "image_name": image_name,
            "dimensions": {"width": width, "height": height},
            "pipeline_method": "v2_rectilinear_blueprint",
            "scale_px_per_cm": None if scale is None else round(float(scale), 4),
            "scale_source": source,
            "total_detected_rooms": len(export),
        },
        "rooms": export,
    }


def reconstruct_v2(image: np.ndarray, image_name: str, config_path: str | None) -> tuple[np.ndarray, dict]:
    """Flatten, segment, fit, scale, and draw one blueprint."""
    corrector = PerspectiveCorrector(config_path)
    segmenter = FloorSeamSegmenter(config_path)
    fitter = RectilinearFitter(config_path)
    estimator = ScaleEstimator(config_path)
    renderer = BlueprintRenderer(config_path)
    parser = OCRScaleParser(config_path)

    flattened = corrector.correct(image).flattened
    floors = segmenter.segment(flattened)
    tolerance = int(fitter.params["grid_snap_tolerance_px"])
    rects = fitter.snap_to_global_grid(
        fitter.split_open_plan_areas(fitter.fit_bounding_rectangles(floors), flattened),
        tolerance,
    )
    polygons = rooms_from_rects(rects)
    scale = estimator.estimate(flattened, polygons)
    source = estimator.result.source if estimator.result is not None else "none"
    try:
        boxes = parser.extract_text_boxes(flattened)
        parser.parse_dimensions(boxes)
        records = parser.associate_text_with_polygons(polygons, boxes)
    except Exception as exc:
        print(f"OCR labels failed ({exc}); rooms stay unlabeled.")
        records = []
    rooms = build_v2_rooms(rects, records, scale, float(renderer.params["sqft_per_sqcm"]))
    blueprint = renderer.render(flattened.shape[:2], rooms)
    return blueprint, v2_payload(image_name, flattened.shape[:2], rooms, scale, source)


def run_v2(args: argparse.Namespace) -> None:
    """Iteration 2: flatten, fit rectilinear rooms, and write a CAD blueprint."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for image_path in list_images(args.input, args.input_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        blueprint, payload = reconstruct_v2(image, image_path.name, args.config)
        blueprint_path = output_dir / f"blueprint_{image_path.stem}.png"
        cv2.imwrite(str(blueprint_path), blueprint)
        payloads.append(payload)
        print(
            f"{image_path.name}: rooms={payload['metadata']['total_detected_rooms']} "
            f"scale={payload['metadata']['scale_px_per_cm']} px/cm "
            f"source={payload['metadata']['scale_source']} -> {blueprint_path}"
        )
    result_path = output_dir / "result_v2.json"
    document = payloads[0] if len(payloads) == 1 else payloads
    result_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")


def v3_payload(
    image_name: str,
    shape: tuple[int, int],
    rooms: list[dict],
    scale: float | None,
    source: str,
    filename_sqft: float | None,
) -> dict:
    height, width = int(shape[0]), int(shape[1])
    export = []
    for room in rooms:
        polygon: Polygon = room["polygon"]
        export.append(
            {
                "room_id": int(room["room_id"]),
                "pixel_area": int(round(room["pixel_area"])),
                "relative_area_pct": round(float(room["relative_area_pct"]), 2),
                "area_sqft": None if room["area_sqft"] is None else round(float(room["area_sqft"]), 1),
                "centroid": [round(room["centroid"][0], 1), round(room["centroid"][1], 1)],
                "polygon_vertices": [[int(x), int(y)] for x, y in polygon.exterior.coords],
                "holes": [[[int(x), int(y)] for x, y in ring.coords] for ring in polygon.interiors],
            }
        )
    return {
        "metadata": {
            "image_name": image_name,
            "dimensions": {"width": width, "height": height},
            "pipeline_method": "v3_shared_edge_topology",
            "scale_px_per_ft": None if scale is None else round(float(scale), 4),
            "scale_source": source,
            "filename_sqft": filename_sqft,
            "total_detected_rooms": len(export),
        },
        "rooms": export,
    }


def reconstruct_v3(image: np.ndarray, image_name: str, config_path: str | None) -> tuple[np.ndarray, dict]:
    """Segment, partition, orthogonalize, scale, and draw one blueprint."""
    segmenter = FloorSegmenter(config_path)
    partitioner = TopologyPartitioner(config_path)
    graph = ManhattanLineGraph(config_path)
    estimator = AreaScaleEstimator(config_path)
    renderer = TopologyBlueprint(config_path)

    masks = segmenter.segment(image)
    footprint = segmenter.walkable_footprint(masks)
    cells = partitioner.partition(footprint, partitioner.seeds(masks), masks)
    polygons = graph.polygons(graph.orthogonalize(graph.extract_segments(cells)))
    scale = estimator.estimate(polygons, image_name, image)
    result = estimator.result
    source = result.source if result is not None else "none"
    filename_sqft = None if result is None else result.sqft
    rooms = renderer.rooms(polygons, scale)
    blueprint = renderer.render(image.shape[:2], polygons, scale)
    return blueprint, v3_payload(image_name, image.shape[:2], rooms, scale, source, filename_sqft)


def run_v3(args: argparse.Namespace) -> None:
    """Iteration 3: shared-edge rooms, filename scale, and a white blueprint."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for image_path in list_images(args.input, args.input_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        blueprint, payload = reconstruct_v3(image, image_path.name, args.config)
        blueprint_path = output_dir / f"blueprint_v3_{image_path.stem}.png"
        cv2.imwrite(str(blueprint_path), blueprint)
        payloads.append(payload)
        scale = payload["metadata"]["scale_px_per_ft"]
        labeled = sum(room["area_sqft"] or 0 for room in payload["rooms"])
        print(
            f"{image_path.name}: rooms={payload['metadata']['total_detected_rooms']} "
            f"scale={scale} px/ft source={payload['metadata']['scale_source']} "
            f"labeled_sqft={round(labeled, 1)} -> {blueprint_path}"
        )
    result_path = output_dir / "result_v3.json"
    document = payloads[0] if len(payloads) == 1 else payloads
    result_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")


def main() -> None:
    args = parse_args()
    if args.version == "v1":
        run_v1(args)
    elif args.version == "v2":
        run_v2(args)
    else:
        run_v3(args)


if __name__ == "__main__":
    main()
