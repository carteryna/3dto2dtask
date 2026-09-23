"""Quantitative checks for room masks and regularized polygons."""

from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Polygon


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Standard intersection over union of two binary masks."""
    prediction = _binary(pred_mask)
    ground_truth = _binary(gt_mask)
    intersection = int(np.logical_and(prediction, ground_truth).sum())
    union = int(np.logical_or(prediction, ground_truth).sum())
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_boundary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, band_width: int = 6) -> float:
    """IoU of the masks restricted to a perimeter band around each region.

    The band is the set of pixels within ``band_width`` of the mask contour,
    produced by the difference of a dilation and an erosion.
    """
    prediction = _boundary_band(pred_mask, band_width)
    ground_truth = _boundary_band(gt_mask, band_width)
    intersection = int(np.logical_and(prediction, ground_truth).sum())
    union = int(np.logical_or(prediction, ground_truth).sum())
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_relative_area_error(pred_area: float, gt_area: float) -> float:
    """Percentage error of a predicted area against a reference area.

    RAE = |Area_pred - Area_gt| / Area_gt * 100.
    """
    if gt_area == 0:
        return 0.0 if pred_area == 0 else float("inf")
    return abs(float(pred_area) - float(gt_area)) / float(gt_area) * 100.0


def compute_orthogonality_deviation(polygon: Polygon) -> float:
    """Mean absolute deviation of edge angles from the nearest Manhattan axis.

    A perfectly axis-aligned ring scores 0. A 45-degree edge scores 45. The
    value is the mean, in degrees, of each wall's distance to 0° or 90°.
    """
    if polygon.is_empty or polygon.exterior is None:
        return 0.0
    coords = list(polygon.exterior.coords)
    if len(coords) < 2:
        return 0.0
    deviations: list[float] = []
    for start, end in zip(coords, coords[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        if math.hypot(dx, dy) < 1.0:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        to_horizontal = min(angle, 180.0 - angle)
        to_vertical = abs(angle - 90.0)
        deviations.append(min(to_horizontal, to_vertical))
    if not deviations:
        return 0.0
    return float(sum(deviations) / len(deviations))


def _boundary_band(mask: np.ndarray, band_width: int) -> np.ndarray:
    binary = _binary(mask).astype(np.uint8)
    if band_width <= 0 or binary.max() == 0:
        return binary.astype(bool)
    size = int(band_width) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(binary, kernel)
    eroded = cv2.erode(binary, kernel)
    return (dilated > 0) & (eroded == 0)
