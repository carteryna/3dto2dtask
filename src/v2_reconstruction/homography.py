"""Phase 1: un-warp the 3D render into an orthographic top-down projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

Line = Tuple[np.ndarray, np.ndarray]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class HomographyResult:
    flattened: np.ndarray
    homography: np.ndarray
    source_quad: np.ndarray
    target_quad: np.ndarray
    footprint_contour: np.ndarray
    applied: bool


class PerspectiveCorrector:
    """Detect the apartment boundary quadrilateral and flatten the canvas.

    Apartment footprints are often L-shaped, so a convex-hull quad would cut
    diagonally across a missing corner. Instead, the outline is split into
    straight segments and the longest near-horizontal segment in the top and
    bottom halves, plus the longest near-vertical segment in the left and
    right halves, are refit with ``cv2.fitLine``. These four exterior walls are
    parallel pairs in the real building, so their intersections form a quad
    that maps to a rectangle under the camera homography.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config["homography"]

    def correct(self, image: np.ndarray) -> HomographyResult:
        contour = self.detect_footprint(image)
        quad = self.detect_boundary_quad(image, contour)
        homography, target = self.compute_homography(quad)
        applied = self._homography_is_plausible(homography, contour, quad)
        if not applied:
            homography = np.eye(3, dtype=np.float64)
            target = quad.copy()
        homography, size = self._fit_canvas(homography, contour)
        target = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), homography).reshape(4, 2) if applied else target
        flattened = self.warp(image, homography, size)
        return HomographyResult(
            flattened=flattened,
            homography=homography,
            source_quad=quad,
            target_quad=target.astype(np.float32),
            footprint_contour=contour,
            applied=applied,
        )

    def detect_footprint(self, image: np.ndarray) -> np.ndarray:
        """Return the outer contour of the apartment silhouette."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        chroma = np.maximum(
            np.abs(channel_a.astype(np.int16) - 128),
            np.abs(channel_b.astype(np.int16) - 128),
        )
        canvas = (
            (lightness >= int(self.params["canvas_lightness_min"]))
            & (chroma <= int(self.params["canvas_chroma_max"]))
        ).astype(np.uint8)
        count, labels = cv2.connectedComponents(canvas, connectivity=8)
        border_ids = set(
            np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])).tolist()
        )
        border_ids.discard(0)
        background = np.isin(labels, list(border_ids)) if count > 1 and border_ids else np.zeros(canvas.shape, bool)
        footprint = np.where(background, 0, 255).astype(np.uint8)

        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(gray, int(self.params["canny_low"]), int(self.params["canny_high"]))
        edges = cv2.bitwise_and(cv2.dilate(edges, np.ones((3, 3), np.uint8)), footprint)
        combined = cv2.bitwise_or(footprint, edges)

        close = int(self.params["footprint_close_px"])
        close = close if close % 2 else close + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            height, width = image.shape[:2]
            return np.array([[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]], [[0, height - 1]]], np.int32)
        return max(contours, key=cv2.contourArea)

    def detect_boundary_quad(self, image: np.ndarray, contour: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the four outer-wall corners ordered TL, TR, BR, BL."""
        if contour is None:
            contour = self.detect_footprint(image)
        lines = self._exterior_wall_lines(contour)
        if lines is not None:
            top, right, bottom, left = lines
            corners = [
                self._intersect(left, top),
                self._intersect(top, right),
                self._intersect(right, bottom),
                self._intersect(bottom, left),
            ]
            if all(corner is not None for corner in corners):
                quad = np.array(corners, dtype=np.float32)
                if cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                    return quad
        return self._hull_quad(contour)

    def compute_homography(self, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map the quad to an axis-aligned rectangle of the same scale."""
        top_left, top_right, bottom_right, bottom_left = quad
        width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
        height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
        target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
        return homography, target

    def warp(self, image: np.ndarray, homography: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.warpPerspective(
            image,
            homography,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    def outline_skew_deg(self, contour: np.ndarray) -> float:
        """Length-weighted mean deviation of outline segments from the nearest axis."""
        total = 0.0
        weighted = 0.0
        for start, end in self._outline_segments(contour):
            length = float(np.linalg.norm(end - start))
            angle = abs(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))) % 90.0
            weighted += min(angle, 90.0 - angle) * length
            total += length
        return weighted / total if total > 0 else 0.0

    def _outline_segments(self, contour: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        epsilon = float(self.params["approx_epsilon_start"]) * 0.5 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
        min_length = float(self.params["min_wall_length_ratio"]) * float(np.sqrt(cv2.contourArea(contour)))
        segments = []
        for index in range(len(polygon)):
            start, end = polygon[index], polygon[(index + 1) % len(polygon)]
            if np.linalg.norm(end - start) >= min_length:
                segments.append((start, end))
        return segments

    def _exterior_wall_lines(self, contour: np.ndarray) -> Optional[Tuple[Line, Line, Line, Line]]:
        """Pick and refit the dominant top, right, bottom and left exterior walls."""
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return None
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], np.float32)
        tolerance = float(self.params["wall_axis_tolerance_deg"])
        best: dict[str, Tuple[float, Tuple[np.ndarray, np.ndarray]]] = {}
        for start, end in self._outline_segments(contour):
            delta = end - start
            length = float(np.linalg.norm(delta))
            angle = abs(np.degrees(np.arctan2(delta[1], delta[0]))) % 180.0
            midpoint = (start + end) / 2.0
            if min(angle, 180.0 - angle) <= tolerance:
                side = "top" if midpoint[1] < center[1] else "bottom"
            elif abs(angle - 90.0) <= tolerance:
                side = "left" if midpoint[0] < center[0] else "right"
            else:
                continue
            if side not in best or length > best[side][0]:
                best[side] = (length, (start, end))
        if len(best) < 4:
            return None
        points = contour.reshape(-1, 2).astype(np.float32)
        band = float(self.params["side_fit_band_ratio"]) * float(cv2.arcLength(contour, True))
        return tuple(self._refit(points, *best[side][1], band) for side in ("top", "right", "bottom", "left"))

    @staticmethod
    def _refit(points: np.ndarray, start: np.ndarray, end: np.ndarray, band: float) -> Line:
        direction = end - start
        length = float(np.linalg.norm(direction))
        unit = direction / length
        relative = points - start
        along = relative @ unit
        across = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
        selected = points[(across <= band) & (along >= 0.0) & (along <= length)]
        if len(selected) < 10:
            return start.astype(np.float32), unit.astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)

    def _hull_quad(self, contour: np.ndarray) -> np.ndarray:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for ratio in np.linspace(float(self.params["approx_epsilon_start"]), float(self.params["approx_epsilon_max"]), 30):
            approx = cv2.approxPolyDP(hull, ratio * perimeter, True)
            if len(approx) == 4:
                return self._order_corners(approx.reshape(4, 2).astype(np.float32))
            if len(approx) < 4:
                break
        return self._order_corners(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32))

    def _homography_is_plausible(self, homography: np.ndarray, contour: np.ndarray, quad: np.ndarray) -> bool:
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.float32)):
            return False
        warped = cv2.perspectiveTransform(contour.reshape(-1, 1, 2).astype(np.float32), homography)
        before = float(cv2.contourArea(contour))
        after = float(cv2.contourArea(warped))
        if before <= 0 or after <= 0:
            return False
        limit = float(self.params["max_area_change_ratio"])
        return 1.0 / limit <= after / before <= limit

    def _fit_canvas(self, homography: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Translate the warped footprint so it sits inside the output with a margin."""
        warped = cv2.perspectiveTransform(contour.reshape(-1, 1, 2).astype(np.float32), homography).reshape(-1, 2)
        margin = float(self.params["output_margin_px"])
        minimum = warped.min(axis=0)
        maximum = warped.max(axis=0)
        shift = np.array([[1, 0, margin - minimum[0]], [0, 1, margin - minimum[1]], [0, 0, 1]], dtype=np.float64)
        size = (int(np.ceil(maximum[0] - minimum[0] + 2 * margin)), int(np.ceil(maximum[1] - minimum[1] + 2 * margin)))
        return shift @ homography, size

    @staticmethod
    def _order_corners(points: np.ndarray) -> np.ndarray:
        sums = points.sum(axis=1)
        diffs = points[:, 1] - points[:, 0]
        return np.array(
            [points[np.argmin(sums)], points[np.argmin(diffs)], points[np.argmax(sums)], points[np.argmax(diffs)]],
            dtype=np.float32,
        )

    @staticmethod
    def _intersect(first: Line, second: Line) -> Optional[np.ndarray]:
        point_a, direction_a = first
        point_b, direction_b = second
        determinant = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
        if abs(float(determinant)) < 1e-8:
            return None
        delta = point_b - point_a
        step = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / determinant
        return (point_a + step * direction_a).astype(np.float32)
