"""Phase 1: Metric Camera Calibration & Ground Rectification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class RectificationResult:
    rectified_image: np.ndarray
    H: np.ndarray
    H_inv: np.ndarray
    vp_vertical: Tuple[float, float]
    vp_horizontal: Tuple[float, float]
    vertical_lines: List[np.ndarray] = field(default_factory=list)
    horizontal_lines: List[np.ndarray] = field(default_factory=list)
    skew_before_deg: float = 0.0
    skew_after_deg: float = 0.0
    is_orthographic: bool = False
    notes: List[str] = field(default_factory=list)


class ManhattanCalibrator:
    """Estimates vanishing points from structural lines and rectifies perspective distortion.

    Elevated 3D cutaway renders feature perspective convergence where vertical walls
    lean radially away from the principal point. By estimating the vertical vanishing point
    VP_v = (vx, vy) and horizontal vanishing point VP_h, this module computes the true
    ground-plane homography H that maps the floor into an orthographic bird's-eye view.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config.get("v4_calibration", {})
        self.min_line_len = float(self.params.get("min_line_length_px", 30))
        self.vert_tol = float(self.params.get("vert_angle_tolerance_deg", 25))
        self.horiz_tol = float(self.params.get("horiz_angle_tolerance_deg", 15))
        self.max_ortho_vy = float(self.params.get("max_orthographic_vy", 50000.0))
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def calibrate_and_rectify(self, image: np.ndarray) -> RectificationResult:
        """Estimate vanishing points, compute metric homography, and rectify the image."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected, _, _, _ = self.lsd.detect(gray)
        lines = detected.reshape(-1, 4) if detected is not None else np.empty((0, 4), dtype=np.float32)

        if len(lines) == 0:
            eye = np.eye(3, dtype=np.float64)
            return RectificationResult(
                rectified_image=image.copy(),
                H=eye,
                H_inv=eye,
                vp_vertical=(w / 2.0, -1e6),
                vp_horizontal=(1e6, h / 2.0),
                is_orthographic=True,
                notes=["No line segments detected."],
            )

        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.hypot(dx, dy)
        valid = lengths >= self.min_line_len
        lines = lines[valid]
        dx = dx[valid]
        dy = dy[valid]
        lengths = lengths[valid]

        angles = np.rad2deg(np.arctan2(dy, dx))
        v_idx = np.abs(np.abs(angles) - 90.0) <= self.vert_tol
        h_idx = np.abs(angles) <= self.horiz_tol

        v_lines = lines[v_idx]
        h_lines = lines[h_idx]

        vp_v, inlier_v = self._fit_vanishing_point_ransac(v_lines, is_vertical=True)
        vp_h, inlier_h = self._fit_vanishing_point_ransac(h_lines, is_vertical=False)

        vx, vy = vp_v
        vhx, vhy = vp_h
        notes = []

        is_orthographic = abs(vy) >= self.max_ortho_vy or vy > 0
        if is_orthographic:
            notes.append(f"Image is nearly orthographic (|vy|={abs(vy):.0f} >= {self.max_ortho_vy:.0f}).")
            H = np.eye(3, dtype=np.float64)
            rectified = image.copy()
            skew_before, skew_after = 0.0, 0.0
        else:
            notes.append(f"Perspective convergence detected: VP_v=({vx:.1f}, {vy:.1f}), VP_h=({vhx:.1f}, {vhy:.1f}).")
            cy = h / 2.0
            scale = 1.0 - (cy / vy)

            T = np.array([[1.0, 0.0, -vx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            H_aff = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, -1.0 / vy, 1.0]], dtype=np.float64)
            T_inv = np.array([[1.0, 0.0, vx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            H_raw = T_inv @ H_aff @ T

            corners = np.array([[0.0, 0.0, 1.0], [float(w), 0.0, 1.0], [float(w), float(h), 1.0], [0.0, float(h), 1.0]], dtype=np.float64).T
            corners_w = H_raw @ corners
            corners_w /= corners_w[2]

            min_x, max_x = float(corners_w[0].min()), float(corners_w[0].max())
            min_y, max_y = float(corners_w[1].min()), float(corners_w[1].max())

            out_w = int(np.ceil(max_x - min_x))
            out_h = int(np.ceil(max_y - min_y))

            T_fit = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
            H = T_fit @ H_raw

            border_val = tuple(self.params.get("border_value", [255, 255, 255]))
            rectified = cv2.warpPerspective(
                image,
                H,
                (out_w, out_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_val,
            )

            skew_before = self._measure_skew(v_lines, w)
            warped_v_lines = self._transform_lines(v_lines, H)
            skew_after = self._measure_skew(warped_v_lines, out_w)

        H_inv = np.linalg.inv(H)
        return RectificationResult(
            rectified_image=rectified,
            H=H,
            H_inv=H_inv,
            vp_vertical=(float(vx), float(vy)),
            vp_horizontal=(float(vhx), float(vhy)),
            vertical_lines=list(v_lines),
            horizontal_lines=list(h_lines),
            skew_before_deg=float(skew_before),
            skew_after_deg=float(skew_after),
            is_orthographic=is_orthographic,
            notes=notes,
        )

    def _fit_vanishing_point_ransac(
        self,
        lines: np.ndarray,
        is_vertical: bool,
        max_iters: int = 150,
        inlier_angle_deg: float = 2.0,
    ) -> Tuple[Tuple[float, float], np.ndarray]:
        if len(lines) < 2:
            default_vp = (0.0, -1e6) if is_vertical else (1e6, 0.0)
            return default_vp, np.ones(len(lines), dtype=bool)

        A, weights = [], []
        for x1, y1, x2, y2 in lines:
            dx_i = x2 - x1
            dy_i = y2 - y1
            length = float(np.hypot(dx_i, dy_i))
            if length < 1e-4:
                length = 1e-4
            a = -dy_i / length
            b = dx_i / length
            c = -(a * x1 + b * y1)
            A.append([a, b, c])
            weights.append(length)

        A = np.array(A, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        W = np.diag(weights)

        best_inliers = np.ones(len(lines), dtype=bool)
        best_count = -1
        rng = np.random.default_rng(42)

        for _ in range(max_iters):
            indices = rng.choice(len(lines), size=min(4, len(lines)), replace=False)
            sub_A = A[indices]
            sub_W = W[indices, :][:, indices]
            _, _, Vt = np.linalg.svd(sub_W @ sub_A)
            candidate = Vt[-1]
            if abs(candidate[2]) < 1e-8:
                continue
            cand_vp = candidate[:2] / candidate[2]

            # Inlier metric: angular error of each line to candidate VP
            mid_x = (lines[:, 0] + lines[:, 2]) / 2.0
            mid_y = (lines[:, 1] + lines[:, 3]) / 2.0
            ray_dx = mid_x - cand_vp[0]
            ray_dy = mid_y - cand_vp[1]
            ray_angle = np.rad2deg(np.arctan2(ray_dy, ray_dx))

            line_dx = lines[:, 2] - lines[:, 0]
            line_dy = lines[:, 3] - lines[:, 1]
            line_angle = np.rad2deg(np.arctan2(line_dy, line_dx))

            diff = np.abs(ray_angle - line_angle) % 180.0
            angular_err = np.minimum(diff, 180.0 - diff)
            inliers = angular_err <= inlier_angle_deg
            count = int(np.sum(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_count >= 2:
            final_A = A[best_inliers]
            final_W = W[best_inliers, :][:, best_inliers]
            _, _, Vt = np.linalg.svd(final_W @ final_A)
            vp = Vt[-1]
            if abs(vp[2]) > 1e-8:
                return (float(vp[0] / vp[2]), float(vp[1] / vp[2])), best_inliers

        _, _, Vt = np.linalg.svd(W @ A)
        vp = Vt[-1]
        z = vp[2] if abs(vp[2]) > 1e-8 else 1e-8
        return (float(vp[0] / z), float(vp[1] / z)), best_inliers

    @staticmethod
    def _transform_lines(lines: np.ndarray, H: np.ndarray) -> np.ndarray:
        if len(lines) == 0:
            return np.empty((0, 4), dtype=np.float32)
        n = len(lines)
        p1 = np.vstack([lines[:, 0], lines[:, 1], np.ones(n, dtype=np.float64)])
        p2 = np.vstack([lines[:, 2], lines[:, 3], np.ones(n, dtype=np.float64)])
        p1_w = H @ p1
        p1_w /= p1_w[2]
        p2_w = H @ p2
        p2_w /= p2_w[2]
        return np.column_stack([p1_w[0], p1_w[1], p2_w[0], p2_w[1]])

    @staticmethod
    def _measure_skew(lines: np.ndarray, width: float) -> float:
        """Measure left-half vs right-half difference in vertical line deviation."""
        if len(lines) < 4:
            return 0.0
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.rad2deg(np.arctan2(dy, dx))
        dev = np.where(angles > 0, angles - 90.0, angles + 90.0)

        mid_x = (lines[:, 0] + lines[:, 2]) / 2.0
        left = dev[mid_x < width * 0.45]
        right = dev[mid_x > width * 0.55]
        if len(left) < 2 or len(right) < 2:
            return 0.0
        return float(abs(np.median(left) - np.median(right)))
