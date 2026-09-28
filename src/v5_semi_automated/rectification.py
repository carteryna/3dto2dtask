"""Phase 1: Ground-Plane Rectification & Input Normalization.

Rectifies oblique 3D perspective renders and synchronously warps human semantic
guide annotations into an orthographic top-down bird's-eye view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def normalize_unit_key(stem: str) -> str:
    """Extract a canonical unit key from a filename stem."""
    cleaned = re.sub(r"seman?tic.*", "", stem, flags=re.IGNORECASE)
    tokens = re.findall(r"[a-z0-9]+", cleaned.lower())
    return "_".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "unit")


def classify_guide_format(stem: str) -> str:
    """Classify human semantic guide format from filename suffix.

    Returns:
        'center_line' | 'no_stroke' | 'stroke' | 'unknown'
    """
    s = stem.lower()
    if "center_line" in s or "central_line" in s:
        return "center_line"
    if "no_stroke" in s:
        return "no_stroke"
    if "stroke" in s:
        return "stroke"
    return "unknown"


@dataclass
class SemanticGuide:
    """Container for a human semantic guide asset."""
    unit_key: str
    stem: str
    format_type: str  # 'center_line' | 'no_stroke' | 'stroke'
    path: Path
    image: np.ndarray  # 4-channel BGRA uint8


@dataclass
class RectificationResult:
    """Result of Phase 1 ground-plane rectification."""
    unit_key: str
    rectified_render: np.ndarray
    rectified_guides: Dict[str, np.ndarray]  # format_type -> rectified BGRA image
    H: np.ndarray
    H_inv: np.ndarray
    vp_vertical: Tuple[float, float]
    is_orthographic: bool
    skew_before_deg: float
    skew_after_deg: float
    vertical_lines: List[np.ndarray] = field(default_factory=list)
    warped_vertical_lines: List[np.ndarray] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class SemiAutomatedRectifier:
    """Computes vertical vanishing-point pitch rectification.

    Synchronously transforms both the perspective 3D render and human semantic
    guides into a metric orthographic coordinate system, eliminating radial wall
    lean while preserving ground-plane aspect ratio.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config.get("v5_rectification", {})
        self.min_line_len = float(self.params.get("min_line_length_px", 25.0))
        self.vert_tol = float(self.params.get("vert_angle_tolerance_deg", 20.0))
        self.max_ortho_vy = float(self.params.get("max_orthographic_vy", 50000.0))
        self.border_val = tuple(self.params.get("border_value", [255, 255, 255]))
        self.guide_dir = Path(self.params.get("guide_dir", "human_semantic_guide"))
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def load_guides_for_unit(self, unit_key_or_path: str | Path) -> Dict[str, SemanticGuide]:
        """Discover and load all semantic guide formats matching a unit."""
        if isinstance(unit_key_or_path, Path):
            target_key = normalize_unit_key(unit_key_or_path.stem)
        else:
            target_key = normalize_unit_key(str(unit_key_or_path))

        guides: Dict[str, SemanticGuide] = {}
        if not self.guide_dir.is_dir():
            return guides

        for guide_path in sorted(self.guide_dir.glob("*.png")):
            g_key = normalize_unit_key(guide_path.stem)
            if g_key == target_key:
                fmt = classify_guide_format(guide_path.stem)
                img = cv2.imread(str(guide_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    guides[fmt] = SemanticGuide(
                        unit_key=target_key,
                        stem=guide_path.stem,
                        format_type=fmt,
                        path=guide_path,
                        image=img,
                    )
        return guides

    def extract_red_centerline_mask(self, guide_bgra: np.ndarray) -> np.ndarray:
        """Extract binary mask of red centerlines from BGRA guide image."""
        if guide_bgra.shape[2] < 4:
            return np.zeros(guide_bgra.shape[:2], dtype=np.uint8)
        b, g, r, a = cv2.split(guide_bgra)
        is_red = ((r.astype(int) - g.astype(int) > 50) & (r > 150) & (a > 50)).astype(np.uint8) * 255
        return is_red

    def detect_vertical_lines(
        self,
        render_image: np.ndarray,
        guides: Dict[str, SemanticGuide],
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract vertical line segments using guides (primary) or render (fallback)."""
        notes: List[str] = []

        # 1. Prefer red centerline if available (pure 1D wall axes)
        if "center_line" in guides:
            red_mask = self.extract_red_centerline_mask(guides["center_line"].image)
            lines, _, _, _ = self.lsd.detect(red_mask)
            if lines is not None and len(lines) > 0:
                lines = lines.reshape(-1, 4)
                v_lines = self._filter_vertical_lines(lines)
                if len(v_lines) >= 4:
                    notes.append(f"Extracted {len(v_lines)} vertical segments from red center-line guide.")
                    return v_lines, notes

        # 2. Next prefer solid wall body or stroke alpha boundary
        for fmt in ("no_stroke", "stroke"):
            if fmt in guides:
                alpha = guides[fmt].image[:, :, 3]
                wall_bin = (alpha > 50).astype(np.uint8) * 255
                lines, _, _, _ = self.lsd.detect(wall_bin)
                if lines is not None and len(lines) > 0:
                    lines = lines.reshape(-1, 4)
                    v_lines = self._filter_vertical_lines(lines)
                    if len(v_lines) >= 4:
                        notes.append(f"Extracted {len(v_lines)} vertical segments from {fmt} guide.")
                        return v_lines, notes

        # 3. Fallback to raw render image
        gray = cv2.cvtColor(render_image, cv2.COLOR_BGR2GRAY)
        lines, _, _, _ = self.lsd.detect(gray)
        if lines is not None and len(lines) > 0:
            lines = lines.reshape(-1, 4)
            v_lines = self._filter_vertical_lines(lines)
            notes.append(f"Extracted {len(v_lines)} vertical segments from raw render LSD.")
            return v_lines, notes

        notes.append("No vertical segments detected.")
        return np.empty((0, 4), dtype=np.float32), notes

    def _filter_vertical_lines(self, lines: np.ndarray) -> np.ndarray:
        """Filter lines within vertical tolerance and minimum length."""
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.hypot(dx, dy)
        angles = np.rad2deg(np.arctan2(dy, dx))
        dev = np.abs(np.abs(angles) - 90.0)
        valid = (dev <= self.vert_tol) & (lengths >= self.min_line_len)
        return lines[valid]

    def estimate_vertical_vp(
        self,
        v_lines: np.ndarray,
        max_iters: int = 150,
        inlier_angle_deg: float = 2.0,
    ) -> Tuple[Tuple[float, float], np.ndarray]:
        """Estimate vertical vanishing point via SVD line intersection with RANSAC."""
        if len(v_lines) < 2:
            return (0.0, -1e6), np.ones(len(v_lines), dtype=bool)

        # Build homogeneous line equations: a*x + b*y + c = 0
        A, weights = [], []
        for x1, y1, x2, y2 in v_lines:
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

        best_inliers = np.zeros(len(v_lines), dtype=bool)
        best_count = 0
        n = len(v_lines)

        rng = np.random.default_rng(42)
        for _ in range(min(max_iters, max(10, n * 2))):
            idx = rng.choice(n, size=2, replace=False)
            l1, l2 = A[idx[0]], A[idx[1]]
            pt = np.cross(l1, l2)
            if abs(pt[2]) < 1e-6:
                continue
            pt = pt / pt[2]
            vx, vy = pt[0], pt[1]

            # Inlier metric: angular agreement between ray to VP and line direction
            residuals = []
            for i, (x1, y1, x2, y2) in enumerate(v_lines):
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ray_dx, ray_dy = vx - mx, vy - my
                seg_dx, seg_dy = x2 - x1, y2 - y1
                cross = abs(ray_dx * seg_dy - ray_dy * seg_dx)
                dot = abs(ray_dx * seg_dx + ray_dy * seg_dy)
                ang = np.rad2deg(np.arctan2(cross, dot + 1e-8))
                residuals.append(ang)

            inliers = np.array(residuals) <= inlier_angle_deg
            count = int(np.sum(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_count < 2:
            best_inliers = np.ones(n, dtype=bool)

        # SVD refinement on inliers
        A_in = A[best_inliers]
        W_in = np.diag(weights[best_inliers])
        WA = W_in @ A_in
        _, _, Vt = np.linalg.svd(WA)
        sol = Vt[-1]
        if abs(sol[2]) < 1e-6:
            return (float(sol[0]), -1e6), best_inliers
        sol = sol / sol[2]
        return (float(sol[0]), float(sol[1])), best_inliers

    def compute_rectification_homography(
        self,
        w: int,
        h: int,
        vp_v: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, int, int, bool]:
        """Compute the 1D ground pitch homography H and target canvas dimensions."""
        vx, vy = vp_v
        is_orthographic = abs(vy) >= self.max_ortho_vy or vy > 0

        if is_orthographic:
            eye = np.eye(3, dtype=np.float64)
            return eye, eye, w, h, True

        cy = h / 2.0
        scale = 1.0 - (cy / vy)

        T = np.array([[1.0, 0.0, -vx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_aff = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, -1.0 / vy, 1.0]], dtype=np.float64)
        T_inv = np.array([[1.0, 0.0, vx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_raw = T_inv @ H_aff @ T

        # Bounding box of original canvas under H_raw
        corners = np.array(
            [[0.0, 0.0, 1.0], [float(w), 0.0, 1.0], [float(w), float(h), 1.0], [0.0, float(h), 1.0]],
            dtype=np.float64,
        ).T
        corners_w = H_raw @ corners
        corners_w /= corners_w[2]

        min_x, max_x = float(corners_w[0].min()), float(corners_w[0].max())
        min_y, max_y = float(corners_w[1].min()), float(corners_w[1].max())

        out_w = int(np.ceil(max_x - min_x))
        out_h = int(np.ceil(max_y - min_y))

        T_fit = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
        H = T_fit @ H_raw
        H_inv = np.linalg.inv(H)

        return H, H_inv, out_w, out_h, False

    def transform_lines(self, lines: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Project line segment coordinates through 3x3 homography H."""
        if len(lines) == 0:
            return np.empty((0, 4), dtype=np.float32)
        n = len(lines)
        p1 = np.column_stack([lines[:, 0], lines[:, 1], np.ones(n, dtype=np.float64)])
        p2 = np.column_stack([lines[:, 2], lines[:, 3], np.ones(n, dtype=np.float64)])

        p1_w = (H @ p1.T).T
        p2_w = (H @ p2.T).T
        p1_w /= p1_w[:, 2:3]
        p2_w /= p2_w[:, 2:3]

        return np.column_stack([p1_w[:, :2], p2_w[:, :2]]).astype(np.float32)

    def measure_vertical_skew(self, v_lines: np.ndarray) -> float:
        """Measure mean angular deviation of lines from true vertical (90 degrees)."""
        if len(v_lines) == 0:
            return 0.0
        dx = v_lines[:, 2] - v_lines[:, 0]
        dy = v_lines[:, 3] - v_lines[:, 1]
        angles = np.rad2deg(np.arctan2(dy, dx))
        dev = np.abs(np.abs(angles) - 90.0)
        return float(np.mean(dev))

    def rectify(
        self,
        render_image: np.ndarray,
        guides: Optional[Dict[str, SemanticGuide]] = None,
        unit_key: str = "",
    ) -> RectificationResult:
        """Execute Phase 1 rectification on render and all loaded semantic guides."""
        h, w = render_image.shape[:2]
        if guides is None:
            guides = self.load_guides_for_unit(unit_key)

        notes: List[str] = []
        v_lines, det_notes = self.detect_vertical_lines(render_image, guides)
        notes.extend(det_notes)

        vp_v, _ = self.estimate_vertical_vp(v_lines)
        vx, vy = vp_v

        H, H_inv, out_w, out_h, is_ortho = self.compute_rectification_homography(w, h, vp_v)
        notes.append(f"Computed homography: VP_v=({vx:.1f}, {vy:.1f}), is_orthographic={is_ortho}, out_size=({out_w}, {out_h}).")

        # Synchronous warping
        rect_render = cv2.warpPerspective(
            render_image,
            H,
            (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_val,
        )

        rect_guides: Dict[str, np.ndarray] = {}
        for fmt, guide_asset in guides.items():
            rect_guide = cv2.warpPerspective(
                guide_asset.image,
                H,
                (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            rect_guides[fmt] = rect_guide

        # Measure skew reduction
        skew_before = self.measure_vertical_skew(v_lines)
        warped_v_lines = self.transform_lines(v_lines, H)
        skew_after = self.measure_vertical_skew(warped_v_lines)
        notes.append(f"Vertical skew reduction: {skew_before:.2f}° -> {skew_after:.2f}°.")

        return RectificationResult(
            unit_key=unit_key,
            rectified_render=rect_render,
            rectified_guides=rect_guides,
            H=H,
            H_inv=H_inv,
            vp_vertical=vp_v,
            is_orthographic=is_ortho,
            skew_before_deg=skew_before,
            skew_after_deg=skew_after,
            vertical_lines=list(v_lines),
            warped_vertical_lines=list(warped_v_lines),
            notes=notes,
        )
