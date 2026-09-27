"""Phase 3: fit rectilinear rooms and snap them to a shared grid."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon

Rect = Dict[str, int]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _as_odd(size: int) -> int:
    size = max(int(size), 1)
    return size if size % 2 else size + 1


class RectilinearFitter:
    """Turn jagged floor masks into orthogonal rectangles on one wall grid.

    A mask that fills its bounding box, aside from shallow furniture notches,
    becomes that box, so the outline no longer hugs beds and counters. A mask
    with a thick missing corner (an L-shape, or a closet joined through a wide
    doorway) is decomposed into the largest inscribed rectangles. Rectangles
    that still cover two floor materials are split on the wall-to-wall colour
    seam. Every edge is then snapped to a shared set of grid lines so adjacent
    rooms share one wall.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config["rectilinear"]

    def fit(self, masks: Sequence[np.ndarray], flattened: Optional[np.ndarray] = None) -> List[Polygon]:
        """Run the full fit, returning one shapely rectangle per room."""
        rects = self.fit_bounding_rectangles(list(masks))
        if flattened is not None:
            rects = self.split_open_plan_areas(rects, flattened)
        tolerance = int(self.params["grid_snap_tolerance_px"])
        rects = self.snap_to_global_grid(rects, tolerance)
        return [self.as_polygon(rect) for rect in rects if self._area(rect) > 0]

    def fit_bounding_rectangles(self, room_masks: List[np.ndarray]) -> List[Rect]:
        """Fit each mask with an axis-aligned box, or a union of inscribed boxes.

        The canvas is already orthographic, so the architectural axes are the
        image axes. ``cv2.minAreaRect`` is computed only to confirm the mask's
        principal direction; the stored rectangle is always the axis-aligned
        box (``cv2.boundingRect``), which is that minimum-area rectangle forced
        onto 0/90 degrees.
        """
        fitted: list[Rect] = []
        min_area = int(self.params["min_rect_area_px"])
        for index, mask in enumerate(room_masks):
            solid = self._solid_mask(mask)
            if int(np.count_nonzero(solid)) < min_area:
                continue
            fitted.extend(self._decompose(solid, index, depth=0))
        return fitted

    def split_open_plan_areas(self, room_rects: List[Rect], flattened_image: np.ndarray) -> List[Rect]:
        """Split oversized rectangles along a wall-to-wall floor-material seam.

        Furniture and rugs are removed with a heavy blur. A split is kept only
        when the colour step runs across most of the rectangle and the two
        sides differ in LAB, which is a wood-to-carpet or tile-to-carpet change
        rather than a sofa edge.
        """
        if not room_rects:
            return []
        total = sum(self._area(rect) for rect in room_rects)
        limit = float(self.params["open_plan_area_ratio"]) * total
        lab = cv2.cvtColor(flattened_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        blur = _as_odd(int(self.params["seam_blur_px"]))
        lab = cv2.GaussianBlur(lab, (blur, blur), 0)

        split: list[Rect] = []
        for rect in room_rects:
            if self._area(rect) < limit:
                split.append(dict(rect))
                continue
            split.extend(self._split_rect(dict(rect), lab, limit))
        return split

    def snap_to_global_grid(self, room_rects: List[Rect], snap_tolerance_px: int = 15) -> List[Rect]:
        """Snap every edge to a clustered grid line, then remove overlaps.

        Coordinates within ``snap_tolerance_px`` of each other become one line,
        the median of the cluster. Adjacent rooms whose walls fell in the same
        cluster then share that line.
        """
        if not room_rects:
            return []
        xs = [int(rect["x"]) for rect in room_rects] + [int(rect["x"] + rect["w"]) for rect in room_rects]
        ys = [int(rect["y"]) for rect in room_rects] + [int(rect["y"] + rect["h"]) for rect in room_rects]
        x_lines = self._cluster_lines(xs, int(snap_tolerance_px))
        y_lines = self._cluster_lines(ys, int(snap_tolerance_px))
        min_side = int(self.params["min_side_px"])

        snapped: list[Rect] = []
        for rect in room_rects:
            x1 = self._nearest_line(int(rect["x"]), x_lines)
            x2 = self._nearest_line(int(rect["x"] + rect["w"]), x_lines)
            y1 = self._nearest_line(int(rect["y"]), y_lines)
            y2 = self._nearest_line(int(rect["y"] + rect["h"]), y_lines)
            if x2 - x1 < min_side or y2 - y1 < min_side:
                snapped.append(dict(rect))
                continue
            updated = dict(rect)
            updated.update(x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1))
            snapped.append(updated)
        return self._resolve_overlaps(snapped, min_side)

    @staticmethod
    def as_polygon(rect: Rect) -> Polygon:
        x, y, w, h = (int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"]))
        return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

    def _decompose(self, mask: np.ndarray, source_index: int, depth: int) -> List[Rect]:
        min_area = int(self.params["min_rect_area_px"])
        area = int(np.count_nonzero(mask))
        if area < min_area:
            return []
        x, y, width, height = self._orthogonal_box(mask)
        bbox = self._rect(x, y, width, height, source_index)
        # Small rooms and near-rectangular rooms keep one box, which overwrites
        # furniture notches. Only a large mask with a thick missing corner is
        # split, and only one level deep so the pieces do not shatter.
        if (
            area < int(self.params["decompose_min_area_px"])
            or self._mostly_rectangular(mask)
            or depth >= 1
        ):
            return [bbox]

        inscribed = self._largest_inscribed_rect(mask)
        if inscribed is None or inscribed[2] * inscribed[3] < min_area:
            return [bbox]
        ix, iy, iwidth, iheight = inscribed
        remaining = mask.copy()
        remaining[iy : iy + iheight, ix : ix + iwidth] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(remaining, connectivity=4)
        arm_share = float(self.params["arm_min_share"])
        arms = sorted(
            (
                np.where(labels == index, 255, 0).astype(np.uint8)
                for index in range(1, count)
                if int(stats[index, cv2.CC_STAT_AREA]) >= max(min_area, int(arm_share * area))
            ),
            key=lambda piece: int(np.count_nonzero(piece)),
            reverse=True,
        )
        if not arms:
            return [bbox]

        blocked = np.zeros_like(mask)
        for arm in arms:
            blocked[arm > 0] = 255
        body = mask.copy()
        body[blocked > 0] = 0
        rects = [self._rect(*self._grow_rect(inscribed, body), source_index)]
        limit = int(self.params["max_rects_per_mask"]) - 1
        for arm in arms[:limit]:
            arm_box = self._largest_inscribed_rect(arm)
            if arm_box is None:
                continue
            grown = self._grow_rect(arm_box, arm)
            if grown[2] * grown[3] >= min_area:
                rects.append(self._rect(*grown, source_index))
        # The inscribed boxes can leave a third wing (the open living area
        # between a sofa column and a kitchen). Give each large leftover its
        # own box instead of dropping it.
        covered = np.zeros_like(mask)
        for rect in rects:
            covered[rect["y"] : rect["y"] + rect["h"], rect["x"] : rect["x"] + rect["w"]] = 255
        leftover = cv2.bitwise_and(mask, cv2.bitwise_not(covered))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(leftover, connectivity=4)
        pieces = sorted(
            (int(stats[index, cv2.CC_STAT_AREA]), index) for index in range(1, count)
        )
        for piece_area, index in pieces[-limit:]:
            if piece_area < min_area:
                continue
            piece = np.where(labels == index, 255, 0).astype(np.uint8)
            px, py, pwidth, pheight = cv2.boundingRect(piece)
            fill = piece_area / float(max(pwidth * pheight, 1))
            if fill >= float(self.params["leftover_fill_min"]):
                # A ring of floor around a table still belongs to one room.
                rects.append(self._rect(px, py, pwidth, pheight, source_index))
                continue
            piece_box = self._largest_inscribed_rect(piece)
            if piece_box is None:
                continue
            rects.append(self._rect(*self._grow_rect(piece_box, piece), source_index))
        return rects or [bbox]

    def _grow_rect(
        self,
        rect: Tuple[int, int, int, int],
        allowed: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """Grow an inscribed box into notches, stopping where the mask gives out.

        A side grows while the new row or column is still mostly inside the
        allowed mask. An L-shaped arm is not in ``allowed`` for the main box,
        so the box stops at the arm instead of swallowing it.
        """
        x, y, width, height = rect
        image_h, image_w = allowed.shape[:2]
        coverage = float(self.params["grow_coverage"])

        def row_coverage(row: int) -> float:
            if row < 0 or row >= image_h or width <= 0:
                return 0.0
            return float(np.count_nonzero(allowed[row, x : x + width])) / float(width)

        def col_coverage(col: int) -> float:
            if col < 0 or col >= image_w or height <= 0:
                return 0.0
            return float(np.count_nonzero(allowed[y : y + height, col])) / float(height)

        grew = True
        while grew:
            grew = False
            if row_coverage(y - 1) >= coverage:
                y -= 1
                height += 1
                grew = True
            if row_coverage(y + height) >= coverage:
                height += 1
                grew = True
            if col_coverage(x - 1) >= coverage:
                x -= 1
                width += 1
                grew = True
            if col_coverage(x + width) >= coverage:
                width += 1
                grew = True
        return x, y, width, height

    def _mostly_rectangular(self, mask: np.ndarray) -> bool:
        """True when the bounding-box gaps are shallow furniture notches."""
        x, y, width, height = cv2.boundingRect(mask)
        if width == 0 or height == 0:
            return False
        roi = mask[y : y + height, x : x + width] > 0
        if np.count_nonzero(roi) / float(width * height) >= float(self.params["rect_fill_min"]):
            return True
        missing = np.where(roi, 0, 255).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, connectivity=4)
        max_inradius = float(self.params["notch_max_inradius_px"])
        corner_share = float(self.params["corner_min_share"])
        distance = cv2.distanceTransform(missing, cv2.DIST_L2, 3)
        box_area = float(width * height)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < corner_share * box_area:
                continue
            inradius = float(distance[labels == index].max())
            if inradius > max_inradius:
                return False
        return True

    def _split_rect(self, rect: Rect, lab: np.ndarray, area_limit: float) -> List[Rect]:
        seam = self._dominant_seam(rect, lab)
        if seam is None:
            return [rect]
        axis, position = seam
        first, second = self._cut(rect, axis, position)
        min_area = int(self.params["min_rect_area_px"])
        if self._area(first) < min_area or self._area(second) < min_area:
            return [rect]
        parts: list[Rect] = []
        for part in (first, second):
            if self._area(part) >= area_limit:
                parts.extend(self._split_rect(part, lab, area_limit))
            else:
                parts.append(part)
        return parts

    def _dominant_seam(self, rect: Rect, lab: np.ndarray) -> Optional[Tuple[str, int]]:
        x, y, width, height = (int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"]))
        height_img, width_img = lab.shape[:2]
        x2 = min(width_img, x + width)
        y2 = min(height_img, y + height)
        x = max(0, x)
        y = max(0, y)
        if x2 - x < 30 or y2 - y < 30:
            return None
        roi = lab[y:y2, x:x2]
        margin_ratio = float(self.params["seam_margin_ratio"])
        best: Optional[Tuple[float, str, int]] = None
        for axis in ("x", "y"):
            profile = roi.mean(axis=0) if axis == "x" else roi.mean(axis=1)
            delta = np.linalg.norm(np.diff(profile, axis=0), axis=1)
            if delta.size < 5:
                continue
            inset = int(margin_ratio * delta.size)
            if inset < 1 or inset >= delta.size // 2:
                continue
            window = delta[inset:-inset]
            if window.size == 0:
                continue
            peak = int(np.argmax(window)) + inset
            prominence = float(window.max()) / max(float(np.median(delta)), 1.0)
            if prominence < float(self.params["seam_prominence_min"]):
                continue
            score = self._seam_score(roi, axis, peak)
            if score is None:
                continue
            if best is None or score > best[0]:
                origin = x if axis == "x" else y
                best = (score, axis, origin + peak)
        if best is None:
            return None
        return best[1], best[2]

    def _seam_score(self, roi: np.ndarray, axis: str, index: int) -> Optional[float]:
        """How wall-to-wall and how strong a colour step is. ``None`` rejects it."""
        span_min = float(self.params["seam_span_min"])
        delta_min = float(self.params["seam_delta_e_min"])
        if axis == "x":
            if index < 1 or index >= roi.shape[1] - 1:
                return None
            step = np.linalg.norm(roi[:, index + 1] - roi[:, index - 1], axis=1)
            side_a = roi[:, :index].reshape(-1, 3)
            side_b = roi[:, index:].reshape(-1, 3)
        else:
            if index < 1 or index >= roi.shape[0] - 1:
                return None
            step = np.linalg.norm(roi[index + 1] - roi[index - 1], axis=1)
            side_a = roi[:index].reshape(-1, 3)
            side_b = roi[index:].reshape(-1, 3)
        if side_a.size == 0 or side_b.size == 0:
            return None
        # A real seam is strong in most segments of the cut. A sofa is strong in one.
        bins = [float(part.mean()) for part in np.array_split(step, 8) if part.size]
        if not bins or max(bins) < 1.5:
            return None
        span = float(sum(value >= 0.45 * max(bins) for value in bins)) / float(len(bins))
        if span < span_min:
            return None
        delta_e = float(np.linalg.norm(side_a.mean(axis=0) - side_b.mean(axis=0)))
        if delta_e < delta_min:
            return None
        return delta_e * span

    @staticmethod
    def _cut(rect: Rect, axis: str, position: int) -> Tuple[Rect, Rect]:
        first, second = dict(rect), dict(rect)
        if axis == "x":
            first["w"] = int(position - rect["x"])
            second["x"] = int(position)
            second["w"] = int(rect["x"] + rect["w"] - position)
        else:
            first["h"] = int(position - rect["y"])
            second["y"] = int(position)
            second["h"] = int(rect["y"] + rect["h"] - position)
        return first, second

    @staticmethod
    def _largest_inscribed_rect(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Largest axis-aligned rectangle of foreground, via the histogram stack."""
        x0, y0, width, height = cv2.boundingRect(mask)
        if width == 0 or height == 0:
            return None
        binary = mask[y0 : y0 + height, x0 : x0 + width] > 0
        heights = np.zeros(width, dtype=np.int32)
        best_area = 0
        best = (0, 0, 0, 0)
        for row in range(height):
            heights = np.where(binary[row], heights + 1, 0)
            stack = [-1]
            for col in range(width + 1):
                current = int(heights[col]) if col < width else 0
                while stack[-1] != -1 and int(heights[stack[-1]]) > current:
                    bar = int(heights[stack.pop()])
                    left = stack[-1]
                    run = col - left - 1
                    area = bar * run
                    if area > best_area:
                        best_area = area
                        best = (x0 + left + 1, y0 + row - bar + 1, run, bar)
                stack.append(col)
        if best_area == 0:
            return None
        return best

    @staticmethod
    def _cluster_lines(values: Sequence[int], tolerance: int) -> List[int]:
        ordered = sorted(int(value) for value in values)
        if not ordered:
            return []
        clusters: list[list[int]] = [[ordered[0]]]
        for value in ordered[1:]:
            if value - clusters[-1][-1] <= tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        return [int(round(float(np.median(cluster)))) for cluster in clusters]

    @staticmethod
    def _nearest_line(value: int, lines: Sequence[int]) -> int:
        return min(lines, key=lambda line: abs(line - value))

    def _resolve_overlaps(self, rects: List[Rect], min_side: int) -> List[Rect]:
        """Pull the larger rectangle back to the smaller one's edge where they overlap."""
        resolved = [dict(rect) for rect in rects]
        for _ in range(6):
            changed = False
            for i in range(len(resolved)):
                for j in range(i + 1, len(resolved)):
                    larger, smaller = resolved[i], resolved[j]
                    if int(larger["source_index"]) == int(smaller["source_index"]):
                        continue
                    if self._area(larger) < self._area(smaller):
                        larger, smaller = smaller, larger
                    overlap = self._overlap(larger, smaller)
                    if overlap is None:
                        continue
                    ox, oy, ow, oh = overlap
                    if ow >= oh:
                        if larger["y"] <= smaller["y"]:
                            new_h = smaller["y"] - larger["y"]
                            if new_h >= min_side:
                                larger["h"] = int(new_h)
                                changed = True
                        else:
                            new_y = smaller["y"] + smaller["h"]
                            new_h = larger["y"] + larger["h"] - new_y
                            if new_h >= min_side:
                                larger["y"] = int(new_y)
                                larger["h"] = int(new_h)
                                changed = True
                    else:
                        if larger["x"] <= smaller["x"]:
                            new_w = smaller["x"] - larger["x"]
                            if new_w >= min_side:
                                larger["w"] = int(new_w)
                                changed = True
                        else:
                            new_x = smaller["x"] + smaller["w"]
                            new_w = larger["x"] + larger["w"] - new_x
                            if new_w >= min_side:
                                larger["x"] = int(new_x)
                                larger["w"] = int(new_w)
                                changed = True
            if not changed:
                break
        return [rect for rect in resolved if rect["w"] >= min_side and rect["h"] >= min_side]

    @staticmethod
    def _overlap(first: Rect, second: Rect) -> Optional[Tuple[int, int, int, int]]:
        x1 = max(int(first["x"]), int(second["x"]))
        y1 = max(int(first["y"]), int(second["y"]))
        x2 = min(int(first["x"] + first["w"]), int(second["x"] + second["w"]))
        y2 = min(int(first["y"] + first["h"]), int(second["y"] + second["h"]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _solid_mask(mask: np.ndarray) -> np.ndarray:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(binary)
        if contours:
            cv2.drawContours(solid, contours, -1, 255, thickness=cv2.FILLED)
        return solid

    @staticmethod
    def _rect(x: int, y: int, width: int, height: int, source_index: int) -> Rect:
        return {"x": int(x), "y": int(y), "w": int(width), "h": int(height), "source_index": int(source_index)}

    def _orthogonal_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Minimum-area rectangle forced onto 0 or 90 degrees.

        ``cv2.minAreaRect`` may lean a few degrees because the mask outline
        still follows furniture. When that lean is within the ortho tolerance
        the angle is snapped to the nearest axis and the box is re-read as an
        axis-aligned rectangle. A larger lean means a concave room, so the
        axis-aligned bounding box is used and decomposition handles the shape.
        """
        x, y, width, height = cv2.boundingRect(mask)
        points = cv2.findNonZero(mask)
        if points is None or len(points) < 4:
            return x, y, width, height
        center, size, angle = cv2.minAreaRect(points)
        deviation = min(abs(angle), abs(angle + 90.0))
        tolerance = float(self.config["geometry"]["ortho_snap_tolerance_deg"])
        if deviation > tolerance:
            return x, y, width, height
        snapped = 0.0 if abs(angle) <= abs(angle + 90.0) else -90.0
        box = cv2.boxPoints((center, size, snapped))
        return cv2.boundingRect(np.round(box).astype(np.int32))

    @staticmethod
    def _area(rect: Rect) -> int:
        return max(int(rect["w"]), 0) * max(int(rect["h"]), 0)
