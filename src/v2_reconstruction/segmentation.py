"""Phase 2: extract walkable floor by targeting floor-wall base seams."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_odd(size: int) -> int:
    size = max(int(size), 1)
    return size if size % 2 else size + 1


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    size = _as_odd(length)
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    theta = np.deg2rad(angle_deg)
    dx = int(round(center * np.cos(theta)))
    dy = int(round(center * np.sin(theta)))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness=1)
    return kernel


@dataclass
class SegmentationDebug:
    footprint: np.ndarray
    ridges: np.ndarray
    barrier: np.ndarray
    cells: List[np.ndarray]
    seam_edges: np.ndarray
    sources: List[str] = field(default_factory=list)
    proposal_count: int = 0
    sam_available: bool = True
    notes: List[str] = field(default_factory=list)


class FloorSeamSegmenter:
    """Segment walkable floor on the flattened render, bounded by wall bases.

    Wall tops cannot be separated from floors by colour: white bathroom tile
    and grey balcony decking share the wall-top palette. They differ in width,
    so a white top-hat keeps only bright ridges narrower than
    ``ridge_max_width_px``. After Phase 1 the walls are axis-aligned, so only
    straight horizontal or vertical runs are kept. Doorways are bridged along
    the wall axes, and only barrier pieces attached to the outer shell survive,
    which drops bright furniture that floats inside a room. The barrier
    partitions the footprint into cells, and cells joined by an unbridged
    doorway are split at the neck.

    A cell still includes the visible inner wall faces between the wall top
    and the floor. FastSAM everything-mode proposals follow floor material
    instead, so each room keeps the union of large proposals lying inside its
    cell. Their outer edge is the seam where the wall base meets the floor.
    Cells where SAM covers too little fall back to the wall-top cell.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.pipeline = self.config["pipeline"]
        self.preprocessing = self.config["preprocessing"]
        self.sam = self.config["segmentation"]
        self.params = self.config["floor_segmentation"]
        self._model = None
        self.debug: SegmentationDebug | None = None

    def segment(self, flattened: np.ndarray) -> List[np.ndarray]:
        """Return one binary floor mask (uint8, 0/255) per room."""
        smoothed = cv2.bilateralFilter(
            flattened,
            d=int(self.preprocessing["bilateral_d"]),
            sigmaColor=float(self.preprocessing["bilateral_sigma_color"]),
            sigmaSpace=float(self.preprocessing["bilateral_sigma_space"]),
        )
        footprint = self.detect_footprint(smoothed)
        ridges = self.detect_wall_ridges(smoothed, footprint)
        barrier = self.build_barrier(ridges, footprint)
        cells = self.extract_cells(barrier, footprint)
        seam_edges = self.detect_seam_edges(smoothed, footprint)

        notes: list[str] = []
        sam_available = True
        try:
            proposals = self.propose_masks(flattened)
        except Exception as exc:
            proposals = []
            sam_available = False
            notes.append(f"FastSAM unavailable ({exc}); floor masks use wall-top cells.")

        floors: list[np.ndarray] = []
        sources: list[str] = []
        for cell in cells:
            floor = self.select_floor(cell, proposals)
            if floor is None:
                floors.append(cell)
                sources.append("cell")
            else:
                floors.append(floor)
                sources.append("sam")

        self.debug = SegmentationDebug(
            footprint=footprint,
            ridges=ridges,
            barrier=barrier,
            cells=cells,
            seam_edges=seam_edges,
            sources=sources,
            proposal_count=len(proposals),
            sam_available=sam_available,
            notes=notes,
        )
        return floors

    def detect_footprint(self, image: np.ndarray) -> np.ndarray:
        """Filled apartment silhouette on the flattened canvas."""
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
        _, labels = cv2.connectedComponents(canvas, connectivity=8)
        border_ids = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        border_ids = border_ids[border_ids != 0]
        silhouette = np.where(np.isin(labels, border_ids), 0, 255).astype(np.uint8)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        footprint = np.zeros_like(silhouette)
        if contours:
            cv2.drawContours(footprint, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
        else:
            footprint[:] = 255
        return footprint

    def detect_wall_ridges(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Flat wall-coloured structure narrower than a floor, in straight axis-aligned runs.

        Two cues propose wall pixels, and both require a flat surface (low
        local lightness deviation), because wall tops and faces are untextured
        while bedding, curtains and decor are shaded:

        - Shell colour: pixels close in LAB to the median colour of the
          footprint's outer band, which is dominated by exterior wall tops.
          This adapts to cream, grey or white walls per render.
        - Brightness top-hat: pixels brighter than their surroundings in a
          structure narrower than ``ridge_max_width_px``.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lightness = lab[:, :, 0]
        window = _as_odd(int(self.params["flat_window_px"]))
        mean = cv2.blur(lightness, (window, window))
        deviation = np.sqrt(np.maximum(cv2.blur(lightness * lightness, (window, window)) - mean * mean, 0.0))
        flat = deviation <= float(self.params["flat_std_max"])

        shell = (footprint > 0) & (cv2.erode(footprint, np.ones((15, 15), np.uint8)) == 0) & flat
        if np.count_nonzero(shell) >= 50:
            reference = np.median(lab[shell], axis=0)
            ab_distance = np.maximum(np.abs(lab[:, :, 1] - reference[1]), np.abs(lab[:, :, 2] - reference[2]))
            shell_colour = (ab_distance <= float(self.params["wall_ab_tolerance"])) & (
                np.abs(lightness - reference[0]) <= float(self.params["wall_lightness_tolerance"])
            )
        else:
            shell_colour = np.zeros(lightness.shape, dtype=bool)

        masked = lightness.astype(np.uint8)
        masked[footprint == 0] = 0
        width = _as_odd(int(self.params["ridge_max_width_px"]))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        tophat = cv2.morphologyEx(masked, cv2.MORPH_TOPHAT, disk)
        bright_ridge = (tophat >= int(self.params["ridge_tophat_min"])) & (
            masked >= int(self.params["ridge_lightness_min"])
        )

        walls = np.where((shell_colour | bright_ridge) & flat & (footprint > 0), 255, 0).astype(np.uint8)
        walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        thick = cv2.morphologyEx(walls, cv2.MORPH_OPEN, disk)
        walls = cv2.subtract(walls, cv2.dilate(thick, np.ones((5, 5), np.uint8)))

        run = int(self.params["ridge_min_run_px"])
        straight = cv2.bitwise_or(
            cv2.morphologyEx(walls, cv2.MORPH_OPEN, _line_kernel(run, 0.0)),
            cv2.morphologyEx(walls, cv2.MORPH_OPEN, _line_kernel(run, 90.0)),
        )
        return self._keep_structural(straight, footprint)

    def build_barrier(self, ridges: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Bridge doorways along the wall axes, close door leaves, and seal the shell."""
        gap = int(self.params["door_gap_px"])
        barrier = ridges.copy()
        for angle in (0.0, 90.0):
            barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, _line_kernel(gap, angle))
        leaf = _as_odd(int(self.preprocessing["door_closing_kernel_size"]))
        for angle in (45.0, 135.0):
            barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, _line_kernel(leaf, angle))
        shell = cv2.subtract(footprint, cv2.erode(footprint, np.ones((5, 5), np.uint8)))
        barrier = cv2.bitwise_or(barrier, shell)
        barrier[footprint == 0] = 0
        count, labels = cv2.connectedComponents(barrier, connectivity=8)
        if count > 1:
            attached = np.unique(labels[shell > 0])
            attached = attached[attached != 0]
            barrier = np.where(np.isin(labels, attached), 255, 0).astype(np.uint8)
        return barrier

    def extract_cells(self, barrier: np.ndarray, footprint: np.ndarray) -> List[np.ndarray]:
        """Floor regions between barriers, split at doorway necks, with furniture holes filled."""
        free = cv2.bitwise_and(footprint, cv2.bitwise_not(barrier))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        min_area = int(self.sam["min_room_area_px"])
        regions = [
            np.where(labels == index, 255, 0).astype(np.uint8)
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
        ]
        regions = self._absorb_enclosed(regions)
        occupied = np.zeros_like(free)
        for region in regions:
            occupied = cv2.bitwise_or(occupied, region)
        min_inradius = float(self.params["cell_min_inradius_px"])
        cells: list[np.ndarray] = []
        for region in regions:
            filled = self._fill_enclosed_holes(region, cv2.subtract(occupied, region))
            for part in self.split_at_necks(filled):
                if int(np.count_nonzero(part)) < min_area:
                    continue
                if float(cv2.distanceTransform(part, cv2.DIST_L2, 5).max()) < min_inradius:
                    continue
                cells.append(part)
        return cells

    def _absorb_enclosed(self, regions: List[np.ndarray]) -> List[np.ndarray]:
        """Merge small regions sitting inside another region's outline into it.

        A bright rug or table ring can enclose a patch of floor. That patch is
        not a room, so it and the ring around it are returned to the room that
        surrounds it. Enclosed regions larger than ``enclosed_max_share`` of
        the parent (such as a walk-in closet) stay separate.
        """
        max_share = float(self.params["enclosed_max_share"])
        outlines = []
        for region in regions:
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outline = np.zeros_like(region)
            cv2.drawContours(outline, contours, -1, 255, thickness=cv2.FILLED)
            outlines.append(outline)
        areas = [int(np.count_nonzero(region)) for region in regions]
        parent_of: dict[int, int] = {}
        for child, region in enumerate(regions):
            for parent, outline in enumerate(outlines):
                if parent == child or areas[child] >= max_share * areas[parent]:
                    continue
                if np.count_nonzero((region > 0) & (outline > 0)) >= 0.95 * areas[child]:
                    parent_of[child] = parent
                    break
        merged = [region.copy() for region in regions]
        for child, parent in parent_of.items():
            hole_count, hole_labels = cv2.connectedComponents(
                cv2.subtract(outlines[parent], regions[parent]), connectivity=8
            )
            for index in range(1, hole_count):
                hole = hole_labels == index
                if np.any(hole & (regions[child] > 0)):
                    merged[parent][hole] = 255
        return [region for index, region in enumerate(merged) if index not in parent_of]

    def split_at_necks(self, region: np.ndarray) -> List[np.ndarray]:
        """Separate rooms joined by an unbridged doorway.

        Opening with a disk of ``neck_width_px`` erases passages narrower than
        a doorway. Every surviving core seeds a watershed on the distance map,
        which splits the region along the narrowest line between cores.
        """
        width = _as_odd(int(self.params["neck_width_px"]))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        cores = cv2.morphologyEx(region, cv2.MORPH_OPEN, disk)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=4)
        min_core = int(self.params["neck_min_core_px"])
        core_ids = [index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_core]
        if len(core_ids) <= 1:
            return [region]

        markers = np.zeros(region.shape, dtype=np.int32)
        outside = len(core_ids) + 1
        markers[region == 0] = outside
        for marker, index in enumerate(core_ids, start=1):
            markers[labels == index] = marker
        distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        relief = cv2.normalize(-distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        labels_ws = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)
        return [
            np.where((labels_ws == marker) & (region > 0), 255, 0).astype(np.uint8)
            for marker in range(1, len(core_ids) + 1)
        ]

    def detect_seam_edges(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Strong colour gradients inside the footprint, used to score seam alignment."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        magnitude = np.zeros(lab.shape[:2], dtype=np.float32)
        for channel in range(3):
            gx = cv2.Sobel(lab[:, :, channel], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lab[:, :, channel], cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.maximum(magnitude, cv2.magnitude(gx, gy))
        inside = magnitude[footprint > 0]
        if inside.size == 0:
            return np.zeros(lab.shape[:2], dtype=np.uint8)
        threshold = float(np.percentile(inside, float(self.params["seam_gradient_percentile"])))
        edges = np.where((magnitude >= threshold) & (footprint > 0), 255, 0).astype(np.uint8)
        return edges

    def propose_masks(self, image: np.ndarray) -> List[np.ndarray]:
        """FastSAM everything-mode proposals as full-resolution binary masks."""
        model = self._load_fastsam()
        results = model(
            image,
            device=str(self.pipeline.get("device", "cpu")),
            retina_masks=True,
            imgsz=int(self.sam.get("imgsz", 1024)),
            conf=float(self.sam.get("sam_conf", 0.25)),
            iou=float(self.sam.get("sam_iou", 0.9)),
            verbose=False,
        )
        if not results or results[0].masks is None:
            return []
        height, width = image.shape[:2]
        proposals: list[np.ndarray] = []
        for mask in results[0].masks.data.detach().cpu().numpy():
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            proposals.append(np.where(mask > 0.5, 255, 0).astype(np.uint8))
        return proposals

    def select_floor(self, cell: np.ndarray, proposals: List[np.ndarray]) -> Optional[np.ndarray]:
        """Union of FastSAM floor pieces inside one cell, or ``None`` to keep the cell.

        A piece must lie mostly inside the cell (so whole-plan and cross-room
        masks are ignored) and cover at least ``floor_piece_min_share`` of it
        (so small wall faces, fixtures and decor are left out). Holes left by
        furniture are filled unless they reach the cell edge.
        """
        cell_area = int(np.count_nonzero(cell))
        if not proposals or cell_area == 0:
            return None
        min_inside = float(self.params["floor_piece_min_inside"])
        min_share = float(self.params["floor_piece_min_share"])
        union = np.zeros_like(cell)
        for proposal in proposals:
            proposal_area = int(np.count_nonzero(proposal))
            if proposal_area == 0:
                continue
            overlap = (proposal > 0) & (cell > 0)
            overlap_area = int(np.count_nonzero(overlap))
            if overlap_area / proposal_area < min_inside or overlap_area / cell_area < min_share:
                continue
            union[overlap] = 255
        if not np.any(union):
            return None
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        union = cv2.bitwise_and(union, cell)
        union = self._fill_enclosed_holes(self._largest_component(union), cv2.bitwise_not(cell))
        if np.count_nonzero(union) / cell_area < float(self.params["floor_min_coverage"]):
            return None
        return union

    def seam_alignment(self, masks: List[np.ndarray], seam_edges: np.ndarray) -> float:
        """Fraction of mask boundary pixels within ``seam_tolerance_px`` of a strong edge."""
        if not masks:
            return 0.0
        distance = cv2.distanceTransform(cv2.bitwise_not(seam_edges), cv2.DIST_L2, 3)
        tolerance = float(self.params["seam_tolerance_px"])
        hits = 0
        total = 0
        for mask in masks:
            boundary = cv2.subtract(mask, cv2.erode(mask, np.ones((3, 3), np.uint8))) > 0
            total += int(np.count_nonzero(boundary))
            hits += int(np.count_nonzero(boundary & (distance <= tolerance)))
        return hits / total if total else 0.0

    def _keep_structural(self, ridges: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Keep ridge pieces that touch the outer shell or are elongated; drop fixtures."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ridges, connectivity=8)
        shell = cv2.subtract(footprint, cv2.erode(footprint, np.ones((9, 9), np.uint8)))
        shell_labels = set(np.unique(labels[shell > 0]).tolist())
        min_area = int(self.params["ridge_min_area_px"])
        min_aspect = float(self.params["ridge_min_aspect"])
        kept = np.zeros_like(ridges)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            aspect = max(width, height) / max(1, min(width, height))
            if index in shell_labels or aspect >= min_aspect or area >= 10 * min_area:
                kept[labels == index] = 255
        return kept

    @staticmethod
    def _fill_enclosed_holes(cell: np.ndarray, others: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(cell)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        holes = cv2.subtract(filled, cell)
        count, labels = cv2.connectedComponents(holes, connectivity=8)
        result = cell.copy()
        for index in range(1, count):
            hole = labels == index
            if not np.any(hole & (others > 0)):
                result[hole] = 255
        return result

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        if count <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, 255, 0).astype(np.uint8)

    def _load_fastsam(self):
        if self._model is not None:
            return self._model
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import FastSAM

        self._model = FastSAM(str(self._resolve_weights()))
        return self._model

    def _resolve_weights(self) -> Path:
        configured = Path(str(self.sam["weights"]))
        if not configured.is_absolute():
            configured = _project_root() / configured
        configured.parent.mkdir(parents=True, exist_ok=True)
        if configured.is_file() and configured.stat().st_size > 0:
            return configured
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(configured.name))
        if not downloaded.is_file():
            raise FileNotFoundError(f"Could not download FastSAM weights: {configured.name}")
        if downloaded.resolve() != configured.resolve():
            shutil.copyfile(downloaded, configured)
        return configured
