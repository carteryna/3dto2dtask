"""Structural wall extraction and topological door-gap closing."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _as_odd(size: int) -> int:
    size = max(int(size), 1)
    return size if size % 2 else size + 1


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Binary line structuring element through the kernel center."""
    size = _as_odd(length)
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 2
    theta = np.deg2rad(angle_deg)
    x1 = int(np.clip(round(center + radius * np.cos(theta)), 0, size - 1))
    y1 = int(np.clip(round(center + radius * np.sin(theta)), 0, size - 1))
    x2 = int(np.clip(round(center - radius * np.cos(theta)), 0, size - 1))
    y2 = int(np.clip(round(center - radius * np.sin(theta)), 0, size - 1))
    cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
    if kernel.max() == 0:
        kernel[center, :] = 1
    return kernel


class WallExtractor:
    """Wall-first preprocessor for extruded 3D floor-plan renders.

    Light-gray and white wall tops are isolated in HSV and LAB, the exterior
    canvas is removed, and 45-degree door gaps are closed so room interiors
    become separate topological cells.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.preprocessing = self.config["preprocessing"]
        self.segmentation = self.config["segmentation"]
        self._footprint: np.ndarray | None = None

    def load_and_denoise(self, image_path: str) -> np.ndarray:
        """Load a floor-plan render and bilateral-filter it.

        The filter smooths carpet grain, wood texture, and soft lighting while
        preserving the high-contrast edges of wall tops.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        params = self.preprocessing
        return cv2.bilateralFilter(
            image,
            d=int(params["bilateral_d"]),
            sigmaColor=float(params["bilateral_sigma_color"]),
            sigmaSpace=float(params["bilateral_sigma_space"]),
        )

    def extract_wall_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask of structural wall tops.

        Pixels inside ``wall_hsv_min`` / ``wall_hsv_max`` that are also bright
        and low-chroma in LAB are kept. The border-connected white canvas is
        removed, and compact interior blobs (rugs, fixtures) are dropped so the
        mask is 255 on walls and 0 on floors and background.
        """
        hsv_min = np.array(self.preprocessing["wall_hsv_min"], dtype=np.uint8)
        hsv_max = np.array(self.preprocessing["wall_hsv_max"], dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        chroma = np.maximum(
            np.abs(channel_a.astype(np.int16) - 128),
            np.abs(channel_b.astype(np.int16) - 128),
        )

        hsv_mask = cv2.inRange(hsv, hsv_min, hsv_max)
        value_min = int(hsv_min[2])
        saturation_max = int(hsv_max[1])
        lab_mask = np.where(
            (lightness >= value_min) & (chroma <= saturation_max),
            255,
            0,
        ).astype(np.uint8)
        wall = cv2.bitwise_and(hsv_mask, lab_mask)

        footprint = self._plan_footprint(lightness, chroma)
        self._footprint = footprint
        wall = cv2.bitwise_and(wall, footprint)
        wall = self._keep_structural_components(wall, footprint)
        return wall

    def close_door_gaps(self, wall_mask: np.ndarray) -> np.ndarray:
        """Bridge open door leaves so adjacent rooms become closed cells.

        An elliptical close plus line closes at 0, 45, 90, and 135 degrees use
        ``door_closing_kernel_size`` from the config. The diagonal kernels
        target 45-degree open doors that break the wall perimeter.
        """
        kernel_size = _as_odd(int(self.preprocessing["door_closing_kernel_size"]))
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, ellipse)
        for angle in (0.0, 45.0, 90.0, 135.0):
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, _line_kernel(kernel_size, angle))
        return closed

    def generate_room_seeds(self, closed_wall_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return interior room centroids and the floor distance map.

        The closed wall mask is inverted so floors are foreground. Distance
        transform peaks inside each floor cell (above ``min_room_area_px``)
        become one centroid seed per core. The exterior canvas is excluded.
        """
        inverted = cv2.bitwise_not(closed_wall_mask)
        if self._footprint is not None:
            floor = cv2.bitwise_and(inverted, self._footprint)
        else:
            floor = self._drop_border_components(inverted)

        distance = cv2.distanceTransform(floor, cv2.DIST_L2, 5)
        min_area = int(self.segmentation["min_room_area_px"])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(floor, connectivity=8)

        seeds: list[tuple[float, float]] = []
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
                continue
            component = labels == index
            peak = float(distance[component].max())
            if peak <= 0:
                continue
            core = (component & (distance >= 0.55 * peak)).astype(np.uint8)
            core_count, _, core_stats, centroids = cv2.connectedComponentsWithStats(core, connectivity=8)
            for core_index in range(1, core_count):
                if int(core_stats[core_index, cv2.CC_STAT_AREA]) < 5:
                    continue
                center_x, center_y = centroids[core_index]
                seeds.append((float(center_x), float(center_y)))

        if seeds:
            seed_array = np.asarray(seeds, dtype=np.float32)
        else:
            seed_array = np.empty((0, 2), dtype=np.float32)
        return seed_array, distance

    def _plan_footprint(self, lightness: np.ndarray, chroma: np.ndarray) -> np.ndarray:
        """Largest non-canvas region, closed so the plan is one solid silhouette."""
        canvas = ((lightness >= 250) & (chroma <= 8)).astype(np.uint8)
        count, labels = cv2.connectedComponents(canvas, connectivity=8)
        border_ids = set(
            np.unique(
                np.concatenate(
                    [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
                )
            ).tolist()
        )
        border_ids.discard(0)
        if count <= 1 or not border_ids:
            background = np.zeros(canvas.shape, dtype=bool)
        else:
            background = np.isin(labels, list(border_ids))

        footprint = np.where(background, 0, 255).astype(np.uint8)
        kernel_size = _as_odd(int(self.preprocessing["door_closing_kernel_size"]) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(footprint, connectivity=8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            footprint = np.where(labels == largest, 255, 0).astype(np.uint8)
        return cv2.dilate(footprint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    def _keep_structural_components(self, wall: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Keep shell-connected and elongated wall pieces; drop compact blobs."""
        opened = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        eroded = cv2.erode(footprint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        boundary = cv2.subtract(footprint, eroded)
        min_partition = max(int(self.segmentation["min_room_area_px"]) // 5, 1)

        kept = np.zeros_like(opened)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            aspect = max(width, height) / max(1, min(width, height))
            touches_shell = bool(np.any((labels == index) & (boundary > 0)))
            if touches_shell or (area >= min_partition and aspect >= 2.5):
                kept[labels == index] = 255
        if kept.max() == 0:
            return opened
        return kept

    @staticmethod
    def _drop_border_components(mask: np.ndarray) -> np.ndarray:
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        if count <= 1:
            return mask
        border_ids = set(
            np.unique(
                np.concatenate(
                    [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
                )
            ).tolist()
        )
        border_ids.discard(0)
        kept = mask.copy()
        for index in border_ids:
            kept[labels == index] = 0
        return kept
