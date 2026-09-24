"""Phase 1: extract walkable floor on the original render."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

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


@dataclass
class FloorProposal:
    mask: np.ndarray
    confidence: float
    canvas_fraction: float
    wall_fraction: float
    inradius: float
    bbox_fill: float
    hole_fraction: float


@dataclass
class FloorSegmentationDebug:
    footprint: np.ndarray
    wall_tops: np.ndarray
    masks: List[np.ndarray]
    unclaimed: np.ndarray
    confidences: List[float] = field(default_factory=list)
    proposal_count: int = 0
    object_count: int = 0
    container_count: int = 0
    recovered_count: int = 0
    sam_available: bool = True
    notes: List[str] = field(default_factory=list)


class FloorSegmenter:
    """FastSAM floor regions on the unwarped render, merged into one footprint.

    The render is never warped. FastSAM everything-mode proposals are filtered
    in place: thin flat shell-coloured ribbons are wall tops and are removed,
    solid high-objectness blobs are furniture, and the remaining thick regions
    are the floors. Masks that swallowed a whole wing of walls are stripped
    back to the non-wall pixels so those floors are not dropped. The kept
    regions are disjoint. ``walkable_footprint`` unions them and fills
    furniture holes that do not cross a wall top.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.pipeline = self.config["pipeline"]
        self.sam = self.config["segmentation"]
        self.params = self.config["v3_segmentation"]
        self._model = None
        self.debug: Optional[FloorSegmentationDebug] = None

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """Return one binary floor mask (uint8, 0/255) per floor region."""
        footprint = self._building_footprint(image)
        wall_tops = self._wall_tops(image, footprint)
        band = _as_odd(int(self.params["interior_erode_px"]))
        self._core = cv2.erode(footprint, np.ones((band, band), np.uint8))
        interior_area = max(int(np.count_nonzero(footprint)), 1)
        min_area = max(int(self.params["min_floor_area_px"]), int(float(self.params["min_area_ratio"]) * interior_area))
        min_inradius = max(
            float(self.params["min_inradius_px"]),
            float(self.params["min_inradius_ratio"]) * float(np.sqrt(interior_area)),
        )

        notes: list[str] = []
        sam_available = True
        try:
            raw_masks, confidences = self._propose(image)
        except Exception as exc:
            raw_masks, confidences = [], []
            sam_available = False
            notes.append(f"FastSAM unavailable ({exc}); no floor masks.")

        pure: list[np.ndarray] = []
        pure_conf: list[float] = []
        containers: list[np.ndarray] = []
        objects: list[np.ndarray] = []
        for mask, confidence in zip(raw_masks, confidences):
            proposal = self._describe(mask, confidence, footprint, wall_tops)
            if proposal is None:
                continue
            if self._is_object(proposal):
                objects.append(proposal.mask)
                continue
            if proposal.wall_fraction > float(self.params["max_wall_fraction"]):
                if self._is_container(proposal, interior_area):
                    containers.append(proposal.mask)
                continue
            if proposal.inradius < min_inradius:
                continue
            for component in self._components(proposal.mask, min_area, min_inradius):
                pure.append(component)
                pure_conf.append(proposal.confidence)

        kept, kept_conf = self._absorb(list(zip(pure, pure_conf)), min_area, min_inradius)
        recovered = self._recover(containers, objects, kept, footprint, wall_tops, min_area, min_inradius)
        if recovered:
            extra, extra_conf = self._absorb(
                [(mask, 0.0) for mask in recovered],
                min_area,
                min_inradius,
                claimed=kept,
            )
            kept.extend(extra)
            kept_conf.extend(extra_conf)
        free = self._free_floor(footprint, wall_tops, kept, objects)
        unclaimed = self._components(free, min_area, min_inradius)
        if unclaimed:
            kept, kept_conf = self._merge_unclaimed(kept, kept_conf, unclaimed, min_area, min_inradius)

        self.debug = FloorSegmentationDebug(
            footprint=footprint,
            wall_tops=wall_tops,
            masks=kept,
            unclaimed=free,
            confidences=kept_conf,
            proposal_count=len(raw_masks),
            object_count=len(objects),
            container_count=len(containers),
            recovered_count=len(recovered) + len(unclaimed),
            sam_available=sam_available,
            notes=notes,
        )
        return kept

    def walkable_footprint(self, masks: Sequence[np.ndarray]) -> np.ndarray:
        """Union the floor masks and fill furniture holes that do not cross a wall top."""
        if self.debug is None:
            raise RuntimeError("segment() must run before walkable_footprint().")
        shape = self.debug.footprint.shape
        if not masks:
            return np.zeros(shape, dtype=np.uint8)
        union = np.zeros(shape, dtype=np.uint8)
        for mask in masks:
            union[mask > 0] = 255
        union = cv2.bitwise_and(union, self.debug.footprint)
        union[self.debug.wall_tops > 0] = 0
        union = self._fill_furniture_holes(union, self.debug.wall_tops)
        union = cv2.bitwise_and(union, self.debug.footprint)
        union[self.debug.wall_tops > 0] = 0
        return union

    def _propose(self, image: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        model = self._load_fastsam()
        results = model(
            image,
            device=str(self.pipeline.get("device", "cpu")),
            retina_masks=True,
            imgsz=int(self.sam.get("imgsz", 1024)),
            conf=float(self.params["proposal_conf"]),
            iou=float(self.sam.get("sam_iou", 0.9)),
            verbose=False,
        )
        if not results or results[0].masks is None:
            return [], []
        height, width = image.shape[:2]
        confidences = (
            results[0].boxes.conf.detach().cpu().numpy().tolist()
            if results[0].boxes is not None and results[0].boxes.conf is not None
            else [1.0] * len(results[0].masks.data)
        )
        masks: list[np.ndarray] = []
        for raw in results[0].masks.data.detach().cpu().numpy():
            if raw.shape != (height, width):
                raw = cv2.resize(raw.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            masks.append(np.where(raw > 0.5, 255, 0).astype(np.uint8))
        return masks, [float(value) for value in confidences]

    def _building_footprint(self, image: np.ndarray) -> np.ndarray:
        """Filled apartment silhouette. The white page around the render is excluded."""
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

    def _wall_tops(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Thin, flat, shell-coloured ribbons. These are wall tops, not floor.

        Wide flat surfaces (tile, a balcony deck, a counter) survive the
        opening and stay eligible as floor. Only the ribbon that disappears
        under ``wall_top_open_px`` is marked.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lightness = lab[:, :, 0]
        window = _as_odd(int(self.params["flat_window_px"]))
        mean = cv2.blur(lightness, (window, window))
        deviation = np.sqrt(np.maximum(cv2.blur(lightness * lightness, (window, window)) - mean * mean, 0.0))
        flat = deviation <= float(self.params["flat_std_max"])

        band = int(self.params["shell_band_px"])
        kernel = np.ones((band, band), np.uint8)
        shell_band = (footprint > 0) & (cv2.erode(footprint, kernel) == 0) & flat
        if int(np.count_nonzero(shell_band)) < 50:
            return np.zeros(footprint.shape, dtype=np.uint8)
        reference = np.median(lab[shell_band], axis=0)
        ab_distance = np.maximum(
            np.abs(lab[:, :, 1] - reference[1]),
            np.abs(lab[:, :, 2] - reference[2]),
        )
        shell_colour = (
            flat
            & (ab_distance <= float(self.params["wall_ab_tolerance"]))
            & (np.abs(lightness - reference[0]) <= float(self.params["wall_lightness_tolerance"]))
            & (footprint > 0)
        )
        width = _as_odd(int(self.params["wall_top_open_px"]))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        thick = cv2.morphologyEx(shell_colour.astype(np.uint8) * 255, cv2.MORPH_OPEN, disk)
        thin = cv2.subtract(shell_colour.astype(np.uint8) * 255, thick)
        thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return thin

    def _describe(
        self,
        mask: np.ndarray,
        confidence: float,
        footprint: np.ndarray,
        wall_tops: np.ndarray,
    ) -> Optional[FloorProposal]:
        area = int(np.count_nonzero(mask))
        if area == 0:
            return None
        canvas_fraction = float(np.count_nonzero((mask > 0) & (footprint == 0)) / area)
        if canvas_fraction > float(self.params["max_canvas_fraction"]):
            return None
        cleaned = np.where((mask > 0) & (footprint > 0) & (wall_tops == 0), 255, 0).astype(np.uint8)
        cleaned_area = int(np.count_nonzero(cleaned))
        if cleaned_area == 0:
            return None
        return FloorProposal(
            mask=cleaned,
            confidence=confidence,
            canvas_fraction=canvas_fraction,
            wall_fraction=float(np.count_nonzero((mask > 0) & (wall_tops > 0)) / area),
            inradius=self._inradius(cleaned),
            bbox_fill=self._bbox_fill(cleaned),
            hole_fraction=self._hole_fraction(cleaned),
        )

    def _is_object(self, proposal: FloorProposal) -> bool:
        """Furniture scores as an object. Floor planes do not, even when the region is a clean rectangle."""
        return proposal.confidence >= float(self.params["object_conf"])

    def _is_container(self, proposal: FloorProposal, interior_area: int) -> bool:
        area = int(np.count_nonzero(proposal.mask))
        return (
            proposal.wall_fraction <= float(self.params["container_wall_max"])
            and area >= float(self.params["container_min_area_ratio"]) * interior_area
            and proposal.inradius >= float(self.params["min_inradius_px"])
        )

    def _recover(
        self,
        containers: Sequence[np.ndarray],
        objects: Sequence[np.ndarray],
        claimed_masks: Sequence[np.ndarray],
        footprint: np.ndarray,
        wall_tops: np.ndarray,
        min_area: int,
        min_inradius: float,
    ) -> List[np.ndarray]:
        """Floors that FastSAM only returned inside a wall-swallowing mask."""
        if not containers:
            return []
        claimed = np.zeros(footprint.shape, dtype=np.uint8)
        for mask in claimed_masks:
            claimed[mask > 0] = 255
        blocked = claimed.copy()
        for mask in objects:
            blocked[mask > 0] = 255
        recovered: list[np.ndarray] = []
        for container in sorted(containers, key=lambda item: int(np.count_nonzero(item)), reverse=True):
            region = np.where((container > 0) & (blocked == 0), 255, 0).astype(np.uint8)
            for component in self._components(region, min_area, min_inradius):
                recovered.append(component)
                blocked[component > 0] = 255
        return recovered

    def _free_floor(
        self,
        footprint: np.ndarray,
        wall_tops: np.ndarray,
        claimed_masks: Sequence[np.ndarray],
        objects: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Interior pixels FastSAM did not claim as a floor, a wall top, or furniture.

        Beige carpet is the usual miss. It is not an object and it is wider than
        a wall top, so it survives here. A short close joins floor pixels that
        furniture notches split, then wall tops are stamped back so the close
        cannot bridge two rooms.
        """
        blocked = np.zeros(footprint.shape, dtype=np.uint8)
        for mask in claimed_masks:
            blocked[mask > 0] = 255
        for mask in objects:
            blocked[mask > 0] = 255
        free = np.where((footprint > 0) & (wall_tops == 0) & (blocked == 0), 255, 0).astype(np.uint8)
        close = _as_odd(int(self.params["unclaimed_close_px"]))
        if close > 1:
            free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
            free[wall_tops > 0] = 0
            free[blocked > 0] = 0
            free = cv2.bitwise_and(free, footprint)
        return free

    def _merge_unclaimed(
        self,
        kept: List[np.ndarray],
        kept_conf: List[float],
        unclaimed: Sequence[np.ndarray],
        min_area: int,
        min_inradius: float,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Add uncovered floor. A solid mask surrounded by that floor is furniture and is absorbed."""
        extra, extra_conf = self._absorb(
            [(mask, 0.0) for mask in unclaimed],
            min_area,
            min_inradius,
            claimed=kept,
        )
        if not extra:
            return kept, kept_conf
        floor = np.zeros(extra[0].shape, dtype=np.uint8)
        for mask in extra:
            floor[mask > 0] = 255
        survivors: list[np.ndarray] = []
        survivor_conf: list[float] = []
        for mask, confidence in zip(kept, kept_conf):
            area = int(np.count_nonzero(mask))
            if area == 0:
                continue
            embedded = self._boundary_contact(mask, floor)
            if (
                embedded >= float(self.params["absorb_contact_min"])
                and self._bbox_fill(mask) >= float(self.params["object_fill_min"])
                and area < int(np.count_nonzero(floor))
            ):
                continue
            survivors.append(mask)
            survivor_conf.append(confidence)
        survivors.extend(extra)
        survivor_conf.extend(extra_conf)
        return survivors, survivor_conf

    @staticmethod
    def _boundary_contact(mask: np.ndarray, floor: np.ndarray) -> float:
        """Share of ``mask``'s outline that touches ``floor``. A bed on carpet scores high; a neighbouring room does not."""
        boundary = cv2.subtract(mask, cv2.erode(mask, np.ones((3, 3), np.uint8)))
        boundary_area = int(np.count_nonzero(boundary))
        if boundary_area == 0:
            return 0.0
        near = cv2.dilate(floor, np.ones((5, 5), np.uint8))
        return int(np.count_nonzero((boundary > 0) & (near > 0))) / boundary_area

    def _absorb(
        self,
        ranked: Sequence[tuple[np.ndarray, float]],
        min_area: int,
        min_inradius: float,
        claimed: Sequence[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Keep the largest region when several proposals describe the same floor."""
        ordered = sorted(ranked, key=lambda item: int(np.count_nonzero(item[0])), reverse=True)
        kept: list[np.ndarray] = []
        confidences: list[float] = []
        if claimed:
            kept.extend(claimed)
        fresh: list[np.ndarray] = []
        fresh_conf: list[float] = []
        coverage = float(self.params["nms_coverage"])
        for mask, confidence in ordered:
            remaining = mask.copy()
            for prior in kept:
                overlap = int(np.count_nonzero((remaining > 0) & (prior > 0)))
                area = int(np.count_nonzero(remaining))
                if area == 0:
                    break
                if overlap / area >= coverage:
                    remaining = np.zeros_like(remaining)
                    break
                remaining[(prior > 0)] = 0
            for component in self._components(remaining, min_area, min_inradius):
                kept.append(component)
                fresh.append(component)
                fresh_conf.append(confidence)
        if claimed:
            return fresh, fresh_conf
        return kept, fresh_conf

    def _components(self, mask: np.ndarray, min_area: int, min_inradius: float) -> List[np.ndarray]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components: list[np.ndarray] = []
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
                continue
            component = np.where(labels == index, 255, 0).astype(np.uint8)
            if self._inradius(component) < min_inradius:
                continue
            if not self._in_interior(component):
                continue
            components.append(component)
        return components

    def _fill_furniture_holes(self, mask: np.ndarray, wall_tops: np.ndarray) -> np.ndarray:
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        filled = mask.copy()
        for index in range(1, count):
            component = np.where(labels == index, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            interior = np.zeros_like(component)
            cv2.drawContours(interior, contours, -1, 255, thickness=cv2.FILLED)
            holes = cv2.subtract(interior, component)
            holes[wall_tops > 0] = 0
            filled[holes > 0] = 255
        return filled

    def _in_interior(self, mask: np.ndarray) -> bool:
        """Reject ribbons that live on the exterior wall band outside the rooms."""
        core = getattr(self, "_core", None)
        if core is None or not np.any(core):
            return True
        area = int(np.count_nonzero(mask))
        if area == 0:
            return False
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return False
        center_x = int(np.clip(round(moments["m10"] / moments["m00"]), 0, core.shape[1] - 1))
        center_y = int(np.clip(round(moments["m01"] / moments["m00"]), 0, core.shape[0] - 1))
        if core[center_y, center_x] == 0:
            return False
        inside = int(np.count_nonzero((mask > 0) & (core > 0)))
        return inside / area >= float(self.params["interior_min_fraction"])

    @staticmethod
    def _inradius(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(cv2.distanceTransform(mask, cv2.DIST_L2, 3).max())

    @staticmethod
    def _bbox_fill(mask: np.ndarray) -> float:
        points = cv2.findNonZero(mask)
        if points is None:
            return 0.0
        _, _, width, height = cv2.boundingRect(points)
        area = int(np.count_nonzero(mask))
        return area / max(width * height, 1)

    @staticmethod
    def _hole_fraction(mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        filled_area = int(np.count_nonzero(filled))
        if filled_area == 0:
            return 0.0
        return 1.0 - int(np.count_nonzero(mask)) / filled_area

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
