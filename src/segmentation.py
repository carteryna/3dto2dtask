"""Hybrid room segmentation: FastSAM proposals with a seeded-watershed fallback."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import yaml

Seed = Tuple[int, int]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class RoomSegmenter:
    """Turn Phase 2 seeds and closed wall masks into discrete room masks.

    FastSAM is prompted with each room centroid. A seed whose proposal is
    missing, low-confidence, or shared with another seed is replaced by the
    seeded-watershed region for that centroid. Overlaps are then resolved so
    the returned masks do not share pixels.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.pipeline = self.config["pipeline"]
        self.segmentation = self.config["segmentation"]
        self._model = None
        self._seeds: list[Seed] = []
        self._confidences: list[float] = []
        self.raw_sam_masks: list[np.ndarray] = []
        self.fallback_count = 0
        self.mask_confidences: list[float] = []

    def segment(
        self,
        image: np.ndarray,
        seeds: Sequence[Sequence[float]],
        closed_wall_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Run FastSAM, fill failed seeds with watershed, and return clean masks."""
        seed_list = self._normalize_seeds(seeds, closed_wall_mask.shape)
        self._seeds = seed_list
        min_area = int(self.segmentation["min_room_area_px"])
        if not seed_list:
            self.raw_sam_masks = []
            self._confidences = []
            self.fallback_count = 0
            return []

        use_watershed_only = self.segmentation.get("model_type") == "watershed_fallback"
        if use_watershed_only:
            sam_masks = [self._empty(closed_wall_mask) for _ in seed_list]
            self._confidences = [0.0] * len(seed_list)
        else:
            try:
                sam_masks = self.segment_with_fastsam(image, seed_list, closed_wall_mask)
            except Exception as exc:
                print(f"FastSAM failed ({exc}); using watershed for every seed.")
                sam_masks = [self._empty(closed_wall_mask) for _ in seed_list]
                self._confidences = [0.0] * len(seed_list)

        self.raw_sam_masks = sam_masks
        watershed_masks = self.segment_with_watershed_fallback(image, seed_list, closed_wall_mask)
        min_conf = float(self.segmentation["min_mask_confidence"])
        hybrid: list[np.ndarray] = []
        hybrid_conf: list[float] = []
        fallbacks = 0
        for mask, water_mask, confidence, seed in zip(
            sam_masks, watershed_masks, self._confidences, seed_list
        ):
            if self._sam_mask_is_valid(
                mask, seed, seed_list, confidence, min_conf, min_area, water_mask
            ):
                hybrid.append(mask)
                hybrid_conf.append(confidence)
            else:
                hybrid.append(water_mask)
                hybrid_conf.append(min_conf)
                fallbacks += 1
        self._confidences = hybrid_conf
        self.fallback_count = fallbacks
        return self.filter_and_merge_masks(hybrid, min_area)

    def segment_with_fastsam(
        self,
        image: np.ndarray,
        seeds: List[Seed],
        wall_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Prompt FastSAM with room centroids and cut the proposals at walls.

        One mask is returned per seed. The mask is empty when no proposal
        contains that centroid. Confidence for each seed is stored on
        ``self._confidences``.
        """
        if not seeds:
            self._confidences = []
            return []

        model = self._load_fastsam()
        device = str(self.pipeline.get("device", "cpu"))
        points = [[int(x), int(y)] for x, y in seeds]
        labels = [1] * len(points)
        results = model.predict(
            image,
            points=points,
            labels=labels,
            device=device,
            retina_masks=True,
            imgsz=int(self.segmentation.get("imgsz", 1024)),
            conf=float(self.segmentation.get("sam_conf", 0.25)),
            iou=float(self.segmentation.get("sam_iou", 0.9)),
            verbose=False,
        )
        proposals, confidences = self._proposal_arrays(results, image.shape[:2])
        footprint = self._footprint_from_walls(wall_mask)

        aligned: list[np.ndarray] = []
        aligned_conf: list[float] = []
        height, width = wall_mask.shape[:2]
        for x, y in points:
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            x, y = self._snap_to_floor(x, y, wall_mask, footprint)
            chosen, confidence = self._select_proposal(proposals, confidences, x, y)
            if chosen is None:
                aligned.append(self._empty(wall_mask))
                aligned_conf.append(0.0)
                continue
            chosen[wall_mask > 0] = 0
            chosen[footprint == 0] = 0
            chosen = self._component_at(chosen, x, y)
            aligned.append(chosen)
            aligned_conf.append(confidence)
        self._confidences = aligned_conf
        return aligned

    def segment_with_watershed_fallback(
        self,
        image: np.ndarray,
        seeds: List[Seed],
        closed_wall_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Label each seed with watershed on the bilateral image, bounded by walls."""
        if not seeds:
            return []

        height, width = closed_wall_mask.shape[:2]
        # Blur the bilateral render so furniture edges do not become room walls.
        # The closed wall mask is stamped back in as the hard barrier.
        guided = cv2.GaussianBlur(image, (31, 31), 0)
        guided[closed_wall_mask > 0] = (255, 255, 255)

        footprint = self._footprint_from_walls(closed_wall_mask)
        barrier = len(seeds) + 1
        markers = np.zeros((height, width), dtype=np.int32)
        markers[footprint == 0] = barrier
        markers[closed_wall_mask > 0] = barrier
        for index, (x, y) in enumerate(seeds, start=1):
            snapped_x, snapped_y = self._snap_to_floor(x, y, closed_wall_mask, footprint)
            cv2.circle(markers, (snapped_x, snapped_y), 2, int(index), thickness=-1)

        labels = cv2.watershed(guided, markers)
        masks: list[np.ndarray] = []
        for index in range(1, len(seeds) + 1):
            mask = np.where(labels == index, 255, 0).astype(np.uint8)
            mask[closed_wall_mask > 0] = 0
            mask[footprint == 0] = 0
            masks.append(mask)
        return masks

    def filter_and_merge_masks(self, raw_masks: List[np.ndarray], min_area: int) -> List[np.ndarray]:
        """Drop tiny fragments and give every disputed pixel to one room.

        A pixel that belongs to several masks is assigned to the higher
        confidence region, with distance to that region's seed as the tie break.
        """
        if not raw_masks:
            self.mask_confidences = []
            return []

        height, width = raw_masks[0].shape[:2]
        kept: list[np.ndarray] = []
        kept_seeds: list[Seed] = []
        kept_conf: list[float] = []
        for index, mask in enumerate(raw_masks):
            binary = np.where(mask > 0, 255, 0).astype(np.uint8)
            if int(np.count_nonzero(binary)) < min_area:
                continue
            kept.append(binary)
            kept_seeds.append(self._seed_for(index, binary))
            confidence = self._confidences[index] if index < len(self._confidences) else 1.0
            kept_conf.append(float(confidence))
        if not kept:
            self.mask_confidences = []
            return []

        score = np.full((len(kept), height, width), -np.inf, dtype=np.float32)
        yy, xx = np.indices((height, width))
        for index, mask in enumerate(kept):
            seed_x, seed_y = kept_seeds[index]
            distance = np.sqrt((xx - seed_x) ** 2 + (yy - seed_y) ** 2)
            score[index] = np.where(mask > 0, kept_conf[index] * 1.0e6 - distance, -np.inf)

        winner = np.argmax(score, axis=0)
        claimed = np.isfinite(score.max(axis=0))
        cleaned: list[np.ndarray] = []
        confidences: list[float] = []
        for index in range(len(kept)):
            mask = np.where((winner == index) & claimed, 255, 0).astype(np.uint8)
            if int(np.count_nonzero(mask)) >= min_area:
                cleaned.append(mask)
                confidences.append(kept_conf[index])
        self.mask_confidences = confidences
        return cleaned

    def _load_fastsam(self):
        if self._model is not None:
            return self._model
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import FastSAM

        weights = self._resolve_weights()
        self._model = FastSAM(str(weights))
        return self._model

    def _resolve_weights(self) -> Path:
        configured = Path(str(self.segmentation["weights"]))
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

    @staticmethod
    def _proposal_arrays(results, shape: tuple[int, int]) -> tuple[list[np.ndarray], list[float]]:
        height, width = shape
        if not results:
            return [], []
        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return [], []
        data = result.masks.data.detach().cpu().numpy()
        if result.boxes is not None and result.boxes.conf is not None:
            confidences = result.boxes.conf.detach().cpu().numpy().tolist()
        else:
            confidences = [1.0] * len(data)
        proposals: list[np.ndarray] = []
        for mask in data:
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            proposals.append(np.where(mask > 0.5, 255, 0).astype(np.uint8))
        return proposals, [float(value) for value in confidences]

    @staticmethod
    def _select_proposal(
        proposals: list[np.ndarray],
        confidences: list[float],
        x: int,
        y: int,
    ) -> tuple[np.ndarray | None, float]:
        """Smallest proposal containing the seed, so a full-plan mask loses to a room."""
        best: np.ndarray | None = None
        best_area: int | None = None
        best_conf = 0.0
        for index, proposal in enumerate(proposals):
            if proposal[y, x] == 0:
                continue
            area = int(np.count_nonzero(proposal))
            confidence = confidences[index] if index < len(confidences) else 0.0
            if best_area is None or area < best_area:
                best = proposal.copy()
                best_area = area
                best_conf = confidence
        return best, best_conf

    def _sam_mask_is_valid(
        self,
        mask: np.ndarray,
        seed: Seed,
        seeds: Sequence[Seed],
        confidence: float,
        min_conf: float,
        min_area: int,
        watershed_mask: np.ndarray | None = None,
    ) -> bool:
        if confidence < min_conf or int(np.count_nonzero(mask)) < min_area:
            return False
        contained = [
            other
            for other in seeds
            if 0 <= other[1] < mask.shape[0]
            and 0 <= other[0] < mask.shape[1]
            and mask[other[1], other[0]] > 0
        ]
        if seed not in contained or len(contained) != 1:
            return False
        if int(np.count_nonzero(mask)) > 0.55 * mask.shape[0] * mask.shape[1]:
            return False
        if watershed_mask is not None:
            watershed_area = int(np.count_nonzero(watershed_mask))
            if watershed_area == 0:
                return False
            overlap = int(np.count_nonzero((mask > 0) & (watershed_mask > 0)))
            # Furniture proposals sit inside the room cell and cover only a fraction of it.
            if overlap / watershed_area < 0.45:
                return False
        return True

    def _seed_for(self, index: int, mask: np.ndarray) -> Seed:
        if index < len(self._seeds):
            return self._seeds[index]
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return (0, 0)
        return (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))

    @staticmethod
    def _normalize_seeds(seeds: Sequence[Sequence[float]], shape: tuple[int, ...]) -> list[Seed]:
        height, width = shape[:2]
        normalized: list[Seed] = []
        for seed in np.asarray(seeds).reshape(-1, 2):
            x = int(np.clip(round(float(seed[0])), 0, width - 1))
            y = int(np.clip(round(float(seed[1])), 0, height - 1))
            normalized.append((x, y))
        return normalized

    @staticmethod
    def _empty(mask: np.ndarray) -> np.ndarray:
        return np.zeros(mask.shape[:2], dtype=np.uint8)

    @staticmethod
    def _footprint_from_walls(wall_mask: np.ndarray) -> np.ndarray:
        binary = np.where(wall_mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        footprint = np.zeros_like(binary)
        if not contours:
            return np.full_like(binary, 255)
        outline = max(contours, key=cv2.contourArea)
        cv2.drawContours(footprint, [outline], -1, 255, thickness=cv2.FILLED)
        return footprint

    @staticmethod
    def _snap_to_floor(x: int, y: int, wall_mask: np.ndarray, footprint: np.ndarray) -> Seed:
        height, width = wall_mask.shape[:2]
        if wall_mask[y, x] == 0 and footprint[y, x] > 0:
            return x, y
        for radius in range(1, 25):
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            window = (wall_mask[y0:y1, x0:x1] == 0) & (footprint[y0:y1, x0:x1] > 0)
            coords = np.argwhere(window)
            if len(coords) == 0:
                continue
            local_y, local_x = coords[np.argmin((coords[:, 0] - (y - y0)) ** 2 + (coords[:, 1] - (x - x0)) ** 2)]
            return int(x0 + local_x), int(y0 + local_y)
        return x, y

    @staticmethod
    def _component_at(mask: np.ndarray, x: int, y: int) -> np.ndarray:
        if mask[y, x] == 0:
            return np.zeros_like(mask)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        return np.where(labels == labels[y, x], 255, 0).astype(np.uint8)
