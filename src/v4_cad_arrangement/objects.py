"""Phase 2: Semantic Object Decoupling & Fixture Anchors."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon, box as shapely_box


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class CADObject:
    category: str  # "bed", "sofa", "dining_table", "bathtub", "toilet", "nightstand", "chair", "rug"
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    polygon: Polygon
    mask: np.ndarray


@dataclass
class ObjectDetectionResult:
    objects: List[CADObject] = field(default_factory=list)
    furniture_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    fixture_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    bathroom_seeds: List[Tuple[float, float]] = field(default_factory=list)
    bedroom_seeds: List[Tuple[float, float]] = field(default_factory=list)
    living_seeds: List[Tuple[float, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class ObjectDetector:
    """Decouples interior furniture from structural floor boundaries and detects fixtures.

    Core Principle: Furniture belongs to a room; it must NEVER be subtracted as a hole in the floor.
    Plumbing fixtures (toilets, bathtubs) are detected to anchor bathroom boundaries and rescue
    photometrically camouflaged tile floors.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.pipeline = self.config.get("pipeline", {})
        self.sam_cfg = self.config.get("segmentation", {})
        self.params = self.config.get("v4_objects", {})
        self._model = None

    def detect(self, rectified_image: np.ndarray) -> ObjectDetectionResult:
        """Detect furniture objects, fixtures, and semantic seeds on the rectified image."""
        h, w = rectified_image.shape[:2]
        footprint = self._building_footprint(rectified_image)
        core = cv2.erode(footprint, np.ones((21, 21), np.uint8))

        objects: List[CADObject] = []
        notes: List[str] = []

        # 1. Direct Blue-Water Bathtub Detection
        tub_objects = self._detect_blue_tubs(rectified_image, footprint)
        objects.extend(tub_objects)

        # 2. FastSAM Object Proposals & Heuristic Classification
        try:
            sam_objects = self._detect_sam_objects(rectified_image, core, existing_tubs=tub_objects)
            objects.extend(sam_objects)
        except Exception as exc:
            notes.append(f"FastSAM object detection failed ({exc}).")

        # 3. Non-Maximum Suppression to remove duplicate proposals
        objects = self._non_max_suppression(objects, iou_thresh=0.50)

        # 4. Generate layer masks
        furniture_mask = np.zeros((h, w), dtype=np.uint8)
        fixture_mask = np.zeros((h, w), dtype=np.uint8)

        raw_bath_seeds: List[Tuple[float, float]] = []
        raw_bed_seeds: List[Tuple[float, float]] = []
        raw_living_seeds: List[Tuple[float, float]] = []

        furniture_cats = {"bed", "sofa", "dining_table", "nightstand", "chair", "rug"}
        fixture_cats = {"bathtub", "toilet", "fixture"}

        for obj in objects:
            cx = (obj.bbox[0] + obj.bbox[2]) / 2.0
            cy = (obj.bbox[1] + obj.bbox[3]) / 2.0

            if obj.category in furniture_cats:
                furniture_mask = cv2.bitwise_or(furniture_mask, obj.mask)
                if obj.category == "bed":
                    raw_bed_seeds.append((float(cx), float(cy)))
                elif obj.category == "sofa":
                    raw_living_seeds.append((float(cx), float(cy)))
            elif obj.category in fixture_cats:
                fixture_mask = cv2.bitwise_or(fixture_mask, obj.mask)
                if obj.category in {"bathtub", "toilet"}:
                    raw_bath_seeds.append((float(cx), float(cy)))

        # Cluster bathroom seeds so each bathroom gets exactly 1 unified centroid seed
        bathroom_seeds = self._cluster_seeds(raw_bath_seeds, max_dist=120.0)
        # Select highest-confidence bed seed for bedroom, sofa seed for living
        bedroom_seeds = self._cluster_seeds(raw_bed_seeds, max_dist=120.0)
        living_seeds = self._cluster_seeds(raw_living_seeds, max_dist=120.0)

        notes.append(
            f"Detected {len(objects)} CAD object(s): "
            f"{sum(1 for o in objects if o.category == 'bed')} bed(s), "
            f"{sum(1 for o in objects if o.category == 'sofa')} sofa(s), "
            f"{sum(1 for o in objects if o.category == 'dining_table')} table(s), "
            f"{sum(1 for o in objects if o.category == 'bathtub')} tub(s), "
            f"{sum(1 for o in objects if o.category == 'toilet')} toilet(s)."
        )

        return ObjectDetectionResult(
            objects=objects,
            furniture_mask=furniture_mask,
            fixture_mask=fixture_mask,
            bathroom_seeds=bathroom_seeds,
            bedroom_seeds=bedroom_seeds,
            living_seeds=living_seeds,
            notes=notes,
        )

    def _detect_blue_tubs(self, image: np.ndarray, footprint: np.ndarray) -> List[CADObject]:
        """Detect rectangular bathtubs with bright blue-water fills via HSV color filtering."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_min = int(self.params.get("tub_hue_min", 85))
        h_max = int(self.params.get("tub_hue_max", 135))
        s_min = int(self.params.get("tub_sat_min", 35))
        v_min = int(self.params.get("tub_val_min", 95))
        min_area = int(self.params.get("tub_min_area_px", 1500))
        max_area = int(self.params.get("tub_max_area_px", 15000))
        min_aspect = float(self.params.get("tub_min_aspect", 1.20))

        blue = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))
        blue = cv2.bitwise_and(blue, footprint)
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        num, _, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
        tubs = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            tw = int(stats[i, cv2.CC_STAT_WIDTH])
            th = int(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = max(tw, th) / max(min(tw, th), 1)
            if min_area <= area <= max_area and aspect >= min_aspect:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[y : y + th, x : x + tw] = 255
                poly = shapely_box(x, y, x + tw, y + th)
                tubs.append(
                    CADObject(
                        category="bathtub",
                        bbox=(float(x), float(y), float(x + tw), float(y + th)),
                        confidence=0.95,
                        polygon=poly,
                        mask=mask,
                    )
                )
        return tubs

    def _detect_sam_objects(
        self,
        image: np.ndarray,
        core: np.ndarray,
        existing_tubs: Sequence[CADObject],
    ) -> List[CADObject]:
        """Classify high-confidence FastSAM segment proposals into CAD objects."""
        model = self._load_fastsam()
        h, w = image.shape[:2]
        conf_thresh = float(self.params.get("proposal_conf", 0.30))
        results = model(
            image,
            device=str(self.pipeline.get("device", "cpu")),
            retina_masks=True,
            imgsz=int(self.sam_cfg.get("imgsz", 1024)),
            conf=conf_thresh,
            iou=0.8,
            verbose=False,
        )
        if not results or results[0].masks is None or results[0].boxes is None:
            return []

        raw_boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        raw_confs = results[0].boxes.conf.detach().cpu().numpy()
        raw_masks = results[0].masks.data.detach().cpu().numpy()

        # Step 1: Collect valid proposals inside core
        proposals = []
        for i in range(len(raw_confs)):
            b = raw_boxes[i]
            conf = float(raw_confs[i])
            mx = int(round((b[0] + b[2]) / 2.0))
            my = int(round((b[1] + b[3]) / 2.0))

            if mx < 0 or mx >= w or my < 0 or my >= h or core[my, mx] == 0:
                continue

            bw = float(b[2] - b[0])
            bh = float(b[3] - b[1])
            box_area = bw * bh
            if box_area > (w * h) * 0.25 or box_area < 200:
                continue

            raw_m = raw_masks[i]
            if raw_m.shape != (h, w):
                raw_m = cv2.resize(raw_m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.where(raw_m > 0.5, 255, 0).astype(np.uint8)
            poly = shapely_box(b[0], b[1], b[2], b[3])
            proposals.append({
                "bbox": (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                "conf": conf,
                "bw": bw,
                "bh": bh,
                "area": box_area,
                "aspect": max(bw, bh) / max(min(bw, bh), 1.0),
                "poly": poly,
                "mask": mask,
                "cx": (b[0] + b[2]) / 2.0,
                "cy": (b[1] + b[3]) / 2.0,
            })

        # Step 2: Identify Bed candidates
        found_beds: List[CADObject] = []
        bed_candidates = [
            p for p in proposals
            if 5000 <= p["area"] <= 24000 and min(p["bw"], p["bh"]) >= 55 and p["aspect"] <= 1.9
        ]
        # Sort by confidence and pick non-overlapping beds
        for p in sorted(bed_candidates, key=lambda x: x["conf"], reverse=True):
            overlap = any(
                p["poly"].intersection(b.polygon).area / min(p["area"], b.polygon.area) > 0.4
                for b in found_beds
            )
            if not overlap:
                found_beds.append(CADObject("bed", p["bbox"], p["conf"], p["poly"], p["mask"]))

        # Step 3: Identify Bathtubs & Toilets (Plumbing Co-occurrence Rule)
        found_tubs: List[CADObject] = list(existing_tubs)
        found_toilets: List[CADObject] = []
        found_small_furniture: List[CADObject] = []

        small_cand = [
            p for p in proposals
            if 250 <= p["area"] <= 1800 and 15 <= min(p["bw"], p["bh"]) <= 45 and p["aspect"] <= 1.6
        ]

        if found_tubs:
            # Blue tub detected: find closest toilet candidate to blue tub
            tb = found_tubs[0].bbox
            tc = ((tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0)
            near_toilets = [
                s for s in small_cand
                if np.hypot(s["cx"] - tc[0], s["cy"] - tc[1]) < 150
            ]
            if near_toilets:
                best_t = max(near_toilets, key=lambda x: x["conf"])
                found_toilets.append(CADObject("toilet", best_t["bbox"], best_t["conf"], best_t["poly"], best_t["mask"]))
        else:
            # No blue tub detected (e.g. white tile camouflage in Highlandlux).
            # Look for a pair of tub proposal (area 1400..6000) and toilet proposal (area 250..1800)
            # that are adjacent, not inside any bedroom, and have high surface lightness.
            tub_cand = [
                p for p in proposals
                if 1400 <= p["area"] <= 6000 and 1.2 <= p["aspect"] <= 2.2 and min(p["bw"], p["bh"]) >= 30
            ]
            best_pair = None
            best_conf = 0.0

            for t in tub_cand:
                # Tub must not overlap with detected beds
                if any(t["poly"].intersection(b.polygon).area / t["area"] > 0.25 for b in found_beds):
                    continue
                for s in small_cand:
                    # Toilet must not overlap with detected beds
                    if any(s["poly"].intersection(b.polygon).area / s["area"] > 0.25 for b in found_beds):
                        continue
                    # Must be separate objects (toilet not inside tub)
                    inter = t["poly"].intersection(s["poly"]).area
                    if inter / s["area"] > 0.10:
                        continue
                    d = np.hypot(t["cx"] - s["cx"], t["cy"] - s["cy"])
                    if 20.0 <= d <= 120.0:
                        # Lightness bonus for white porcelain/tile fixtures
                        crop_s = image[int(s["bbox"][1]) : int(s["bbox"][3]), int(s["bbox"][0]) : int(s["bbox"][2])]
                        l_bonus = 0.20 if (crop_s.size > 0 and np.mean(crop_s) > 130) else 0.0
                        score = t["conf"] + s["conf"] + l_bonus
                        if score > best_conf:
                            best_conf = score
                            best_pair = (t, s)

            if best_pair is not None:
                bt, bs = best_pair
                found_tubs.append(CADObject("bathtub", bt["bbox"], bt["conf"], bt["poly"], bt["mask"]))
                found_toilets.append(CADObject("toilet", bs["bbox"], bs["conf"], bs["poly"], bs["mask"]))

        # Step 4: Classify remaining small furniture (Nightstands vs Chairs)
        toilet_bboxes = {t.bbox for t in found_toilets}
        for p in small_cand:
            if p["bbox"] in toilet_bboxes:
                continue
            near_bed = any(
                np.hypot(p["cx"] - (b.bbox[0] + b.bbox[2]) / 2.0, p["cy"] - (b.bbox[1] + b.bbox[3]) / 2.0) < 80
                for b in found_beds
            )
            if near_bed:
                found_small_furniture.append(CADObject("nightstand", p["bbox"], p["conf"], p["poly"], p["mask"]))
            else:
                found_small_furniture.append(CADObject("chair", p["bbox"], p["conf"], p["poly"], p["mask"]))

        # Step 5: Identify Sofas
        found_sofas: List[CADObject] = []
        sofa_candidates = [
            p for p in proposals
            if 3000 <= p["area"] <= 20000 and min(p["bw"], p["bh"]) >= 28 and p["aspect"] >= 1.35
        ]
        for p in sorted(sofa_candidates, key=lambda x: x["conf"], reverse=True):
            # Must not overlap with beds or bathtubs
            overlap = any(
                p["poly"].intersection(b.polygon).area / min(p["area"], b.polygon.area) > 0.35
                for b in (found_beds + found_tubs + found_sofas)
            )
            if not overlap:
                found_sofas.append(CADObject("sofa", p["bbox"], p["conf"], p["poly"], p["mask"]))

        # Step 6: Identify Dining Tables
        found_tables: List[CADObject] = []
        table_candidates = [
            p for p in proposals
            if 2000 <= p["area"] <= 12000 and min(p["bw"], p["bh"]) >= 35 and p["aspect"] <= 1.55
        ]
        for p in sorted(table_candidates, key=lambda x: x["conf"], reverse=True):
            overlap = any(
                p["poly"].intersection(b.polygon).area / min(p["area"], b.polygon.area) > 0.35
                for b in (found_beds + found_tubs + found_sofas + found_tables)
            )
            if not overlap:
                found_tables.append(CADObject("dining_table", p["bbox"], p["conf"], p["poly"], p["mask"]))

        # Combine all newly detected objects (excluding existing tubs to prevent double-count)
        all_objs: List[CADObject] = []
        all_objs.extend(found_beds)
        if not existing_tubs:
            all_objs.extend(found_tubs)
        all_objs.extend(found_toilets)
        all_objs.extend(found_sofas)
        all_objs.extend(found_tables)
        all_objs.extend(found_small_furniture)

        return all_objs

    def _cluster_seeds(self, seeds: Sequence[Tuple[float, float]], max_dist: float = 120.0) -> List[Tuple[float, float]]:
        """Cluster nearby seed points into a single mean centroid seed."""
        if not seeds:
            return []
        clusters: List[List[Tuple[float, float]]] = []
        for s in seeds:
            placed = False
            for c in clusters:
                # Compare to first seed in cluster
                if np.hypot(s[0] - c[0][0], s[1] - c[0][1]) < max_dist:
                    c.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])

        # Return centroid of each cluster
        result = []
        for c in clusters:
            mx = float(np.mean([pt[0] for pt in c]))
            my = float(np.mean([pt[1] for pt in c]))
            result.append((round(mx, 1), round(my, 1)))
        return result

    def _non_max_suppression(self, objects: List[CADObject], iou_thresh: float = 0.50) -> List[CADObject]:
        """Remove overlapping duplicate detections."""
        if len(objects) <= 1:
            return objects

        ordered = sorted(objects, key=lambda obj: obj.confidence, reverse=True)
        kept: List[CADObject] = []

        for cand in ordered:
            suppressed = False
            b1 = cand.bbox
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            for prior in kept:
                # Allow different categories that legitimately sit close (e.g. toilet next to tub, nightstand next to bed)
                if cand.category != prior.category and {cand.category, prior.category} in [
                    {"toilet", "bathtub"}, {"bed", "nightstand"}, {"sofa", "dining_table"}
                ]:
                    continue
                b2 = prior.bbox
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])

                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    iou = inter / (area1 + area2 - inter + 1e-6)
                    io_min = inter / min(area1, area2 + 1e-6)
                    if iou > iou_thresh or io_min > 0.85:
                        suppressed = True
                        break
            if not suppressed:
                kept.append(cand)

        return kept

    @staticmethod
    def _building_footprint(image: np.ndarray) -> np.ndarray:
        """Extract filled apartment silhouette excluding the surrounding canvas."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        chroma = np.maximum(np.abs(A.astype(np.int16) - 128), np.abs(B.astype(np.int16) - 128))
        canvas = ((L >= 245) & (chroma <= 10)).astype(np.uint8)
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

    def _load_fastsam(self):
        if self._model is not None:
            return self._model
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import FastSAM

        self._model = FastSAM(str(self._resolve_weights()))
        return self._model

    def _resolve_weights(self) -> Path:
        configured = Path(str(self.sam_cfg.get("weights", "models/FastSAM-s.pt")))
        if not configured.is_absolute():
            configured = _project_root() / configured
        configured.parent.mkdir(parents=True, exist_ok=True)
        if configured.is_file() and configured.stat().st_size > 0:
            return configured
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(configured.name))
        if not downloaded.is_file():
            raise FileNotFoundError(f"Could not download FastSAM weights: {configured.name}")
        return configured
