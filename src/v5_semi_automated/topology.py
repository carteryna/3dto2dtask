"""Phase 2: Topological Extraction & Axial Door Bridging.

Ingests rectified human semantic guides, bridges doorway swings using constrained
axial morphological operators, and extracts closed room topological cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass
class TopologicalRoomNode:
    """Represents a topologically closed room region."""
    room_id: int
    label: str
    mask: np.ndarray  # Binary mask uint8
    contour: np.ndarray  # (N, 1, 2) int32 contour points
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area_px: float


@dataclass
class TopologyResult:
    """Result of Phase 2 topological extraction."""
    unit_key: str
    wall_mask: np.ndarray  # Binary mask uint8 of structural walls
    centerline_mask: np.ndarray  # Binary mask uint8 of 1D wall axes
    exterior_envelope_mask: np.ndarray  # Binary mask uint8 of unit footprint
    rooms: List[TopologicalRoomNode] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class SemiAutomatedTopology:
    """Extracts topological wall centerlines and closed room cycles."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.params = self.config.get("v5_topology", {})
        self.min_room_area_px = float(self.params.get("min_room_area_px", 800.0))
        self.door_gap_px = int(self.params.get("door_bridge_gap_px", 95))
        self.wall_alpha_thresh = int(self.params.get("wall_alpha_thresh", 50))
        self.red_diff_thresh = int(self.params.get("red_diff_thresh", 50))
        self.red_min_thresh = int(self.params.get("red_min_thresh", 150))

    def extract_wall_mask(self, rectified_guides: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract solid wall mask from no_stroke or stroke guides."""
        # Prefer no_stroke
        if "no_stroke" in rectified_guides:
            guide = rectified_guides["no_stroke"]
            if guide.shape[2] >= 4:
                return (guide[:, :, 3] > self.wall_alpha_thresh).astype(np.uint8) * 255
            gray = cv2.cvtColor(guide[:, :, :3], cv2.COLOR_BGR2GRAY)
            return (gray > 0).astype(np.uint8) * 255

        # Secondary: stroke guide
        if "stroke" in rectified_guides:
            guide = rectified_guides["stroke"]
            if guide.shape[2] >= 4:
                return (guide[:, :, 3] > self.wall_alpha_thresh).astype(np.uint8) * 255

        # Fallback if only centerline exists
        if "center_line" in rectified_guides:
            guide = rectified_guides["center_line"]
            if guide.shape[2] >= 4:
                b, g, r, a = cv2.split(guide)
                is_red = ((r.astype(int) - g.astype(int) > self.red_diff_thresh) & (r > self.red_min_thresh) & (a > self.wall_alpha_thresh)).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                return cv2.dilate(is_red, kernel)

        raise ValueError("No valid guide format available to extract wall mask.")

    def extract_raw_centerline(self, rectified_guides: Dict[str, np.ndarray], wall_mask: np.ndarray) -> np.ndarray:
        """Extract raw 1D wall centerline from center_line guide or skeletonize wall mask."""
        if "center_line" in rectified_guides:
            guide = rectified_guides["center_line"]
            if guide.shape[2] >= 4:
                b, g, r, a = cv2.split(guide)
                is_red = ((r.astype(int) - g.astype(int) > self.red_diff_thresh) & (r > self.red_min_thresh) & (a > self.wall_alpha_thresh)).astype(np.uint8) * 255
                if np.sum(is_red > 0) > 100:
                    return is_red

        # Fallback: compute morphological skeleton of wall_mask
        return self._morphological_skeleton(wall_mask)

    def _morphological_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Standard morphological skeleton extraction."""
        skel = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = mask.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            subset = cv2.subtract(eroded, opened)
            cv2.bitwise_or(skel, subset, skel)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skel

    def bridge_doorways_constrained(
        self,
        raw_centerline: np.ndarray,
        wall_mask: np.ndarray,
    ) -> np.ndarray:
        """Bridge doorway gaps in centerline strictly constrained within the physical wall envelope.

        Applies directional 1D morphological closing along horizontal and vertical axes,
        masking the resulting bridged segments so they cannot bleed into room interiors.
        """
        k_w = self.door_gap_px
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_w))

        closed_h = cv2.morphologyEx(raw_centerline, cv2.MORPH_CLOSE, k_h)
        closed_v = cv2.morphologyEx(raw_centerline, cv2.MORPH_CLOSE, k_v)
        axial_bridges = cv2.bitwise_or(closed_h, closed_v)

        # Constraint: Bridged pixels MUST lie within the solid wall envelope
        constrained_bridge = cv2.bitwise_and(axial_bridges, wall_mask)
        bridged_centerline = cv2.bitwise_or(raw_centerline, constrained_bridge)

        # Pin-hole cleanup (3x3 closing)
        k_pin = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bridged_centerline = cv2.morphologyEx(bridged_centerline, cv2.MORPH_CLOSE, k_pin)

        return bridged_centerline

    def extract_exterior_envelope(self, wall_mask: np.ndarray) -> np.ndarray:
        """Extract solid exterior footprint envelope of the entire apartment."""
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        envelope_mask = np.zeros_like(wall_mask)
        if contours:
            # Union of exterior wall contours filled solid
            cv2.drawContours(envelope_mask, contours, -1, 255, -1)
        return envelope_mask

    def extract_rooms(
        self,
        wall_mask: np.ndarray,
        envelope_mask: np.ndarray,
        human_labels: Optional[Dict | List] = None,
    ) -> List[TopologicalRoomNode]:
        """Extract closed room cycles from the interior footprint."""
        interior = cv2.bitwise_and(envelope_mask, cv2.bitwise_not(wall_mask))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interior)

        valid_indices = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_room_area_px:
                valid_indices.append(i)

        # Sort room indices by area descending (Room 1 is largest)
        valid_indices.sort(key=lambda idx: stats[idx, cv2.CC_STAT_AREA], reverse=True)

        rooms: List[TopologicalRoomNode] = []
        for rank, idx in enumerate(valid_indices):
            room_id = rank + 1
            area = float(stats[idx, cv2.CC_STAT_AREA])
            cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
            bbox = (
                int(stats[idx, cv2.CC_STAT_LEFT]),
                int(stats[idx, cv2.CC_STAT_TOP]),
                int(stats[idx, cv2.CC_STAT_WIDTH]),
                int(stats[idx, cv2.CC_STAT_HEIGHT]),
            )
            mask_i = (labels == idx).astype(np.uint8) * 255

            # Extract clean room perimeter contour
            cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = cnts[0] if cnts else np.empty((0, 1, 2), dtype=np.int32)

            # Assign human label if provided, else neutral Room ID (zero heuristic hallucination)
            label = self._resolve_room_label(room_id, mask_i, (cx, cy), human_labels)

            rooms.append(
                TopologicalRoomNode(
                    room_id=room_id,
                    label=label,
                    mask=mask_i,
                    contour=contour,
                    centroid=(cx, cy),
                    bbox=bbox,
                    area_px=area,
                )
            )

        return rooms

    def _resolve_room_label(
        self,
        room_id: int,
        room_mask: np.ndarray,
        centroid: Tuple[float, float],
        human_labels: Optional[Dict | List],
    ) -> str:
        """Resolve room label from human annotations or default to neutral Room {id}."""
        if not human_labels:
            return f"Room {room_id}"

        # 1. Direct ID dictionary: {"1": "Living Room", 2: "Bedroom"}
        if isinstance(human_labels, dict):
            str_id = str(room_id)
            if str_id in human_labels:
                return str(human_labels[str_id])
            if room_id in human_labels:
                return str(human_labels[room_id])

            # Spatial point dict: {"Living Room": [x, y], "Bedroom": [x, y]}
            h, w = room_mask.shape[:2]
            for lbl, pt in human_labels.items():
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= px < w and 0 <= py < h and room_mask[py, px] > 0:
                        return str(lbl)

        # 2. List of point annotations: [{"label": "Kitchen", "point": [x, y]}]
        elif isinstance(human_labels, list):
            h, w = room_mask.shape[:2]
            for item in human_labels:
                if isinstance(item, dict) and "label" in item and "point" in item:
                    pt = item["point"]
                    if len(pt) >= 2:
                        px, py = int(round(pt[0])), int(round(pt[1]))
                        if 0 <= px < w and 0 <= py < h and room_mask[py, px] > 0:
                            return str(item["label"])

        return f"Room {room_id}"

    def load_human_labels_for_unit(self, unit_key: str) -> Optional[Dict | List]:
        """Optionally load human label annotations from guide_dir if present."""
        guide_dir = Path(self.params.get("guide_dir", "human_semantic_guide"))
        if not guide_dir.is_dir():
            return None

        # Check for <unit_key>_labels.json or <unit_key>_labels.yaml
        for p in guide_dir.glob("*_labels.*"):
            from src.v5_semi_automated.rectification import normalize_unit_key
            if normalize_unit_key(p.stem) == unit_key or unit_key in p.stem.lower():
                import json
                try:
                    if p.suffix.lower() == ".json":
                        with p.open("r", encoding="utf-8") as f:
                            return json.load(f)
                    elif p.suffix.lower() in (".yaml", ".yml"):
                        with p.open("r", encoding="utf-8") as f:
                            return yaml.safe_load(f)
                except Exception:
                    pass
        return None

    def process(
        self,
        rectified_guides: Dict[str, np.ndarray],
        unit_key: str = "",
        human_labels: Optional[Dict | List] = None,
    ) -> TopologyResult:
        """Execute Phase 2 topological extraction and door bridging."""
        notes: List[str] = []

        # Optional human label annotations
        if human_labels is None:
            human_labels = self.load_human_labels_for_unit(unit_key)
        if human_labels:
            notes.append(f"Loaded human label annotations for unit '{unit_key}'.")
        else:
            notes.append("No human labels provided; defaulting to neutral Room IDs (Room 1, Room 2, ...).")

        # 1. Wall Mask Extraction
        wall_mask = self.extract_wall_mask(rectified_guides)
        notes.append(f"Extracted solid wall mask ({np.sum(wall_mask > 0)} wall pixels).")

        # 2. Raw Centerline Extraction
        raw_centerline = self.extract_raw_centerline(rectified_guides, wall_mask)
        notes.append(f"Extracted raw centerline ({np.sum(raw_centerline > 0)} stroke pixels).")

        # 3. Constrained Axial Doorway Bridging
        bridged_centerline = self.bridge_doorways_constrained(raw_centerline, wall_mask)
        bridged_pixels = np.sum(bridged_centerline > 0) - np.sum(raw_centerline > 0)
        notes.append(f"Constrained axial closing bridged {bridged_pixels} centerline pixels across door openings.")

        # 4. Exterior Building Envelope
        envelope_mask = self.extract_exterior_envelope(wall_mask)
        notes.append(f"Extracted building envelope ({np.sum(envelope_mask > 0)} total footprint pixels).")

        # 5. Extract Closed Room Cycles
        rooms = self.extract_rooms(wall_mask, envelope_mask, human_labels=human_labels)
        notes.append(f"Identified {len(rooms)} closed topological room cycles.")

        return TopologyResult(
            unit_key=unit_key,
            wall_mask=wall_mask,
            centerline_mask=bridged_centerline,
            exterior_envelope_mask=envelope_mask,
            rooms=rooms,
            notes=notes,
        )
