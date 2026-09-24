"""Phase 4: scale from the unit area named in the filename."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import yaml
from shapely.geometry import Polygon

_SQFT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|square\s+feet|sqft|sq)\b",
    re.IGNORECASE,
)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class FixtureAnchor:
    kind: str
    pixel_span: float
    scale_px_per_ft: float


@dataclass
class ScaleResult:
    scale_px_per_ft: Optional[float]
    pixel_area: float
    sqft: Optional[float]
    source: str
    anchors: List[FixtureAnchor] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class AreaScaleEstimator:
    """Pixels per foot from filename square footage, with a fixture fallback.

    The filename scale is ``sqrt(pixel area / square feet)``, which is a
    linear scale in pixels per foot. When the name has no size, YOLOv8n
    supplies the scale from a toilet bowl width or a bathtub length. Gap
    widths are not used.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _default_config_path()
        with path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.pipeline = self.config["pipeline"]
        self.params = self.config["v3_scale"]
        self._model = None
        self.result: Optional[ScaleResult] = None

    def estimate(
        self,
        polygons: Sequence[Polygon],
        filename: str,
        image: np.ndarray | None = None,
    ) -> Optional[float]:
        """Return pixels per foot, or ``None`` when neither source is available."""
        pixel_area = float(sum(polygon.area for polygon in polygons))
        sqft = self.square_feet(filename)
        notes: list[str] = []
        anchors: list[FixtureAnchor] = []
        scale: Optional[float] = None
        source = "none"
        if sqft is not None and sqft > 0 and pixel_area > 0:
            scale = float(np.sqrt(pixel_area / sqft))
            source = "filename"
        elif pixel_area <= 0:
            notes.append("The room polygons have no area.")
        else:
            notes.append(f"No square footage in {filename!r}.")
            if image is None:
                notes.append("Fixture fallback needs the render.")
            else:
                scale, anchors, fixture_notes = self._fixture_scale(image)
                notes.extend(fixture_notes)
                source = "fixtures" if scale is not None else "none"
        self.result = ScaleResult(
            scale_px_per_ft=scale,
            pixel_area=pixel_area,
            sqft=sqft,
            source=source,
            anchors=anchors,
            notes=notes,
        )
        return scale

    @staticmethod
    def square_feet(filename: str) -> Optional[float]:
        """The last size written in the filename, such as ``593 Sq. Ft.`` or ``625sq``."""
        matches = _SQFT_PATTERN.findall(Path(filename).stem)
        if not matches:
            return None
        return float(matches[-1])

    def _fixture_scale(self, image: np.ndarray) -> tuple[Optional[float], list[FixtureAnchor], list[str]]:
        notes: list[str] = []
        try:
            model = self._load_yolo()
            names = {str(name).lower() for name in model.names.values()}
            if not any(name in {"bathtub", "bath tub", "tub"} or "bathtub" in name for name in names):
                notes.append("YOLOv8n has no bathtub class; the fallback uses toilets when they are detected.")
            detections = self._detect(image)
        except Exception as exc:
            return None, [], [f"YOLOv8n unavailable ({exc})."]
        anchors: list[FixtureAnchor] = []
        cm_per_foot = float(self.params["cm_per_foot"])
        for kind, span in detections:
            cm = float(self.params["toilet_width_cm"] if kind == "toilet" else self.params["bathtub_length_cm"])
            feet = cm / cm_per_foot
            if feet <= 0 or span <= 0:
                continue
            anchors.append(FixtureAnchor(kind=kind, pixel_span=float(span), scale_px_per_ft=float(span / feet)))
        if not anchors:
            notes.append("No toilet or bathtub was detected.")
            return None, [], notes
        scale = float(np.median([anchor.scale_px_per_ft for anchor in anchors]))
        return scale, anchors, notes

    def _detect(self, image: np.ndarray) -> list[tuple[str, float]]:
        model = self._load_yolo()
        names = {int(index): str(name).lower() for index, name in model.names.items()}
        toilet_ids = [index for index, name in names.items() if name == "toilet"]
        bathtub_ids = [
            index for index, name in names.items() if name in {"bathtub", "bath tub", "tub"} or "bathtub" in name
        ]
        wanted = toilet_ids + bathtub_ids
        if not wanted:
            return []
        results = model.predict(
            image,
            device=str(self.pipeline.get("device", "cpu")),
            conf=float(self.params["fixture_conf"]),
            classes=wanted,
            verbose=False,
        )
        found: list[tuple[str, float]] = []
        if not results or results[0].boxes is None:
            return found
        boxes = results[0].boxes
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        xyxy = boxes.xyxy.detach().cpu().numpy()
        for class_id, box in zip(classes, xyxy):
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            if int(class_id) in bathtub_ids:
                found.append(("bathtub", max(width, height)))
            elif int(class_id) in toilet_ids:
                found.append(("toilet", min(width, height)))
        return found

    def _load_yolo(self):
        if self._model is not None:
            return self._model
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import YOLO

        self._model = YOLO(str(self._resolve_weights()))
        return self._model

    def _resolve_weights(self) -> Path:
        configured = Path(str(self.params["weights"]))
        if not configured.is_absolute():
            configured = _project_root() / configured
        configured.parent.mkdir(parents=True, exist_ok=True)
        if configured.is_file() and configured.stat().st_size > 0:
            return configured
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(configured.name))
        if not downloaded.is_file():
            raise FileNotFoundError(f"Could not download YOLOv8n weights: {configured.name}")
        if downloaded.resolve() != configured.resolve():
            shutil.copyfile(downloaded, configured)
        return configured
