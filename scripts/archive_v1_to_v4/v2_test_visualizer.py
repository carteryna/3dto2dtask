#!/usr/bin/env python3
"""Visual QA for Iteration 2 Phase 5 blueprint rendering.

Run inside the container:

    python scripts/v2_test_visualizer.py

Writes ``11_blueprint.png`` under ``data/output/debug_v2/<image_stem>/``.
The CLI writes the same drawing as ``blueprint_<stem>.png`` plus ``result_v2.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import list_images, reconstruct_v2

INPUT_DIR = ROOT / "inout_data"
DEBUG_DIR = ROOT / "data" / "output" / "debug_v2"
CONFIG = ROOT / "configs" / "config.yaml"


def main() -> None:
    images = list_images(None, str(INPUT_DIR))
    print(f"Processing {len(images)} image(s) from {INPUT_DIR}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        blueprint, payload = reconstruct_v2(image, image_path.name, str(CONFIG))
        output_dir = DEBUG_DIR / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "11_blueprint.png"
        cv2.imwrite(str(path), blueprint)
        print(
            f"{image_path.name}: rooms={payload['metadata']['total_detected_rooms']} "
            f"scale={payload['metadata']['scale_px_per_cm']} px/cm -> {path}"
        )


if __name__ == "__main__":
    main()
