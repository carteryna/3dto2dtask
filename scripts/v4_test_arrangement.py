"""Visual verification script for Iteration 4 Phase 3: Planar Wall-Graph Line Arrangement."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v4_cad_arrangement.arrangement import PlanarWallArranger
from src.v4_cad_arrangement.calibration import ManhattanCalibrator
from src.v4_cad_arrangement.objects import ObjectDetector


def draw_label(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, text_color, thick, cv2.LINE_AA)


# Distinct pastel colors for face rendering (BGR)
FACE_COLORS = [
    (240, 200, 160),  # Soft Light Blue
    (180, 230, 180),  # Soft Mint
    (200, 180, 240),  # Soft Rose / Lavender
    (170, 230, 240),  # Soft Yellow
    (230, 190, 220),  # Soft Purple
    (200, 230, 200),  # Soft Sage
    (240, 220, 180),  # Soft Sky
    (190, 210, 240),  # Soft Peach
    (220, 220, 170),  # Soft Aqua
    (210, 180, 210),  # Soft Mauve
    (180, 240, 220),  # Soft Lemon
    (230, 210, 240),  # Soft Lilac
    (210, 240, 190),  # Soft Lime
    (240, 180, 190),  # Soft Coral
    (190, 240, 240),  # Soft Gold
    (220, 190, 180),  # Soft Blue-Grey
]


def test_arrangement():
    data_dir = PROJECT_ROOT / "inout_data"
    output_dir = PROJECT_ROOT / "data" / "output" / "debug_v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(data_dir.glob("*.webp"))
    if not input_paths:
        print("No .webp files found in inout_data/")
        return

    calibrator = ManhattanCalibrator()
    detector = ObjectDetector()
    arranger = PlanarWallArranger()

    print(f"=== Testing Iteration 4 Phase 3: Planar Wall-Graph Line Arrangement ({len(input_paths)} images) ===")

    for path in input_paths:
        stem = path.stem
        out_stem_dir = output_dir / stem
        out_stem_dir.mkdir(parents=True, exist_ok=True)

        rectified_path = out_stem_dir / "03_rectified_floor.png"
        if rectified_path.is_file():
            rectified_img = cv2.imread(str(rectified_path))
        else:
            orig_img = cv2.imread(str(path))
            calib_res = calibrator.calibrate(orig_img)
            rectified_img = calib_res.rectified_image
            cv2.imwrite(str(rectified_path), rectified_img)

        # Run Phase 2 Object Detection
        obj_res = detector.detect(rectified_img)

        # Run Phase 3 Planar Wall Arrangement
        arr_res = arranger.arrange(rectified_img, object_result=obj_res)

        print(f"\n--- {stem} ---")
        for note in arr_res.notes:
            print(f"  [Note] {note}")

        # Check disjointness / zero overlaps
        max_overlap_area = 0.0
        n_faces = len(arr_res.faces)
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                inter = arr_res.faces[i].intersection(arr_res.faces[j]).area
                if inter > max_overlap_area:
                    max_overlap_area = inter
        print(f"  Max mutual overlap area between any two room faces: {max_overlap_area:.4f} px^2")

        print(f"  Retained {n_faces} room faces:")
        for idx, face in enumerate(arr_res.faces):
            b = [round(float(v), 1) for v in face.bounds]
            print(f"    - Face {idx:02d}: area={face.area:.0f} px^2, bounds={b}")

        # -------------------------------------------------------------
        # Render 06_wall_centerlines.png
        # -------------------------------------------------------------
        vis_lines = rectified_img.copy()

        # Draw exterior envelope
        env_coords = np.array(arr_res.exterior_envelope.exterior.coords, dtype=np.int32)
        cv2.polylines(vis_lines, [env_coords], isClosed=True, color=(40, 200, 40), thickness=3)

        # Draw interior wall segments
        for wall in arr_res.wall_segments:
            if wall.wall_type == "exterior":
                continue
            p1 = (int(round(wall.start[0])), int(round(wall.start[1])))
            p2 = (int(round(wall.end[0])), int(round(wall.end[1])))
            color = (240, 150, 30) if wall.orientation == "horizontal" else (40, 40, 230)
            cv2.line(vis_lines, p1, p2, color, 2, cv2.LINE_AA)
            cv2.circle(vis_lines, p1, 4, (0, 0, 0), -1)
            cv2.circle(vis_lines, p1, 2, color, -1)
            cv2.circle(vis_lines, p2, 4, (0, 0, 0), -1)
            cv2.circle(vis_lines, p2, 2, color, -1)

        draw_label(vis_lines, "EXTERIOR ENVELOPE", (30, 30), (40, 180, 40))
        draw_label(vis_lines, "HORIZONTAL WALLS", (30, 55), (240, 140, 20))
        draw_label(vis_lines, "VERTICAL WALLS", (30, 80), (30, 30, 220))

        p6 = out_stem_dir / "06_wall_centerlines.png"
        cv2.imwrite(str(p6), vis_lines)
        print(f"  Saved {p6.name}")

        # -------------------------------------------------------------
        # Render 07_planar_faces.png
        # -------------------------------------------------------------
        vis_faces = rectified_img.copy()
        face_overlay = np.zeros_like(vis_faces)

        for idx, face in enumerate(arr_res.faces):
            color = FACE_COLORS[idx % len(FACE_COLORS)]
            coords = np.array(face.exterior.coords, dtype=np.int32)
            cv2.fillPoly(face_overlay, [coords], color)
            # Outline
            cv2.polylines(vis_faces, [coords], isClosed=True, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)

        vis_faces = cv2.addWeighted(vis_faces, 0.60, face_overlay, 0.40, 0)

        # Draw labels at interior representative points
        for idx, face in enumerate(arr_res.faces):
            rep = face.representative_point()
            rx, ry = int(round(rep.x)), int(round(rep.y))
            color = FACE_COLORS[idx % len(FACE_COLORS)]
            darker_col = tuple(max(0, int(c * 0.75)) for c in color)
            draw_label(vis_faces, f"Face {idx} ({face.area:.0f} px)", (rx - 25, ry), darker_col)

        p7 = out_stem_dir / "07_planar_faces.png"
        cv2.imwrite(str(p7), vis_faces)
        print(f"  Saved {p7.name}")


if __name__ == "__main__":
    test_arrangement()
