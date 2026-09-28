"""Visual and quantitative verification script for Iteration 5 Phase 1: Ground-Plane Rectification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.v5_semi_automated.rectification import (
    SemiAutomatedRectifier,
    normalize_unit_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Iteration 5 Phase 1 Ground-Plane Rectification.")
    parser.add_argument("--input", default=None, help="Path to single floor-plan image (.webp).")
    parser.add_argument("--input_dir", default="inout_data", help="Directory of floor-plan images.")
    parser.add_argument("--output_dir", default="data/output/debug_v5", help="Directory for debug outputs.")
    parser.add_argument("--config", default=None, help="Path to configs/config.yaml.")
    return parser.parse_args()


def overlay_alpha_rgba(bg_bgr: np.ndarray, fg_bgra: np.ndarray, alpha_weight: float = 0.8) -> np.ndarray:
    """Alpha-blend a BGRA foreground onto a BGR background."""
    out = bg_bgr.copy()
    if fg_bgra is None or fg_bgra.shape[2] < 4:
        return out
    h, w = bg_bgr.shape[:2]
    if fg_bgra.shape[:2] != (h, w):
        fg_bgra = cv2.resize(fg_bgra, (w, h), interpolation=cv2.INTER_NEAREST)

    alpha = (fg_bgra[:, :, 3].astype(np.float32) / 255.0) * alpha_weight
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    fg_bgr = fg_bgra[:, :, :3].astype(np.float32)
    bg_float = out.astype(np.float32)

    blended = fg_bgr * alpha_3d + bg_float * (1.0 - alpha_3d)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    output_base = PROJECT_ROOT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_paths = [Path(args.input)]
    else:
        input_dir = PROJECT_ROOT / args.input_dir
        input_paths = sorted(input_dir.glob("*.webp"))

    if not input_paths:
        print("No .webp images found to process.")
        sys.exit(1)

    rectifier = SemiAutomatedRectifier(args.config)
    print(f"=== Iteration 5 Phase 1: Ground-Plane Rectification ({len(input_paths)} assets) ===")

    results_summary = []

    for path in input_paths:
        stem = path.stem
        unit_key = normalize_unit_key(stem)
        unit_out_dir = output_base / stem
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        render_img = cv2.imread(str(path))
        if render_img is None:
            print(f"[Error] Could not read {path}")
            continue

        # Discover and load all guide formats for this unit
        guides = rectifier.load_guides_for_unit(path)
        guide_names = list(guides.keys())
        print(f"\n--- Processing: {stem} ---")
        print(f"  Unit Key: {unit_key}")
        print(f"  Found {len(guides)} guide formats: {guide_names}")

        # Execute Phase 1 Rectification
        res = rectifier.rectify(render_img, guides=guides, unit_key=unit_key)

        for note in res.notes:
            print(f"  [Note] {note}")

        # Gate 1 QA verification
        gate1_passed = res.skew_after_deg <= 1.0
        status_str = "PASS" if gate1_passed else "FAIL"
        print(f"  [QA Gate 1] Vertical Skew: {res.skew_before_deg:.2f}° -> {res.skew_after_deg:.2f}° [{status_str}]")

        # 1. Save Detected Vertical Lines on original render
        vis_lines = render_img.copy()
        for x1, y1, x2, y2 in res.vertical_lines:
            cv2.line(vis_lines, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (255, 120, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(unit_out_dir / "01_detected_vertical_lines.png"), vis_lines)

        # 2. Save Rectified Render
        cv2.imwrite(str(unit_out_dir / "02_rectified_render.png"), res.rectified_render)

        # 3. Save Rectified Guides
        for fmt, rect_guide in res.rectified_guides.items():
            cv2.imwrite(str(unit_out_dir / f"03_rectified_{fmt}.png"), rect_guide)

        # 4. Save Composite Overlays demonstrating synchronous registration
        # Overlay center_line if present, else stroke, else no_stroke
        overlay_fmt = "center_line" if "center_line" in res.rectified_guides else ("stroke" if "stroke" in res.rectified_guides else "no_stroke")
        if overlay_fmt in res.rectified_guides:
            composite = overlay_alpha_rgba(res.rectified_render, res.rectified_guides[overlay_fmt], alpha_weight=0.85)
            cv2.imwrite(str(unit_out_dir / "04_rectified_composite_overlay.png"), composite)

        # Plumb vertical verification overlay on rectified render
        vis_rect_lines = res.rectified_render.copy()
        for x1, y1, x2, y2 in res.warped_vertical_lines:
            cv2.line(vis_rect_lines, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(unit_out_dir / "05_rectified_plumb_lines.png"), vis_rect_lines)

        results_summary.append({
            "unit": stem,
            "in_shape": render_img.shape[:2],
            "out_shape": res.rectified_render.shape[:2],
            "skew_before": res.skew_before_deg,
            "skew_after": res.skew_after_deg,
            "passed": gate1_passed,
        })

    print("\n" + "=" * 80)
    print("PHASE 1 QA GATE 1 SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Unit':<35} | {'Input Dim':<12} | {'Output Dim':<12} | {'Skew Before':<12} | {'Skew After':<12} | {'Gate 1'}")
    print("-" * 80)
    all_passed = True
    for r in results_summary:
        in_dim = f"{r['in_shape'][1]}x{r['in_shape'][0]}"
        out_dim = f"{r['out_shape'][1]}x{r['out_shape'][0]}"
        gate_res = "PASS" if r['passed'] else "FAIL"
        if not r['passed']:
            all_passed = False
        print(f"{r['unit'][:35]:<35} | {in_dim:<12} | {out_dim:<12} | {r['skew_before']:>10.2f}° | {r['skew_after']:>10.2f}° | {gate_res}")
    print("=" * 80)
    if all_passed:
        print("ALL ASSETS PASSED QA GATE 1 (Vertical Skew <= 1.0°). Ready for Phase 2!")
    else:
        print("SOME ASSETS FAILED QA GATE 1. Review notes above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
