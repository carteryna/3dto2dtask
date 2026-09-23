# 3D Floor Plan to 2D Room Layout

Exploratory pipeline that turns a top-down 3D apartment render into regularized room polygons, relative pixel areas, and square-footage estimates when a printed scale is present.

The design is wall-first. Floor textures, rugs, and furniture break color segmentation, so the structural path closes the wall network before any room is labeled. A vision model proposes floor regions, and classical watershed fills the seeds that the model misses. Shapely then snaps the boundaries toward a Manhattan plan.

## 🔬 R&D Methodology & Iterative Progression

To solve the 3D Floor Plan to 2D Layout task, this project was developed across two distinct R&D stages:

### Stage 1: Exploratory Baseline & Diagnostic Phase (Iteration 1)

- **Objective:** Build an exploratory pipeline using classical CV (`OpenCV`), zero-shot vision-foundation segmentation (`FastSAM`), vector processing (`Shapely`), and scale text parsing (`EasyOCR`)[cite: 1, 2, 3].
- **Key Feasibility Insights Gained:**
  1. *Perspective Distortion:* Direct pixel masking on 3D renders retains camera skew; true 2D plans require homography warping prior to segmentation[cite: 4].
  2. *Wall Top vs. Floor Seam:* Isolating light-gray/white wall tops causes a spatial offset due to 3D wall height extrusion[cite: 1, 4]. True boundaries must target floor-wall base seams[cite: 4].
  3. *Contour Tracing vs. Rectilinear Fitting:* Tracing pixel contours creates jagged cutouts around furniture and rugs (e.g., 12 fragmented polygons on `limestone ranch`)[cite: 5, 6]. 2D blueprints require fitting orthogonal bounding boxes (`cv2.minAreaRect`).
  4. *Heuristic Scale Anchoring:* OCR callouts provide high accuracy when present (10.683 px/ft on `heritage towers`)[cite: 1], but missing text requires fallback scale anchors using standard architectural priors (60cm countertops, 80cm doorways, 150cm bathtubs).

### Stage 2: True 2D Blueprint Reconstruction Engine (Iteration 2 — Target Execution)

- **Objective:** Pivot from 3D perspective masking to synthetic 2D CAD blueprint generation.
- **Core Architecture Changes:**
  - **Homography Matrix:** Un-warp 3D renders into flat, orthographic top-down projections (`cv2.getPerspectiveTransform`).
  - **Floor-Seam Base Detection:** Anchor room perimeters directly to the walkable floor-wall junction.
  - **Rectilinear Room Fitting:** Replace contour simplification with Manhattan grid-aligned rectangular box fitting.
  - **Synthetic Blueprint Rendering:** Generate clean, publication-ready black-and-white 2D vector layouts filled with room labels, scale ratios, and area metrics.

### 📐 Heuristic Scale Prior Engine (Fallback Scale System)

When OCR dimension parsing returns `None` (as in `highlandlux-citadel` and `limestone ranch`)[cite: 2, 3], the pipeline falls back to an Architectural Heuristic Prior System:

- **Standard Anchor Dimensions:**
  - **Kitchen Countertop Depth:** $60\text{ cm}$ ($24\text{ in}$)
  - **Standard Doorway Width:** $80\text{ cm}$ ($32\text{ in}$)
  - **Standard Bathtub Length:** $150\text{ cm}$ ($60\text{ in}$)
  - **Toilet Bowl Width:** $40\text{ cm}$ ($15.5\text{ in}$)
- **Consensus Scale Calculation:**
  For detected anchor objects, compute candidate scale ratios:
  $$S_i = \frac{\text{Standard Real Dimension (cm)}}{\text{Pixel Dimension (px)}}$$
  The global image scale $S_{\text{final}}$ is set to $\text{Median}(S_1, S_2, \dots, S_n)$ to eliminate outlier detections.

## Architecture

```text
Input render
    │
    ├─ Wall extractor
    │    HSV/LAB wall tops, morphological door-gap closing, distance-transform seeds
    │
    ├─ Room segmenter
    │    FastSAM point prompts, watershed fallback, overlap resolution
    │
    ├─ Geometry regularizer
    │    contour → polygon, Douglas–Peucker, orthogonal snap, shared-edge cleanup
    │
    ├─ OCR scale parser
    │    EasyOCR callouts, feet-and-inches parsing, label-to-room fusion
    │
    └─ Metrics, annotated overlay, result.json
```

| Stage | Module | Role |
| --- | --- | --- |
| Walls and seeds | `src/preprocessing.py` | Isolate light wall tops and close 45° door gaps |
| Room masks | `src/segmentation.py` | FastSAM proposals with a seeded-watershed fallback |
| Vectors | `src/geometry.py` | Simplify contours and snap near-axis edges |
| Scale and labels | `src/ocr.py` | Parse dimensions and attach them to room polygons |
| Scores | `src/metrics.py` | IoU, boundary IoU, relative area error, orthogonality |
| Overlay | `src/visualizer.py` | Final color-coded QA image |
| CLI | `main.py` | Batch or single-image run and JSON export |

`configs/config.yaml` holds the wall HSV range, door-closing kernel, FastSAM settings, simplification ratio, orthogonal tolerance, and OCR languages. The container defaults to CPU.

## Assumptions and inputs

The three evaluation renders are orthographic or near-orthographic cutaways, not perspective photos.

- Wall tops are light gray or white. Inner wall faces are darker and are not the segmentation cue.
- Hardwood, carpet, tile, and decking vary enough that color thresholding over-segments. The wall mask, not the floor color, defines the cell.
- Open doors at about 45° break the perimeter. A morphological close at 0°, 45°, 90°, and 135° bridges the narrow gaps. Wide cased openings stay open, so an open-plan living and kitchen space remains one cell.
- Furniture and light rugs can still punch holes or pull a boundary off the wall. Douglas–Peucker and the 10° orthogonal snap remove the small jogs. Edges steeper than 10° stay diagonal.
- Only some sheets print dimensions. Heritage has `13'3" x 13'3"` and `11'9" x 10'9"`. Highland and limestone do not, so those runs stay in relative-area fallback.
- There is no pixel-level ground truth in this bundle. Region scores below compare the hybrid masks with the classical watershed masks from the same seeds.

## Quickstart

From the repository root, with Docker running:

```bash
docker compose run --rm cv-floorplan python main.py \
  --input_dir /app/inout_data \
  --output_dir /app/data/output \
  --method hybrid
```

One image:

```bash
docker compose run --rm cv-floorplan python main.py \
  --input "/app/inout_data/heritage towers_a1_1 Bed 1 Bath 593 Sq. Ft..webp" \
  --output_dir /app/data/output \
  --method hybrid
```

`--method watershed` skips FastSAM and regularizes the classical watershed masks only.

The first run downloads `models/FastSAM-s.pt` and the EasyOCR English weights into `models/`. Later runs reuse those files through the Compose mount.

Outputs:

- `data/output/result.json` — one schema object for a single image, or a list of those objects for a directory
- `data/output/annotated_<image_stem>.png` — filled polygons, outlines, room id, OCR label, relative area, and square feet when a scale exists
- `data/output/metrics_summary.json` — agreement with watershed and orthogonality

Phase debug panels from the earlier scripts remain under `data/output/debug/<image>/`.

### Result schema

```json
{
  "metadata": {
    "image_name": "heritage towers_a1_1 Bed 1 Bath 593 Sq. Ft..webp",
    "dimensions": {"width": 1140, "height": 855},
    "pipeline_method": "hybrid_fastsam_shapely",
    "scale_px_per_ft": 10.683,
    "fallback_mode": false,
    "total_detected_rooms": 7
  },
  "rooms": [
    {
      "room_id": 2,
      "label": "Room 2",
      "confidence": 0.35,
      "pixel_area": 129205,
      "relative_area_pct": 56.0,
      "estimated_sqft": 175.6,
      "polygon_vertices": [[150, 200], [450, 200], [450, 550], [150, 550]],
      "bbox": [150, 200, 450, 550]
    }
  ]
}
```

`scale_px_per_ft` is the median of `text-box width (px) / parsed width (ft)`. `estimated_sqft` is the nominal callout area (`width × height`) when a dimension is matched to that room, otherwise the pixel area converted with the plan scale. It is `null` in fallback mode. `bbox` is `[min_x, min_y, max_x, max_y]`.

## Evaluation

These renders have no hand-labeled room masks. IoU, boundary IoU, and relative area error measure agreement between the hybrid masks and the classical watershed masks grown from the same seeds. Boundary IoU uses a 6-pixel perimeter band. Orthogonality is the mean absolute deviation of each polygon edge from the nearest 0°/90° axis, in degrees. Watershed orthogonality is measured on the raw contour, before Manhattan snapping. Hybrid orthogonality is measured after snapping.

| Image | Mean IoU | Boundary IoU | RAE % | Watershed Δθ (°) | Hybrid + Shapely Δθ (°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| heritage towers_a1 | 0.948 | 0.876 | 4.60 | 22.53 | 14.77 |
| highlandlux-citadel | 1.000 | 1.000 | 0.00 | 22.48 | 16.98 |
| limestone ranch_santa fe | 0.933 | 0.847 | 5.58 | 22.97 | 14.18 |
| **Unweighted mean** | **0.960** | **0.907** | **3.39** | **22.66** | **15.31** |

Highland scores perfect agreement because every surviving room on that sheet fell back to watershed. Snapping cuts the mean axis error by about 7° against the raw watershed contours. Residual error is the edges steeper than the 10° snap tolerance: window cuts, open doorways, and rug boundaries.

## Failure modes and limitations

- **Open plans.** Highland's living, dining, and kitchen share one carpet and wide openings. Door-gap closing does not invent a wall, so they stay one polygon. Limestone's light rug still splits the living room.
- **Missing scale.** Highland has no text. Limestone's only text is the disclaimer. Both runs set `fallback_mode` and report relative area only.
- **Callout geometry.** The published scale uses the OCR box width, which is the width of the text, not the room. Nominal square footage on a matched room comes from the parsed `width × height` and is the figure to trust. Estimates on unlabeled rooms are approximate because the mask does not fill the architectural bounding box.
- **Labels.** Only text that is actually printed is attached. Heritage gets `Mech:`. Rooms without a name stay `Room N`.
- **Small cells.** Masks under 2,000 pixels are dropped, so some closets and the Highland bath do not appear.
- **45° doors and furniture.** A 15-pixel close bridges narrow leaves. Wider openings, curtains, and thick fixtures remain in the contour if they exceed the snap tolerance.

## Future R&D

- **Pose.** Estimate the ground plane with vanishing lines or a homography so slightly extruded renders land in a true top-down metric frame before area is computed.
- **Walls.** Replace the HSV wall prior with a trained wall segmenter, then keep the morphological close as a constraint rather than the primary detector.
- **Open plan.** Use door-swing detection and room-name language to split carpeted spaces that have no wall between them.
- **Scale.** Calibrate pixels-per-foot from the matched room's bounding box against the printed dimension, and fall back to a standard door width only when OCR finds nothing.
- **Takeoff API.** Serve `result.json` from a small backend: polygon in, flooring material and waste factor in, square footage and cut list out.
