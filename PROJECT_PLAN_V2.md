# Iteration 2 — True 2D Blueprint Reconstruction Engine

Iteration 1 (`src/v1_baseline/`) segmented rooms directly on the perspective render. That kept camera skew, traced furniture, and anchored to extruded wall tops. Iteration 2 (`src/v2_reconstruction/`) flattens the render first, then rebuilds each room as an orthogonal rectangle and draws a clean 2D CAD-style plan.

Run with `python main.py --version v2` (the default). `--version v1` still runs the archived baseline.

## Phase 1: Homography and perspective correction (`homography.py`)

**Objective:** Un-warp the 3D render into a strictly orthographic top-down projection before segmentation.

- Detect the outermost quadrilateral boundary of the apartment with OpenCV edge detection and `cv2.approxPolyDP`.
- Compute the homography matrix with `cv2.getPerspectiveTransform`.
- Apply `cv2.warpPerspective` to flatten the canvas. Every downstream module processes this flattened array.

## Phase 2: Floor-seam segmentation (`segmentation.py`)

**Objective:** Extract walkable floor space by targeting wall bases instead of extruded wall tops.

- Run FastSAM on the flattened image.
- Filter masks toward the high-gradient seams where the floor meets the wall base, avoiding the offset caused by 3D wall height.

## Phase 3: Rectilinear room fitting (`geometry.py`)

**Objective:** Replace contour tracing, which hugs furniture, with architectural rectangles.

- Convert floor masks into minimum-area rectangles with `cv2.minAreaRect`.
- For complex rooms such as L-shapes, fit the union of the largest inscribed overlapping rectangles.
- Snap every rectangle vertex to one global grid so adjacent rooms share straight, orthogonal, single-line walls.

## Phase 4: Heuristic scale prior engine (`scale.py`)

**Objective:** Establish real-world scale when explicit OCR text is missing.

- Run OCR first. If no dimensions are found, run a lightweight detector (for example YOLO11n or GroundingDINO) to find standard architectural anchors.
- Anchor targets: kitchen countertops ($60\text{ cm}$), interior doorways ($80\text{ cm}$), bathtubs ($150\text{ cm}$).
- Consensus: compute a pixel-to-cm ratio for each detected anchor box and use the median as the global scale $S_{\text{final}}$.

## Phase 5: Synthetic blueprint generation (`visualizer.py` and `main.py`)

**Objective:** Produce a clean 2D CAD-style floor plan detached from the original render.

- Start from a blank white canvas matching the flattened image size.
- Draw the fitted rectangles as thick solid black lines for walls.
- Fill each room with its label, parsed or heuristic dimensions, and relative area percentage.
- Export `result_v2.json` with rectilinear vertex coordinates and standardized areas.

## Status

| Phase | Module | Status |
| --- | --- | --- |
| 1 | `homography.py` | Done (QA: `scripts/v2_test_homography.py`) |
| 2 | `segmentation.py` | Done (QA: `scripts/v2_test_segmentation.py`) |
| 3 | `geometry.py` | Stub |
| 4 | `scale.py` | Stub |
| 5 | `visualizer.py` | Stub |

Implementation proceeds one phase at a time; each phase starts only after the previous one is approved.

## Execution Report

### Phase 1: Homography and perspective correction

**Deliverables**

- `src/v2_reconstruction/homography.py`: `PerspectiveCorrector.correct(image)` returns a `HomographyResult` (flattened image, 3×3 homography, source and target quads, footprint contour, `applied` flag).
- `configs/config.yaml`: new `homography:` section; every threshold is config-driven.
- `scripts/v2_test_homography.py`: QA panels in `data/output/debug_v2/<stem>/` (`01_boundary_quad.png`, `02_flattened.png`, `03_flattened_check.png`) and before/after skew metrics.

**Pipeline**

1. Footprint: remove the near-white canvas connected to the image border (LAB lightness ≥ 250, chroma ≤ 8), add Canny edges inside the footprint, then close and open. Keep the largest external contour.
2. Boundary quad: simplify the outline with `cv2.approxPolyDP`. Pick the longest roughly horizontal segment in the top half and in the bottom half, and the longest roughly vertical segment in the left half and in the right half (±25° tolerance). Refit each wall with `cv2.fitLine` (Huber loss) on the contour points in a narrow band, then intersect adjacent lines to get the corners TL, TR, BR, BL.
3. Homography: `cv2.getPerspectiveTransform` maps the quad to a rectangle whose width and height are the longer of each pair of opposite sides, which keeps the original pixel scale.
4. Canvas fit: transform the whole footprint contour and translate it so the output holds everything, including balconies and bump-outs outside the quad, plus a 20 px margin.
5. Warp: `cv2.warpPerspective` with a white border.
6. Safety check: if the quad is not convex, or the warp would change the footprint area by more than 1.6×, skip the warp and only crop. If fewer than four wall segments are found, fall back to a convex-hull quad.

**Results**

| Render | Outline skew before → after | Wall-quad skew before → after | Output size |
| --- | --- | --- | --- |
| heritage towers (593 sq ft) | 4.00° → 0.71° | 3.05° → 0.02° | 773×723 |
| highlandlux citadel (623 sq ft) | 0.38° → 0.40° | 0.08° → 0.08° | 661×650 |
| limestone ranch (625 sq ft) | 2.74° → 0.35° | 2.63° → 0.01° | 784×763 |

- Outline skew is the length-weighted average lean of every footprint outline segment from its nearest axis. It is measured the same way on the input and on the flattened output, so it is an independent check.
- Wall-quad skew is the average lean of the four detected exterior wall sides.
- The homography was applied on all three renders. The safety check did not trigger.

**Insights**

- **The literal spec approach fails on real footprints.** The first version followed the spec exactly: take the convex hull and reduce it to four points with `approxPolyDP`. On L-shaped or notched apartments (limestone's balcony wing, highland's entry bump-out) the hull quad cut diagonally across the missing corner. The resulting homography would have sheared the plan instead of straightening it. Heritage worked only because its footprint is close to a trapezoid.
- **Exterior walls are the right geometric primitive.** In a Manhattan-world building the dominant outer walls form two pairs of parallel lines. Under a pinhole projection those pairs are exactly what a homography needs to become axis-aligned again. Using the longest wall on each side is also robust to protrusions, because a balcony is shorter than the main wall it sits on.
- **Size the canvas from the full footprint, not the quad.** The quad only covers the main body of the apartment. If the target rectangle were used as the canvas, balconies and bump-outs would be cropped away. Fitting the canvas to the warped footprint keeps them.
- **A self-consistent QA metric can hide failures.** The first metric re-detected the same quad on the output and reported near-zero skew even when the warp was visibly wrong. Measuring every outline segment the same way on the input and the output exposed the problem. Visual panels were still needed to confirm the result.
- **Near-orthographic renders pass through almost unchanged.** Highland was already top-down (0.38° skew), so the homography is close to the identity, as it should be. The stage is safe to run unconditionally.
- **Residual skew is an extrusion artifact, not a homography error.** The remaining 0.35–0.71° outline skew comes from the extruded 3D wall tops. They sit on a different plane from the floor, so no single plane-to-plane homography can flatten both. Phase 2's plan to target the floor-wall seams, not the wall tops, is the right answer here.

**Known limitations**

- It assumes one dominant rectangle-like outer shell. Plans made of several disconnected units, or with strongly rotated wings, would need line clustering (for example with a Hough transform or vanishing-point estimation) instead of taking the longest segment per side.
- The homography flattens the floor plane only approximately, because the footprint outline includes the wall tops. Output pixels keep roughly the input pixel scale but carry no real-world units yet; converting to real-world units is Phase 4.
- The canvas detection relies on the white render background (LAB lightness ≥ 250). Renders on coloured or textured backgrounds would need a different footprint detector.

### Phase 2: Floor-seam segmentation

**Deliverables**

- `src/v2_reconstruction/segmentation.py`: `FloorSeamSegmenter.segment(flattened)` returns one binary floor mask per room. A `SegmentationDebug` record on `segmenter.debug` exposes the footprint, wall ridges, barrier, room cells, seam edges, and the per-room source (`sam` or `cell`).
- `configs/config.yaml`: new `floor_segmentation:` section. FastSAM weights and inference settings are reused from `segmentation:`.
- `scripts/v2_test_segmentation.py`: runs Phase 1 then Phase 2 and writes `04_wall_ridges.png`, `05_room_cells.png`, `06_floor_sources.png` and `07_floor_masks.png` to `data/output/debug_v2/<stem>/`.

**Pipeline**

1. Footprint: the filled non-canvas silhouette of the flattened image.
2. Wall pixels: a pixel must be flat (lightness standard deviation ≤ 4 in a 3×3 window) and match one of two cues:
   - Shell colour: close in LAB to the median colour of the footprint's outer 15 px band, which is mostly exterior wall tops (a/b within 4, lightness within 25).
   - Brightness top-hat: at least 18 lightness levels brighter than its surroundings, inside a structure narrower than 37 px.
3. Wall cleanup: remove bright blobs wider than 37 px (counters, beds, tile floors). Keep only straight horizontal or vertical runs of at least 41 px, which is valid because Phase 1 made the walls axis-aligned.
4. Barrier: bridge doorways up to 49 px along both axes, close 45° door leaves, add the outer shell, and keep only barrier pieces connected to the shell.
5. Cells: the connected free space between barriers. Small regions fully enclosed by another room are merged into it. Furniture holes are filled. Regions joined by a passage narrower than 41 px are split with an opening plus a distance-map watershed. Cells thinner than a 12 px inradius are dropped.
6. Floor refinement: FastSAM everything-mode proposals that lie at least 85% inside a cell and cover at least 8% of it are merged. The result replaces the cell if it covers at least half of it; otherwise the room keeps the cell.

**Results**

| Render | Rooms | From FastSAM / from cell | Trimmed by FastSAM | Seam alignment (cell → final) |
| --- | --- | --- | --- | --- |
| heritage towers | 7 | 1 / 6 | 0.1% | 0.765 → 0.764 |
| highlandlux citadel | 8 | 1 / 7 | 0.7% | 0.657 → 0.662 |
| limestone ranch | 8 | 4 / 4 | 4.4% | 0.692 → 0.703 |

Seam alignment is the share of mask boundary pixels within 3 px of a strong colour gradient (top 20% of LAB Sobel magnitude inside the footprint).

- Heritage separates the bedroom, bathroom and laundry niche. The walk-in closet is merged with the open living and kitchen area through its doorway, and the balcony is cut into three cells by the railing, so 7 masks describe 5 real spaces.
- Highland separates the bathroom, walk-in closet, bedroom, balcony and three storage niches. The kitchen, hall and living room stay one open-plan region.
- Limestone separates the bedroom, walk-in closet, bathroom, lower storage closet, and both balcony-side niches. The living room, kitchen, hallway and balcony form one region, the coffee-table patch is a false room, and the small upper storage shelf falls below the minimum room area.

**Insights**

- **Colour alone cannot tell walls from floors.** Measured LAB values showed that highland's bathroom tile (L≈230, chroma≈2) and balcony decking (L≈208, chroma≈0) share the wall-top palette. A global colour threshold, as in Iteration 1, either eats white floors or misses walls.
- **Width and flatness separate what colour cannot.** Wall tops and faces are thin and untextured. Floors are wide, and bedding, curtains and decor are shaded or textured. A local flatness test plus a thickness limit removed most furniture false positives. The flatness window had to shrink from 7×7 to 3×3, because highland's wall tops are only about 8 px wide and a larger window straddles both edges.
- **The render's own shell calibrates the wall colour.** Sampling the exterior band gives each render its wall colour (grey in heritage, white in limestone, cream in highland) with no manual tuning, and it catches wall faces, which share that colour.
- **Phase 1 pays off here.** Because walls are axis-aligned after flattening, straight-run filtering and axis-aligned doorway bridging become valid. They remove curved and diagonal furniture edges that a plain top-hat kept.
- **Doorway bridging needs a clean wall network first.** With noisy ridges, 49 px bridging welded curtains, text and beds to the walls, so a "connected to the shell" filter kept everything. The flatness and straight-run filters had to come first.
- **Carving wall faces with FastSAM failed.** The first design removed thin proposals hugging walls as "wall faces". In practice FastSAM rarely proposes faces as separate masks, and faces of tall walls on the far side are 50–75 px deep. The accepted strips were TV consoles, shelving and counters. The final design builds floors from large proposals instead, and wall faces are handled by the shell-colour wall cue.
- **The seam metric barely moves, and that is expected.** Since wall faces are now part of the barrier, cell boundaries already sit at the wall base. FastSAM changes the final masks by only 0.1–4.4%. In this design FastSAM refines the floor and is not the main segmenter.

**Known limitations**

- Wide openings are not closed. Limestone's balcony door and the kitchen passage merge the balcony, living room, kitchen and hallway into one region, highland's kitchen, hall and living room are one region, and heritage's walk-in closet leaks into the living area. Splitting open-plan areas needs a semantic cue such as a floor-material change, which fits naturally into Phase 3.
- Balcony railings are straight bright runs, so heritage's balcony is cut into pieces. Phase 3 should merge adjacent cells separated only by thin, low-contrast barrier.
- A patch of floor enclosed by a bright rug that also touches a wall recess is not merged back (limestone's coffee-table area stays its own region). The enclosed-region merge only handles rings that close on themselves.
- Masks follow furniture that touches the walls (beds, counters, bathtubs), so outlines are notched. Rectilinear fitting in Phase 3 is expected to remove these notches.
- Thresholds were tuned on three renders from two render styles. A larger labelled set is needed before these values can be trusted.
