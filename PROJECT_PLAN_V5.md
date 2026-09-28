# Iteration 5 — Semi-Automated Architectural CAD Vectorization & Planar Graph Synthesis

## 1. Executive Summary & Paradigm Shift

Across Iterations 1 through 4, we explored fully automated Computer Vision (CV) and Machine Learning (ML) pipelines to convert 3D perspective marketing floor plans into 2D planar CAD layouts. The definitive post-mortem documented in `RND_HISTORY.md` established:
> **Automating visual perception on stylized 3D marketing renders has diminishing returns, whereas software is vastly superior at computational geometry, metric scaling, and CAD formatting.**

In modern 3D floor plan renders, interior furniture (sofas, beds, kitchen islands) occludes 30–50% of the floor, wall heights ($Z = H$) cast perspective shadows and occlude ground seams ($Z = 0$), and open floor plans (living/dining/kitchen) lack physical dividers. Automated line extractors conflate shadows with walls and fracture rooms into fragmented slivers.

### The Semi-Automated Strategy
We fundamentally pivot to a **Semi-Automated Architecture**:
1. **Human-in-the-Loop (30–60s Input):** A human designer provides a clean structural semantic guide in one of three standardized formats (central lines, solid wall bodies, or dual-perimeter stroked walls). This resolves 100% of perceptual ambiguities, furniture occlusions, and open-plan divisions.
2. **Computational Geometry Engine:** Software performs 1D vanishing-point pitch rectification, Manhattan axis snapping, planar graph polygonization (guaranteeing $0.0000\text{ px}^2$ overlap and $0\text{ px}$ gaps), conformal metric scaling, wall thickness synthesis, and export to GeoJSON, SVG, and high-resolution CAD blueprints.

---

## 2. Component Salvage & Algorithmic Analysis

Based on the empirical evidence across Iterations 1 through 4, we retain the proven "winning mechanics" and discard the failure modes:

| Component | Legacy Iteration | Evaluation & Decision | Role in Iteration 5 |
| :--- | :--- | :--- | :--- |
| **FastSAM / Semantic Segmentation** | Iteration 1 & 3 | **Discard for room perimeter extraction.** Hugs curved sofa cushions and bed frames; blind to beige carpet and white tile. | Repurposed strictly as an optional secondary layer for decoupled interior furniture symbols. |
| **Planar Line Graph Polygonization** | Iteration 4 | **Surviving Architectural Winner.** Shapely `unary_union` + `polygonize` mathematically guarantees **$0.0000\text{ px}^2$ mutual overlap** and **$0\text{ px}$ inter-room gaps**. | Core geometric engine to partition rectified walls into closed room polygons. |
| **Manhattan Snapping & Collinear Merging** | Iteration 3 & 4 | **Surviving Architectural Winner.** Clusters parallel segments and snaps near-axis lines ($\pm 10^\circ$) to exact $0^\circ / 90^\circ$ Manhattan axes. | Eliminates residual tracing wobbles and closes T-junctions and corners. |
| **1D Vanishing-Point Pitch Homography** | Iteration 4 | **Surviving Architectural Winner.** Avoids 4-point $Z=H$ floor shear. Corrects perspective vertical wall lean ($>90\%$ reduction) without shearing horizontal ground axes. | Phase 1 metric rectification of both raw render and human guide. |
| **Conformal Filename Scaling Engine** | Iteration 3 & 4 | **Surviving Architectural Winner.** $S = \sqrt{\frac{\sum A_{px}}{A_{sqft}}}$ is 100% deterministic, robust to font variance, and accurate across all test assets. | Phase 4 primary metric scale calculation. |
| **Fixture Co-occurrence Validator** | Iteration 2 & 4 | **Surviving Architectural Winner.** Standard bathtub ($150\text{ cm} \pm 10\%$) provides a topological sanity check. | Phase 4 secondary validation check. |
| **Floor Integrity Rule (Decoupled Layers)** | Iteration 4 | **Surviving Architectural Winner.** Floors are solid, continuous polygons; furniture is never carved out as holes. | Room polygons remain solid 2D manifold faces. |
| **Dual-Line Wall Blueprint Visualizer** | Iteration 2 & 4 | **Surviving Architectural Winner.** Produces publication-grade 2D CAD layouts with hatched walls, room dimensions, and labels. | Phase 5 rendering and vector export. |

---

## 3. Human Semantic Guide Specifications (`human_semantic_guide/`)

The human-in-the-loop provides structural annotations registered 1:1 to the original image dimensions. Iteration 5 natively supports three standardized formats:

```
human_semantic_guide/
├── *_semantic_center_line.png  (Format A: Pure Wall Centerlines in Red)
├── *_semantic_no_stroke.png    (Format B: Solid Wall Footprint Mask)
└── *_semantic_stroke.png       (Format C: Dual-Perimeter Wall Mask with Strokes)
```

1. **Format A: Central Line (`_semantic_center_line.png` / `_semantic_central_line.png`)**
   * *Description:* Red vector-like centerlines ($R > 150, R - G > 50$) drawn along the exact middle axes of all interior partition walls and exterior perimeter walls.
   * *Strength:* Provides direct 1D centerline topology without needing medial axis skeletonization.
2. **Format B: No Stroke (`_semantic_no_stroke.png` / `_sementic_no_stroke.png`)**
   * *Description:* Solid raster wall mask with clean interior cutouts for every functional room.
   * *Strength:* Exactly defines wall thickness and room interior boundaries; inverting the mask directly yields disjoint connected components.
3. **Format C: Dual Perimeter Line with Stroke (`_semantic_stroke.png`)**
   * *Description:* Wall body bordered by distinct dark/black boundary strokes along inner and outer perimeters.
   * *Strength:* Explicitly marks physical wall-face boundaries for dual-line architectural drafting.

---

## 4. End-to-End Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ INPUT: Raw 3D Perspective Render (.webp) + Human Semantic Guide (.png)            │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: GROUND-PLANE RECTIFICATION (rectification.py)                            │
│  - Vertical Vanishing Point (VP_v) estimation via RANSAC line intersection        │
│  - 1D pitch homography H_floor anchored to ground plane (Z=0)                     │
│  - Synchronous transformation of both Render and Guide into orthographic space    │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: TOPOLOGICAL EXTRACTION & AXIAL BRIDGING (topology.py)                     │
│  - Extraction of centerline graph from Guide Formats (A, B, or C)                 │
│  - Constrained axial morphological closing across doorway swings                  │
│  - Room seed generation and initial topological node identification               │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: MANHATTAN PLANAR GRAPH POLYGONIZATION (geometry.py)                      │
│  - Manhattan grid snapping (0° and 90° axis projection)                           │
│  - Collinear segment clustering (15 px tolerance) and T-junction snapping         │
│  - Shapely unary_union + polygonize: Guarantee 0.0000 px² overlap, 0 px gaps      │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: CONFORMAL METRIC SCALE & DIMENSIONAL ENGINE (scale.py)                   │
│  - Parse target square footage from filename (e.g. 593 Sq. Ft.)                   │
│  - Conformal scale S = sqrt(sum(A_px) / target_sqft) (cm/px)                      │
│  - Room dimensional spans formatted into architectural feet/inches (W'H" x W'H")  │
│  - Sanity check against standard fixtures (Bathtub ≈ 150 cm ± 10%)                │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: PRODUCTION CAD BLUEPRINT & VECTOR EXPORT (visualizer.py / export.py)     │
│  - Dual-line wall rendering with architectural hatch fills (exterior/interior)    │
│  - Centered room callouts: Room Name + Area (sq ft) + Dimensions (ft/in)          │
│  - Export: Standard GeoJSON, 2D Vector SVG, High-Res PNG CAD Blueprint            │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Mathematical Resolution of Past Critical Constraints

### 1. Avoiding Floor-Plane Shearing
* **Past Failure:** In Iteration 2, computing homography on elevated wall tops ($Z = H$) introduced severe parallax shear across the ground plane ($Z = 0$), skewing rectangular rooms into parallelograms.
* **Iteration 5 Solution:**
  1. We compute camera pitch angle solely from the vertical vanishing point $VP_v = (v_x, v_y)$ of vertical wall lines.
  2. The rectification matrix is parameterized as:
     $$H_{\text{raw}} = T_{\text{inv}} \cdot H_{\text{pitch}} \cdot T$$
     where $T$ centers on $v_x$, and $H_{\text{pitch}}$ scales the $y$-axis proportionally to distance from $v_y$:
     $$H_{\text{pitch}} = \begin{bmatrix} s & 0 & 0 \\ 0 & s & 0 \\ 0 & -1/v_y & 1 \end{bmatrix}, \quad s = 1.0 - \frac{c_y}{v_y}$$
  3. This eliminates radial wall lean while preserving orthogonal ground diagonals ($0.0^\circ$ horizontal shear).

### 2. Preventing Room Overlaps & Gaps
* **Past Failure:** In Iteration 2, fitting independent `cv2.minAreaRect` bounding boxes caused adjacent rooms to overlap by 50–100 px or leave wide gaps.
* **Iteration 5 Solution:**
  1. **Strictly Ban Independent Bounding Boxes.**
  2. Walls are represented as 1D segments in a single continuous planar graph.
  3. `shapely.ops.unary_union` intersects all segments into a topologically consistent 1D network.
  4. `shapely.ops.polygonize` generates the dual 2D polygonal faces.
  5. Adjacent rooms share exact 1D boundary edges, mathematically enforcing **$0.0000\text{ px}^2$ mutual overlap** and **$0\text{ px}$ gaps**.

---

## 6. Stage-Gated Execution Plan ("Tune Until It Works")

Development proceeds through five strict stage gates. No phase begins until the preceding phase passes its quantitative QA threshold across all three test assets:

```
[Phase 1: Rectification] ──▶ Gate 1: Vertical Deviation ≤ 1.0°, Floor Shear ≤ 1.0°
          │
          ▼
[Phase 2: Topology]      ──▶ Gate 2: 100% Closed Room Cycles, Zero Door Gaps
          │
          ▼
[Phase 3: Geometry]      ──▶ Gate 3: 0.0000 px² Overlap, 0 px Gaps, Orthogonality Dev ≤ 0.5°
          │
          ▼
[Phase 4: Scale]         ──▶ Gate 4: Total Area Error ≤ 3.0%, Fixture Sanity Check ± 10%
          │
          ▼
[Phase 5: Blueprint CAD] ──▶ Gate 5: Valid GeoJSON, Scaled SVG Blueprint, Publication PNG
```

### Phase 1: Input Normalization & Ground-Plane Rectification (`rectification.py`)
* **Objective:** Produce an orthographic bird's-eye view of both the render and the semantic guide.
* **QA Gating Metric (Gate 1):**
  * Vertical line deviation from $90.0^\circ$: $\le 1.0^\circ$ mean across all vertical structural lines.
  * Synchronous transformation: Render and semantic guide remain registered within $< 1.0\text{ px}$.
  * Pass criterion: Verified across `heritage towers`, `highlandlux`, and `limestone ranch`.

### Phase 2: Topological Extraction & Axial Bridging (`topology.py`)
* **Objective:** Ingest the guide (central line, no stroke, or stroke), bridge doorways along wall axes, and extract closed room cycles.
* **QA Gating Metric (Gate 2):**
  * Closed room cycle recovery: 100% of functional rooms (Living, Bedroom, Bath, Balcony, Closets) isolated.
  * Zero doorway leakage: Wall lines cleanly bridge door openings without bleeding into interior floors.

### Phase 3: Manhattan Planar Graph Polygonization (`geometry.py`)
* **Objective:** Snap wall lines to Manhattan axes, compute planar arrangement, and output solid room polygons.
* **QA Gating Metric (Gate 3):**
  * Mutual overlap between any two room polygons: **$0.0000\text{ px}^2$**.
  * Unclaimed gaps inside the unit footprint: **$0.0000\text{ px}^2$**.
  * Segment orthogonality deviation: $\le 0.5^\circ$ from $0^\circ / 90^\circ$.

### Phase 4: Conformal Metric Scale & Dimensional Engine (`scale.py`)
* **Objective:** Compute metric conversion factor from filename square footage and compute architectural room spans.
* **QA Gating Metric (Gate 4):**
  * Total unit area relative error: $\le 3.0\%$ compared to filename target square footage.
  * Room dimensions formatted as standard architectural feet/inches (e.g. `14'8" x 12'4"`).
  * Bathtub fixture dimension: $150\text{ cm} \pm 10\%$.

### Phase 5: Architectural CAD Blueprint Synthesis & Vector Export (`visualizer.py` / `export.py`)
* **Objective:** Render dual-line hatched wall blueprints and export structured GeoJSON/SVG assets.
* **QA Gating Metric (Gate 5):**
  * Export GeoJSON strictly compliant with RFC 7946 (valid polygons, right-hand rule).
  * 2D SVG vector blueprint rendered with clean wall hatching and centered room labels.
  * Publication-ready high-resolution PNG blueprints generated in `data/output/`.

---

## 7. Implementation Status

| Phase | Module | Status | QA Gate Metric | Gate Result |
| :---: | :--- | :---: | :--- | :---: |
| **1** | `rectification.py` | **Done** | Vertical axis skew $\le 1.0^\circ$, 1:1 synchronous warp | **PASS (0.13° - 0.33°)** |
| **2** | `topology.py` | **Done** | 100% closed room cycles, zero door leakage | **PASS (0 px overlap, 100% closed)** |
| **3** | `geometry.py` | Next | Overlap $= 0.0000\text{ px}^2$, Gaps $= 0.0000\text{ px}^2$, Ortho dev $\le 0.5^\circ$ | Pending |
| **4** | `scale.py` | Pending | Conformal area error $\le 3.0\%$, Fixture sanity check $\pm 10\%$ | Pending |
| **5** | `visualizer.py` / `export.py` | Pending | RFC 7946 GeoJSON, SVG blueprint, publication PNG | Pending |

---

## 8. Execution Report

### Phase 1: Ground-Plane Rectification & Input Normalization

#### Deliverables
* `src/v5_semi_automated/rectification.py`: Implements `SemiAutomatedRectifier` to discover and load semantic guides, estimate vertical vanishing points ($VP_v$), compute metric 1D pitch homographies ($H$), and synchronously warp renders and guides.
* `src/v5_semi_automated/__init__.py`: Module entry point exposing public interfaces and container data structures (`RectificationResult`, `SemanticGuide`, `SemiAutomatedRectifier`).
* `configs/config.yaml`: Added `v5_rectification` configuration section (`min_line_length_px: 25.0`, `vert_angle_tolerance_deg: 20.0`, `max_orthographic_vy: 50000.0`, `border_value: [255, 255, 255]`, `guide_dir: "human_semantic_guide"`).
* `docker-compose.yml`: Mounted `./human_semantic_guide:/app/human_semantic_guide` volume to support containerized execution.
* `scripts/v5_test_rectification.py`: Verification harness running detection, vanishing-point calibration, synchronous warping, and quantitative Gate 1 assertion.

#### Mathematical Pipeline & Mechanics
1. **Multi-Format Guide Discovery:** Automatically inspects `human_semantic_guide/` and matches unit assets by normalized key (e.g., `heritage_towers`, `highlandlux_citadel`, `limestone_ranch`), loading up to three complementary guide formats:
   * **Centerline:** Red 1D wall centerline axes ($R > 150, R - G > 50$).
   * **No Stroke:** Solid wall raster footprint with interior room cutouts.
   * **Stroke:** Dual-perimeter bordered wall body.
2. **Structural Vertical Line Extraction:** Extracts vertical lines using OpenCV LSD (`LineSegmentDetector`). Prioritizes red centerline guides to capture ground-plane wall axes directly without perspective shadow interference.
3. **Vertical Vanishing Point Estimation ($VP_v$):**
   * Formulates line equations $a_i x + b_i y + c_i = 0$ weighted by segment length $L_i$.
   * Executes RANSAC hypothesis generation followed by weighted SVD refinement on inliers to find the projective intersection $VP_v = (v_x, v_y)$.
4. **1D Ground Pitch Homography ($Z = 0$ Anchor):**
   * If $|v_y| \ge 50,000$ or $v_y > 0$, the camera is orthographic and $H = I_3$.
   * Otherwise, constructs the 1D pitch homography:
     $$H_{\text{raw}} = T_{\text{inv}} \cdot H_{\text{pitch}} \cdot T$$
     where $T$ centers on horizontal coordinate $v_x$, and:
     $$H_{\text{pitch}} = \begin{bmatrix} s & 0 & 0 \\ 0 & s & 0 \\ 0 & -1/v_y & 1 \end{bmatrix}, \quad s = 1.0 - \frac{c_y}{v_y}$$
   * Translates the transformed bounding quad into a non-negative bounding canvas: $H = T_{\text{fit}} \cdot H_{\text{raw}}$.
5. **Synchronous Dual-Stream Warping:**
   * Warps the 3-channel BGR 3D render with high-order Lanczos interpolation (`INTER_LANCZOS4`) with white background padding.
   * Warps the 4-channel BGRA semantic guides with nearest-neighbor interpolation (`INTER_NEAREST`) with transparent background padding `(0, 0, 0, 0)` to preserve pristine discrete label boundaries.

#### Quantitative QA Gate 1 Verification Results

Executed via Docker (`docker compose run --rm cv-floorplan python scripts/v5_test_rectification.py`):

| Unit Asset | Input Shape | Rectified Shape | Skew Before | Skew After | Skew Reduction | QA Gate 1 ($\le 1.0^\circ$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`heritage towers_a1_1 Bed 1 Bath 593 Sq. Ft.`** | $1140 \times 855$ | $1313 \times 756$ | $3.02^\circ$ | **$0.31^\circ$** | **89.7%** | **PASS** |
| **`highlandlux-citadel-1 Bed 1 Bath 623 Sq. Ft.`** | $659 \times 651$ | $659 \times 651$ | $0.13^\circ$ | **$0.13^\circ$** | *(Orthographic)* | **PASS** |
| **`limestone ranch_santa fe_625sq`** | $1140 \times 855$ | $1287 \times 768$ | $3.04^\circ$ | **$0.33^\circ$** | **89.1%** | **PASS** |

#### Visual Debug Artifacts Generated (`data/output/debug_v5/<unit>/`)
* `01_detected_vertical_lines.png`: Input render with detected converging vertical lines.
* `02_rectified_render.png`: Orthographic bird's-eye rectified floor plan.
* `03_rectified_center_line.png`: Rectified red wall centerline guide.
* `03_rectified_no_stroke.png`: Rectified solid wall mask guide.
* `03_rectified_stroke.png`: Rectified stroked wall boundary guide.
* `04_rectified_composite_overlay.png`: Alpha-blended composite verifying synchronous registration.
* `05_rectified_plumb_lines.png`: Rectified render overlaid with transformed vertical lines, confirming walls are plumb ($\le 0.33^\circ$).

#### Key Engineering Insights
1. **Vanishing Point Accuracy via Red Centerlines:** Deriving $VP_v$ from the red centerline guide rather than raw image textures eliminated false-positive perspective lines caused by diagonal bed headboards, drapery folds, and radial shadows.
2. **Preservation of Floor Plane Area:** Because homography is parameterized as a 1D vertical pitch rather than a full 8-DOF planar homography, horizontal room widths remain strictly invariant while vertical foreshortening is uniformly unrolled.
3. **Zero Sub-Pixel Guide Drift:** Using the identical transformation matrix $H$ across all render and guide layers ensured zero pixel shift between the floor image and the structural guides.

---

### Phase 2: Topological Extraction & Axial Door Bridging

#### Deliverables
* `src/v5_semi_automated/topology.py`: Implements `SemiAutomatedTopology` with multi-format guide ingestion, solid wall extraction, constrained axial door bridging, building envelope extraction, and closed room cycle isolation.
* `src/v5_semi_automated/__init__.py`: Exported Phase 2 data structures (`TopologicalRoomNode`, `TopologyResult`, `SemiAutomatedTopology`).
* `configs/config.yaml`: Added `v5_topology` configuration parameters (`min_room_area_px: 800.0`, `door_bridge_gap_px: 95`, `wall_alpha_thresh: 50`, `red_diff_thresh: 50`, `red_min_thresh: 150`).
* `scripts/v5_test_topology.py`: Visual and quantitative verification script testing all units against QA Gate 2 criteria.

#### Topological Mechanics & Formulations
1. **Multi-Format Guide Ingestion:**
   * Natively ingests any available guide format: `center_line` (red axes), `no_stroke` (solid wall body), or `stroke` (perimeter-bordered wall).
   * When both centerline and solid wall body are available, executes a cooperative fusion where the solid wall envelope acts as a topological constraint on the centerline graph.
2. **Constrained Axial Doorway Bridging:**
   * Architectural 3D renders frequently feature open door swings (e.g., $45^\circ$ or $90^\circ$ open interior doors), leaving gaps along partition wall centerlines (e.g., an 85 px gap in Limestone Ranch).
   * Unconstrained morphological dilation bridges these door openings but violently bleeds into narrow closets (erasing 23 px storage spaces) or carves artificial bridges across living areas.
   * **The Wall-Constrained Formulation:**
     $$\mathcal{B} = \mathcal{L}_{\text{center}} \cup \left( \left[ \mathcal{C}_{H}(\mathcal{L}_{\text{center}}, K) \cup \mathcal{C}_{V}(\mathcal{L}_{\text{center}}, K) \right] \cap \mathcal{M}_{\text{wall}} \right)$$
     where $K = 95\text{ px}$ 1D directional kernels bridge doorway openings along wall axes, but the intersection with $\mathcal{M}_{\text{wall}}$ mathematically guarantees that no bridged stroke can cross a room interior.
3. **Exterior Footprint Envelope Isolation:**
   * Extracts the exterior contour of $\mathcal{M}_{\text{wall}}$ and fills its interior to construct $\mathcal{M}_{\text{envelope}}$, defining the valid interior boundary of the apartment unit.
4. **Closed Room Cycle Decomposition:**
   * Evaluates interior walkable space:
     $$\mathcal{M}_{\text{interior}} = \mathcal{M}_{\text{envelope}} \setminus \mathcal{M}_{\text{wall}}$$
   * Computes disjoint connected components $\mathcal{C}_i$ on $\mathcal{M}_{\text{interior}}$ with $\text{Area}(\mathcal{C}_i) \ge \text{min\_room\_area\_px}$.
   * Extracts clean perimeter contours, centroids $(c_x, c_y)$, bounding boxes, and architectural room category priors (Living, Primary Bedroom, Secondary Bedroom, Bathroom, Balcony, Closet).

#### Quantitative QA Gate 2 Verification Results

Executed via Docker (`docker compose run --rm cv-floorplan python scripts/v5_test_topology.py`):

| Unit Asset | Closed Rooms Extracted | Expected Min Rooms | Inter-Room Overlap | QA Gate 2 |
| :--- | :---: | :---: | :---: | :---: |
| **`heritage towers_a1_1 Bed 1 Bath 593 Sq. Ft.`** | **6** | 6 | **0 px** | **PASS** |
| **`highlandlux-citadel-1 Bed 1 Bath 623 Sq. Ft.`** | **9** | 9 | **0 px** | **PASS** |
| **`limestone ranch_santa fe_625sq`** | **9** | 9 | **0 px** | **PASS** |

#### Extracted Room Inventory Breakdown

1. **`heritage towers` (6 Closed Rooms):**
   * Room 1: `LIVING / DINING / KITCHEN` (165,295 px, centroid (511.9, 361.8))
   * Room 2: `PRIMARY BEDROOM` (64,815 px, centroid (825.5, 300.0))
   * Room 3: `BATHROOM` (30,236 px, centroid (905.0, 534.2))
   * Room 4: `WALK-IN CLOSET` (25,031 px, centroid (762.1, 554.0))
   * Room 5: `FOYER / COAT CLOSET` (5,452 px, centroid (715.5, 461.4))
   * Room 6: `BALCONY` (5,123 px, centroid (385.2, 425.0))

2. **`highlandlux-citadel` (9 Closed Rooms):**
   * Room 1: `LIVING / DINING / KITCHEN` (117,648 px, centroid (224.0, 303.1))
   * Room 2: `PRIMARY BEDROOM` (66,098 px, centroid (500.3, 332.5))
   * Room 3: `SECONDARY BEDROOM / DEN` (28,922 px, centroid (422.8, 118.0))
   * Room 4: `BATHROOM` (21,487 px, centroid (572.3, 115.6))
   * Room 5: `BALCONY` (8,977 px, centroid (58.4, 117.6))
   * Room 6: `PATIO / TERRACE` (8,716 px, centroid (598.4, 556.4))
   * Room 7: `WALK-IN CLOSET` (4,503 px, centroid (293.9, 47.9))
   * Room 8: `PANTRY / UTILITY` (1,718 px, centroid (144.4, 225.3))
   * Room 9: `LINEN / ENTRY CLOSET` (1,223 px, centroid (335.5, 106.9))

3. **`limestone ranch` (9 Closed Rooms):**
   * Room 1: `LIVING / DINING / KITCHEN` (141,291 px, centroid (521.3, 412.1))
   * Room 2: `PRIMARY BEDROOM` (83,737 px, centroid (817.1, 320.2))
   * Room 3: `COVERED PATIO / BALCONY` (28,085 px, centroid (833.9, 619.3))
   * Room 4: `BATHROOM` (18,847 px, centroid (882.6, 494.0))
   * Room 5: `WALK-IN CLOSET` (9,533 px, centroid (352.1, 609.5))
   * Room 6: `PANTRY / UTILITY` (5,749 px, centroid (673.0, 587.9))
   * Room 7: `BALCONY STORAGE` (5,511 px, centroid (424.4, 111.5))
   * Room 8: `ENTRY CLOSET` (2,938 px, centroid (352.3, 516.5))
   * Room 9: `LINEN CLOSET` (1,503 px, centroid (688.8, 497.9))

#### Visual Debug Artifacts Generated (`data/output/debug_v5/<unit>/`)
* `06_topological_wall_mask.png`: Binary mask of structural walls with bridged door openings.
* `07_bridged_centerline_mask.png`: 1D wall centerline network rendered in electric cyan.
* `08_topological_rooms_overlay.png`: Publication-grade composite with distinct pastel room fills, solid dark borders, and text badges showing `#ID Label (Area px)`.
* `09_room_seeds_centroids.png`: Point seeds and centroids plotted directly onto the rectified render.

#### Key Engineering Insights
1. **Zero Overlap by Construction:** Because rooms are derived as the connected components of the complement of the closed wall network ($\mathcal{M}_{\text{interior}}$), mutual overlap between any two room masks is mathematically guaranteed to be **$0\text{ px}$**.
2. **Door Bridging Without Narrow-Room Corruption:** Constraining the axial closing within the physical wall envelope solved the persistent problem from Iteration 2 and 4 where door-closing kernels erased small 20–30 px closets or bridged open-plan living rooms.
3. **Pristine Seed Generation for Geometry:** Every room now has a stable, well-defined centroid and solid footprint mask, providing inputs for Phase 3 Manhattan polygonization.


