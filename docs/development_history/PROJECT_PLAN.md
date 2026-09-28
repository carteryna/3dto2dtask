# 3D Floor Plan to 2D Room Layout Pipeline — Project Blueprint

## 1. Position Alignment & Senior R&D Context

- **Role:** Middle/Senior Computer Vision & Machine Learning Engineer
- **Domain:** Architectural drawings, real estate tech, and 3D floor plan layout extraction for downstream flooring takeoff workflows.
- **Key Capabilities Demonstrated:**
  - Hybrid pipeline design (combining classical geometry with modern vision-foundation models).
  - Vector geometry processing & regularization (`Shapely`, contour simplification, Manhattan snapping).
  - Design of robust fallback mechanics for imperfect, noisy visual inputs.
  - Quantitative evaluation, error analysis, and production-ready containerization.

---

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

---

## 2. Test Task Objective & Requirements

- **Goal:** Build a robust exploratory prototype that converts a 3D apartment floor plan rendering into a simplified 2D room layout with regularized boundaries and relative pixel-based areas.
- **Primary Deliverables:**
  1. **Annotated Image:** Visual overlay displaying detected room boundaries, IDs, centroids, and area labels.
  2. **JSON Payload (`data/output/result.json`):** Regularized polygon vertices, relative area %, confidence scores, and bounding boxes.
  3. **Containerized Execution:** `Dockerfile` and `docker-compose.yml` for easy setup and local directory mounting.
  4. **Documentation (`README.md`):** Methodology description, assumptions, limitations, metric evaluation, and future R&D roadmap.

---

## 3. Input Data Insights & Strategy Adjustments

- **Orthographic / Extruded Perspective:** Renders feature top-down views with extruded wall heights and depth shadows along base seams.
- **High Texture Variance:** Distinct room materials (hardwood, carpet, granite tiles, balcony decking) cause simple color thresholding to over-segment.
- **Furniture Occlusions & Open Doors:** Sofas, beds, and 45° open doors interrupt continuous floor regions and break room enclosures.
- **Strategy Pivot:** Implement a **Wall-First Structural Engine** to isolate light-gray/white wall tops and close door gaps via morphological dilation before seeding room segmentation.

### 3.1 Exploratory Data Analysis (EDA) Breakdown

Source examples in `inout_data`:

- `heritage towers_a1_1 Bed 1 Bath 593 Sq. Ft..webp`
- `highlandlux-citadel-1 Bed 1 Bath 623 Sq. Ft..webp`
- `limestone ranch_santa fe_625sq.webp`

#### 📸 Image 1: `heritage towers_a1` Analysis

- **Visual Characteristics:**
  - Orthographic / near-top-down 3D render with inner wall height extrusion and shadow shear[cite: 1].
  - High floor texture variance: dark wood planks in living/kitchen, neutral carpet in bedroom, dark granite/tile in bathroom, light decking on balcony[cite: 1].
  - Heavy occlusion from furniture (sofas, beds, tables) and $45^\circ$ open doors cutting wall perimeters[cite: 1].
  - Printed OCR dimensions (`13'3" x 13'3"`, `11'9" x 10'9"`) and unit square footage in metadata (`593 Sq. Ft.`)[cite: 1].
- **Pipeline Impact:**
  - Color thresholding fails due to wood grain patterns and area rugs[cite: 1].
  - Primary reliance on high-contrast white wall tops and OCR scale anchoring[cite: 1].

#### 📸 Image 2: `highlandlux-citadel` Analysis

- **Visual Characteristics:**
  - Clean overhead top-down view with minimal wall extrusion shear[cite: 2].
  - Continuous beige carpet across living, dining, entry, and bedroom corridors[cite: 2].
  - No printed dimension text or scale annotations[cite: 2].
  - Dense internal noise in small areas (clothes racks in walk-in closet, kitchen cabinetry)[cite: 2].
- **Pipeline Impact:**
  - Pure color/texture segmentation merges living, hallway, and bedroom into a single mask[cite: 2].
  - Closing topological door gaps is critical to prevent mask bleeding across open entryways[cite: 2].
  - Scale estimation must rely on heuristic pixel metrics (e.g., standard doorway width or counter depth)[cite: 2].

#### 📸 Image 3: `limestone ranch_santa fe` Analysis

- **Visual Characteristics:**
  - Moderate 3D wall extrusion showing inner vertical faces and soft ambient occlusion shadows along lower wall seams[cite: 3].
  - Crisp material transition lines: dark wood floor (kitchen/bath) vs. tan carpet (living/bedroom)[cite: 3].
  - Directional sunlight casting rectangular shadows across bedroom carpet[cite: 3].
  - High frequency edge noise in closets (exposed wire racks)[cite: 3].
- **Pipeline Impact:**
  - Soft window light shadows trigger false edges if using simple Canny/Sobel filters[cite: 3].
  - Material transition lines serve as reliable implicit boundaries for open-concept room divisions[cite: 3].

---

#### ⚠️ Identified Failure Modes & Mitigations Taxonomy

| Challenge in Input | Potential Failure Mode | Pipeline Mitigation Strategy |
| :--- | :--- | :--- |
| **$45^\circ$ Open Doors**[cite: 1, 2] | Merges adjacent rooms into a single mask (under-segmentation)[cite: 2]. | Morphological line dilation on white wall tops to bridge door gaps[cite: 1, 2]. |
| **Area Rugs & Furniture**[cite: 1] | Splits a single room into fragmented sub-polygons (over-segmentation)[cite: 1]. | Segment structural walls first to define bounding cells rather than floor colors[cite: 1]. |
| **Uniform Carpet Across Rooms**[cite: 2] | Prevents color-based room boundary detection[cite: 2]. | Rely on door-gap closed wall masks and distance-transform seeds[cite: 1, 2]. |
| **Window Light Shadows**[cite: 3] | Creates false internal edge boundaries[cite: 3]. | Apply Bilateral / Median filtering to smooth lighting gradients while preserving wall edges[cite: 1, 3]. |
| **Missing OCR Dimensions**[cite: 2, 3] | Inability to compute absolute square footage[cite: 2, 3]. | Fallback to relative pixel area percentages and standard architectural prior heuristics[cite: 2, 3]. |

---

## 4. High-Level Architecture & Technical Strategy

```text
                        ┌────────────────────────┐
                        │   Input 3D Render      │
                        └───────────┬────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────┐
│  Structural Wall &    │                       │  Flooring & Texture   │
│  Door-Gap Engine      │                       │  Separation Engine    │
└──────────┬────────────┘                       └──────────┬────────────┘
           │                                               │
           │  • HSV/LAB Wall Thresholding                  │  • Bilateral Denoising
           │  • Morphological Line Closing (45°/90°)       │  • Foundation Model (FastSAM/MobileSAM)
           │  • Connected Component Wall Mask              │  • Seeded Watershed Fallback
           │                                               │
           └───────────────────────┬───────────────────────┘
                                   │
                                   ▼
                        ┌────────────────────────┐
                        │   Topology Merger &    │
                        │   Room Cell Extraction │
                        └───────────┬────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │ Shapely Regularization │
                        │  (Orthogonal Snapping) │
                        └───────────┬────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │ OCR & Scale Parsing    │
                        │  (EasyOCR Integration) │
                        └───────────┬────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │ Output Generators      │
                        │ (JSON & Visual Overlay)│
                        └────────────────────────┘
```

The pipeline splits the input render into two complementary paths, then fuses them:

1. **Structural Wall & Door-Gap Engine** isolates light-gray and white wall tops, closes 45° and 90° door gaps with morphological line closing, and produces a connected-component wall mask that defines room enclosures.
2. **Flooring & Texture Separation Engine** denoises the render, proposes floor regions with a vision-foundation model (FastSAM or MobileSAM), and falls back to seeded watershed when texture variance or furniture occlusion breaks the primary proposal.
3. **Topology Merger & Room Cell Extraction** combines the wall mask with floor proposals into discrete room cells.
4. **Shapely Regularization** simplifies contours and snaps boundaries to an orthogonal (Manhattan) world.
5. **OCR & Scale Parsing** reads scale annotations with EasyOCR and attaches spatial context to the geometry.
6. **Output Generators** write the annotated overlay and the JSON payload.

---

## 5. Evaluation Metrics Taxonomy

- **Mean IoU & Boundary IoU:** Evaluates region mask precision and wall alignment accuracy within a 6-pixel perimeter band.
- **Relative Area Error (RAE %):** Measures percentage error in pixel-based room area calculations.
- **Orthogonality Deviation ($\Delta \theta_{\text{ortho}}$):** Mean absolute deviation of wall vector angles from $90^\circ$ (Manhattan world geometry adherence).

---

## 6. Sequential Implementation Phases

- **Phase 1:** Project Scaffolding & Environment Setup (`configs/`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`).
- **Phase 2:** Structural Wall Extraction & Topological Door-Gap Closing (`src/preprocessing.py`).
- **Phase 3:** Hybrid Room Segmentation & Fallback Engine (`src/segmentation.py`).
- **Phase 4:** Polygon Vector Regularization & Geometry Engine (`src/geometry.py`).
- **Phase 5:** OCR Scale Parsing & Spatial Context Fusion (`src/ocr.py`).
- **Phase 6:** Evaluation Metrics, Visualizer, CLI & Integration (`src/metrics.py`, `src/visualizer.py`, `main.py`).

Implementation starts at Phase 1 only after this blueprint is accepted. No application code is generated as part of this document.
