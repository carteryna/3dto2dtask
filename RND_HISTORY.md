# Comprehensive R&D Retrospective & Failure Analysis: Iterations 1 to 4

## Executive Summary: The Structural Pivot

Over four intensive engineering iterations, we investigated and implemented fully automated Computer Vision (CV) and Machine Learning (ML) pipelines to convert 3D perspective architectural apartment floor-plan cutaways (`.webp`) into clean, 2D planar CAD layouts with metric dimensions.

The fundamental conclusion of this R&D cycle is:
> **Fully automated 2D architectural reconstruction from stylized 3D perspective renders is inherently ill-posed and brittle.**

Artistic 3D marketing renders are optimized for visual appeal, not photometric or geometric consistency. They introduce severe non-linear visual artifacts:
1. **3D Wall Extrusions ($Z = H$ vs. $Z = 0$):** High walls occlude floor boundaries, cast perspective-dependent shadows, and lean radially away from the virtual camera's principal point.
2. **Interior Furniture Occlusion:** Sectional sofas, king beds, dining tables, and kitchen islands cover 30% to 50% of the true walkable floor plane.
3. **Photometric Camouflage & Texture Noise:** High-gloss white bathroom tiles share identical LAB color space profiles ($L \ge 250, \text{chroma} \le 8$) with painted wall tops; wood plank textures, tile grouting, and carpet patterns generate hundreds of false-positive gradient edges.
4. **Open-Plan Topological Ambiguity:** Real-world apartments frequently merge Living, Dining, Hallway, and Kitchen zones without physical partition walls. Purely mathematical line arrangements cannot intuit human functional boundaries across continuous flooring.

While manual preprocessing in tools like Photoshop can isolate structural wall masks in under 60 seconds with 100% human semantic fidelity, automated algorithmic pipelines require fragile, cascading heuristics that collapse when exposed to new rendering styles.

We are therefore fundamentally pivoting to a **semi-automated architecture (Iteration 5)**:
- **Human in the Loop:** Rapid manual prep/cleanup handles the visual chaos (removing furniture, isolating clear wall perimeters).
- **Automation Engine:** High-precision computational geometry handles metric scale calibration, Manhattan vectorization, CAD snapping, topological verification, and structured GeoJSON/CAD blueprint export.

Below is the definitive post-mortem of Iterations 1 through 4 to preserve our mathematical models, experimental findings, and failure modes.

---

## Iteration 1: Perspective Semantic Segmentation (`src/v1_baseline/`)

### 1. Architectural Approach
The baseline pipeline attempted direct, bottom-up image segmentation and polygonization on the raw 3D perspective render:
- **Segmentation:** FastSAM point-prompting with a fallback to marker-based OpenCV watershed seeded on high-contrast floor regions.
- **Boundary Regularization:** Contour extraction via OpenCV `findContours`, simplified using Shapely's Douglas-Peucker algorithm (`approxPolyDP`).
- **Scale Estimation:** EasyOCR character height detection under the hypothesis that font point sizes in render disclaimers scaled proportionally with unit dimensions.

### 2. Root Causes of Failure
- **Perspective Camera Skew Retained:** The render's oblique camera angle ($\sim 60^\circ - 70^\circ$ elevation) was unrectified. Rectangular rooms appeared as trapezoids and rhombuses; parallel opposite walls converged towards vanishing points.
- **Extruded Wall-Top Anchoring:** North-facing walls exposed the wall-floor base seam, while south-facing walls occluded the base seam, revealing only the flat wall top at height $Z = H$. Contours snapped haphazardly between floor bases and wall tops, warping real-world room depth by 20% to 40%.
- **Furniture Contour Hugging:** FastSAM segment perimeters followed the exact curved silhouettes of sofas, bed pillows, and coffee tables. Room polygons became irregular, jagged blobs with 40+ vertices rather than 4-sided Manhattan polygons.
- **Font-to-Scale Hallucination:** OCR text size varied arbitrarily according to graphic designer choices across different marketing agencies, bearing zero mathematical correlation to physical room dimensions.

---

## Iteration 2: Homography & Independent Bounding Boxes (`src/v2_reconstruction/`)

### 1. Architectural Approach
Iteration 2 recognized the need to correct perspective distortion and enforce rectangular geometry:
- **Perspective Correction:** 4-point planar homography computed from manually identified or detected exterior wall-top corners.
- **Floor Seam Extraction:** Directional Sobel filters along Manhattan axes combined with morphological closing to bridge doorway breaks.
- **Room Fitting:** Extracted connected floor components and fit independent minimum-area bounding boxes (`cv2.minAreaRect`) to each detected room mask.
- **Physical Scale Heuristics:** Detected architectural fixtures (standard 80 cm doorway gaps, 60 cm kitchen countertops, 150 cm bathtubs) using heuristic slit filters and HSV blue-water detection.

### 2. Root Causes of Failure
- **The Planar Homography Shearing Paradox:**
  A 2D homography $H \in \mathbb{R}^{3 \times 3}$ is valid only for a single planar surface. By computing homography on the exterior wall tops at height $Z = H$, the floor plane at height $Z = 0$ was violently sheared. Ground-level objects suffered severe perspective parallax, turning rectangular rooms into skewed parallelograms.
- **Independent Bounding Box Overlap Catastrophe:**
  In architecture, adjacent rooms share a single dividing wall. Fitting independent `cv2.minAreaRect` boxes to segmented floor regions ignored global topology:
  - Neighboring room boxes overlapped by 50 to 100 pixels, creating invalid physical intersections where one room occupied the space of another.
  - In corridors and foyers, boxes failed to meet, leaving wide dead-space gaps where walls did not exist.
- **Heuristic Slit & Fixture Hallucination:**
  The doorway gap detector searched for narrow interruptions in wall lines ($16\text{ px} \le \text{gap} \le 80\text{ px}$). It continually hallucinated doorways across curtain folds, between sectional sofa cushions, across dark area rugs, and through kitchen cabinet gaps. Countertop depth detection failed due to variable cabinet styling and perspective foreshortening.

---

## Iteration 3: Topological Graph & Shared Edges (`src/v3_topology/`)

### 1. Architectural Approach
Iteration 3 completely eliminated homography warping to preserve raw edge sharpness and introduced topological constraints:
- **Walkable Footprint Partitioning:** Processed the original render directly. Filtered FastSAM everything-mode proposals in LAB color space to eliminate thin wall-top ribbons ($L \ge 250, \text{chroma} \le 8$, morphological opening 31 px).
- **Marker-Based Watershed Cell Complex:** Seeded interior rooms with proposal centroids and executed watershed constrained within the building footprint. This mathematically guaranteed a disjoint cell complex: **0 overlap and 0 gaps** across all cells.
- **Manhattan Line Graph Extraction:** Extracted contours of each cell, snapped edges to the dominant axes ($0^\circ$ and $90^\circ$), clustered parallel walls within $12\text{ px}$ to a shared coordinate, and subtracted inner furniture blocks as structural holes.
- **Filename Area Scaling:** Conformal scale $S = \sqrt{\frac{\sum A_{px}}{A_{sqft}}}$ extracted directly from marketing metadata in filenames (e.g., `593 Sq. Ft.`, `625sq`).

### 2. Root Causes of Failure
- **Perspective Radial Wall Lean ("Double-Wall Chasms"):**
  Because homography was abandoned, the 3D perspective distortion remained uncorrected. Vertical partition walls leaned radially outward from the center of the image. For example, in a central hallway, the left partition wall leaned left ($+5^\circ$) while the right partition wall leaned right ($-5^\circ$). At the top of the corridor, the walls were 25 px further apart than at the bottom. A rigid Manhattan coordinate snap ($12\text{ px}$ tolerance) could not bridge this non-parallel divergence, leaving massive, unmerged "double-wall chasms."
- **Furniture Subtraction ("Rooms with Holes"):**
  Large opaque furniture items (such as the king bed in Limestone Ranch or sectional sofas) survived the wall-ribbon filter. When subtracted as holes from the parent room polygons, they produced hollow, doughnut-shaped rooms and U-shaped living rooms with structural voids.
- **Photometric Tile Camouflage:**
  In Highland Lux, the white bathroom floor tiles shared identical color properties with painted wall tops. Morphological ribbon subtraction completely erased the bathroom floor, leaving an unseeded dead zone that was absorbed into neighboring closets or corridors during watershed flooding.
- **Topological Graph Over-Segmentation:**
  Fine textural changes in open-plan areas (e.g., transitions from kitchen wood laminate to living room carpet) triggered multiple disconnected watershed basins that could not be reliably merged without human knowledge of architectural intent.

---

## Iteration 4: Metric Calibration, Planar Arrangement & Semantic CAD (`src/v4_cad_arrangement/`)

### 1. Architectural Approach
Iteration 4 was our most rigorous mathematical and algorithmic attempt:
- **Phase 1: Manhattan Camera Calibration & Ground Rectification (`calibration.py`):**
  Extracted structural line segments via OpenCV's `LineSegmentDetector` (LSD). Estimated orthogonal vertical ($VP_v$) and horizontal ($VP_h$) vanishing points via RANSAC line intersection with SVD least-squares refinement. Constructed a ground rectification homography $H_{\text{floor}}$ centered at $(v_x, c_y)$ with scale normalization $s = 1.0 - c_y / v_y$, eliminating perspective vertical wall lean ($>92\%$ to $100\%$ skew elimination).
- **Phase 2: Semantic Object Decoupling (`objects.py`):**
  Trained and filtered FastSAM proposals into decoupled CAD layers (`bed`, `sofa`, `dining_table`, `bathtub`, `toilet`). Enforced the **Floor Integrity Rule**: furniture masks were explicitly added back to the floor plane rather than carved out as holes. Rescued camouflaged bathrooms via a plumbing co-occurrence rule (pairing enamel bathtubs with toilets).
- **Phase 3: Planar Wall-Graph Line Arrangement (`arrangement.py`):**
  Extracted exterior Manhattan envelope polygons. Detected structural partition centerlines on the rectified image excluding furniture. Snapped parallel lines within $15\text{ px}$, merged collinear intervals across doorways, extended dangling endpoints via ray-shooting ($75\text{ px}$ tolerance), and executed Shapely planar graph polygonization (`unary_union` + `polygonize`) to guarantee a zero-overlap, zero-gap dual cell complex.
- **Phase 4: Spatial Containment Join & Metric Classification (`classification.py`):**
  Performed spatial intersection between planar room faces and decoupled furniture/fixtures to assign architectural labels (`BEDROOM`, `BATHROOM`, `LIVING / DINING`, `BALCONY`, `CLOSET`). Merged adjacent same-category sub-cells and formatted room spans into architectural feet and inches (e.g., `15'5" x 13'6"`).

### 2. Root Causes of Failure
While Iteration 4 solved the perspective wall-lean and hole-punching failures of Iterations 1–3, it encountered the insurmountable boundary limits of fully automated vision:
- **Visual Noise vs. Structural Boundary Conflation:**
  In rectified images, cast shadows from 3D furniture, baseboard highlights, door swings, and kitchen cabinetry grout lines produced high-gradient edges that were indistinguishable from real partition walls. Even with furniture exclusion masks, cabinet edges and shadow boundaries triggered false-positive wall centerlines.
- **Ray-Shooting Brittleness (Undershoot vs. Overshoot):**
  The planar arrangement algorithm relies on extending line segments to close polygonal faces. A fixed extension tolerance ($75\text{ px}$) caused an insoluble dilemma:
  - If tolerance was too low, legitimate partition walls terminated 5 px short of perpendicular walls, failing to close faces and collapsing multiple rooms into an enormous unpartitioned space.
  - If tolerance was too high, wall centerlines overshot across open-plan living/dining areas, carving artificial walls where none existed.
- **Semantic Ambiguity of Open Floor Plans:**
  In modern apartments, the kitchen, dining, foyer, and living room occupy one seamless continuous floor. An automated line detector has no physical cue to separate them. Minor texture changes or bar stools caused the arrangement engine to fracture the living area into 4 to 6 artificial micro-polygons, while spatial joins arbitrarily assigned conflicting labels across merged sub-cells.
- **Edge Hallucinations from Non-Standard Geometry:**
  Balconies with angled railings, diagonal kitchen counters, and multi-wing apartment geometries violated pure Manhattan orthogonal assumptions, causing the line graph to generate sliver triangles and degenerate faces.

---

## Synthesis of R&D Insights

| Dimension | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Geometry Representation** | Jagged 2D contours | Rotated bounding boxes | Watershed cell complex | Dual planar graph arrangement |
| **Perspective Handling** | None (raw 3D skew) | 4-point wall homography | None (unwarped render) | RANSAC vanishing point ground homography |
| **Room Overlap** | Uncontrolled | Massive (50–100 px) | 0.0 px (disjoint cells) | **0.0000 px² (shared 1D edges)** |
| **Furniture Handling** | Hugged as perimeters | Ignored | Subtracted as structural holes | Decoupled CAD layer (solid floors) |
| **Scale Method** | EasyOCR text size | Fixture heuristics | Filename unit area ($S = \sqrt{A/A_{sqft}}$) | Conformal filename scaling |
| **Fatal Failure Mode** | 3D skew & contour noise | Severe floor plane shear & box overlap | Perspective radial lean & double-wall chasms | Open-plan over-segmentation & shadow edges |

---

## Architectural Pivot: Iteration 5 (Semi-Automated Workflow)

### The Core Thesis
1. **Automating visual perception on 3D marketing renders has diminishing returns:**
   Every heuristic designed to close a door gap opens a false door on a sofa; every filter designed to remove furniture removes part of a small closet or bathroom.
2. **Human input is vastly superior for topological segmentation:**
   A human designer in Photoshop or a lightweight markup interface can draw or trace wall centerlines or clean floor masks in under 60 seconds with 100% semantic accuracy.
3. **Software is vastly superior for mathematical precision:**
   Once clean raster masks or vector guides are provided, automation excels at metric scaling, Manhattan edge orthogonalization, CAD wall hatching, room dimension formatting, furniture block insertion, and export to DXF / SVG / GeoJSON.

### Next Steps: Clean Slate Reset
1. All legacy automated experimental code from Iterations 1 through 4 is archived in `src/archive_v1_to_v4/`.
2. Legacy automated test scripts are moved to `scripts/archive_v1_to_v4/`.
3. `src/v5_semi_automated/` is created as the clean placeholder for the new semi-automated CAD vectorization architecture.
