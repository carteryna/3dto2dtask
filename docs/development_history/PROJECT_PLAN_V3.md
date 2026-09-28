# Iteration 3 — Topological Graph and Shared-Edge Reconstruction

Iteration 2 (`src/v2_reconstruction/`) flattened the render from the wall tops, then fitted each room as its own rectangle. That sheared the floor plane, left massive overlaps between independent boxes, and treated curtains and furniture gaps as doors. Iteration 3 (`src/v3_topology/`) keeps the original image, partitions the floor into a shared-edge cell complex, orthogonalizes the wall lines rather than the boxes, and scales from the unit area in the filename.

Run with `python main.py --version v3` (the default). `--version v2` still runs Iteration 2. `--version v1` still runs the archived baseline.

## Phase 1: Deep floor segmentation in original space (`segmentation.py`)

**Objective:** Extract pure floor masks without distorting the original image.

- Drop the homography module. Process the original render.
- Use FastSAM to extract high-confidence floor regions, ignoring wall tops.
- Group the floor masks into a single, unified walkable-area footprint.

## Phase 2: Topological partitioning (`topology.py`)

**Objective:** Divide the walkable footprint into non-overlapping room cells that share exact boundaries.

- Extract room centroids (seeds) from the Phase 1 masks.
- Run a marker-based watershed on the unified walkable footprint.
- Every pixel belongs to exactly one room. Adjacent rooms share a single edge, with no overlaps and no gaps.

## Phase 3: Manhattan line graph (`geometry.py`)

**Objective:** Extract and orthogonalize the shared walls.

- Extract the boundaries between the watershed cells as a graph of line segments.
- Force every segment to strictly horizontal ($0^\circ$) or vertical ($90^\circ$).
- Re-intersect the orthogonal lines so room polygons meet at clean corners.

## Phase 4: Global area scale engine (`scale.py`)

**Objective:** Use the filename's square footage as the scale.

- Parse the filename for total square footage (for example `593 Sq. Ft.`).
- Measure the total pixel area of the combined room polygons.
- Compute the global scale $S = \sqrt{\frac{\text{Total Pixel Area}}{\text{Total Sq Ft}}}$.
- Fallback: if the filename has no size, use YOLOv8n to detect bathtubs and toilets. Do not scale from untextured gaps.

## Phase 5: Blueprint rendering (`visualizer.py` and `main.py`)

**Objective:** Draw the 2D layout.

- Render the orthogonal line graph on a blank white canvas.
- Draw the lines with a set stroke width so the walls have thickness.
- Place the room id, relative area, and square footage from the global scale at each polygon centroid.

## Status

| Phase | Module | Status |
| --- | --- | --- |
| 1 | `segmentation.py` | Done |
| 2 | `topology.py` | Done |
| 3 | `geometry.py` | Done |
| 4 | `scale.py` | Done |
| 5 | `visualizer.py` | Done |

All five phases are implemented. `python main.py --version v3` writes the blueprint.

## Execution Report

### Phase 1: Deep floor segmentation in original space

**Deliverables**

- `src/v3_topology/segmentation.py`: `FloorSegmenter.segment(image)` returns disjoint binary floor masks on the original render. `walkable_footprint(masks)` unions them and fills furniture holes that do not cross a wall top. `segmenter.debug` holds the building silhouette, the wall-top mask, the unclaimed interior, and the proposal counts.
- `configs/config.yaml`: new `v3_segmentation:` section. FastSAM weights and image size are reused from `segmentation:`.
- `scripts/v3_test_segmentation.py`: writes `01_wall_tops.png`, `02_floor_masks.png`, `03_walkable_footprint.png`, and `04_unclaimed.png` under `data/output/debug_v3/<stem>/`.

**Pipeline**

1. The image is not warped. The white page around the render is the only thing removed, so the apartment silhouette stays in the original pixel space.
2. FastSAM everything-mode runs at confidence 0.10. Floor planes score about 0.12–0.25. Furniture scores about 0.45 and up, so a high objectness cutoff would keep the sofa and drop the floor.
3. Wall tops are thin, flat ribbons in the colour of this render's exterior band. A 31 px opening deletes them and leaves wide floors, including tile and balcony decking. Those ribbons are subtracted from every proposal.
4. A proposal is a floor when it stays inside the silhouette, keeps a thick core, and is not high-objectness furniture. The largest copy wins when several proposals describe the same floor.
5. A proposal that swallowed a whole wing of walls is stripped back to the non-wall pixels, which recovers floors FastSAM bundled with the walls.
6. Whatever is still empty inside the silhouette, and is neither a wall top nor furniture, is recovered as floor. That is where the beige carpet comes from. A 15 px close joins notches cut by furniture, then the wall tops are stamped back so the close cannot bridge two rooms.
7. A solid blob whose outline sits on a recovered floor is furniture on that floor and is dropped. Masks are disjoint. Overlap on all three renders is 0 px, and no walkable pixel lies on a wall top.

**Results**

| Render | Proposals | Floors | Silhouette covered | Wall pixels in the footprint |
| --- | --- | --- | --- | --- |
| heritage towers | 300 | 10 | 0.522 | 0 |
| highlandlux citadel | 300 | 13 | 0.719 | 0 |
| limestone ranch | 300 | 11 | 0.638 | 0 |

Coverage is the walkable footprint divided by the apartment silhouette. The silhouette includes the extruded walls, so the uncovered share is mostly wall and furniture rather than missing rooms.

- Heritage keeps the open living, kitchen, and hall as one wood floor, and separates the bedroom, closet, bathroom, tub, and balcony. A few fragments remain: the kitchen back counter, a hallway sliver, and a spill past the lower-left wall.
- Highland keeps the kitchen and hall, the living and dining room, the bedroom, the balcony, and part of the closet. The white bathroom tile is missing. A few masks sit on the exterior wall face, outside the rooms.
- Limestone keeps the living-room carpet, the bedroom carpet, the kitchen wood, the walk-in closet, and the bathroom. The bed and the sofa are still their own masks. The balcony and the small storage closets are missing.

**Insights**

- **FastSAM does not propose floors at the confidence used for objects.** At 0.25 and above, the masks are sofas, beds, counters, and the white page. The actual floor planes show up only once the cutoff drops to 0.10, and even then their scores stay near 0.15. "High confidence" here means a mask that survives the floor tests, not a high objectness score.
- **Wall tops and floors cannot be split by colour.** The same flat shell colour covers wall tops, white tile, and some counters. Width is the split that holds: a ribbon that disappears under a 31 px opening is a wall top, and a wide region is eligible to be floor.
- **Beige carpet is invisible to FastSAM as an object.** Dark wood was proposed directly. The tan living-room floors were not. They are recovered as the interior left after wall tops and furniture are removed. That complement is doing the work the model does not.
- **The footprint is a cell complex with no overlaps.** Every kept pixel belongs to one mask, and subtracting wall tops leaves a shared boundary instead of two boxes fighting over the same floor. That is the input Phase 2's watershed needs.

**Known limitations**

- Open doorways stay open, so a closet that shares a wide opening with a hall can be one region or two depending on what FastSAM proposed. Splitting those is Phase 2.
- Highland's bathroom tile matches the wall colour closely enough that it never becomes a floor mask. The shower and toilet are not a substitute for that floor.
- Beds and sofas whose outline is not surrounded by a recovered floor stay as extra seeds. Limestone's bed and sofa are the clear cases.
- The silhouette follows the outer edge of the extruded walls, so a thick exterior face can still produce a mask. Highland has a few of these outside the rooms.
- Thresholds were set on these three renders. A wider set would move the 31 px wall-top width and the 0.45 objectness cutoff.

### Phase 2: Topological partitioning

**Deliverables**

- `src/v3_topology/topology.py`: `TopologyPartitioner.seeds(masks)` returns one centroid per Phase 1 floor mask. `partition(footprint, seeds, masks)` runs a marker watershed and returns one cell per seed. The cells are disjoint and their union is the walkable footprint.
- `configs/config.yaml`: new `v3_topology:` section.
- `scripts/v3_test_topology.py`: writes `05_seeds.png` and `06_room_cells.png` under `data/output/debug_v3/<stem>/`.

**Pipeline**

1. Each Phase 1 mask contributes its centroid. If the centroid falls in a hole, it snaps to the nearest pixel of that mask, then onto the footprint.
2. The watershed marker is the mask itself, not a dot. The room keeps the pixels Phase 1 already assigned, and only the gaps in the footprint are flooded. A dot at the centroid is the fallback when the mask is missing.
3. The height map is the inverse distance transform of the footprint, so the dam between two markers prefers the narrow part of the shape.
4. Pixels the watershed leaves on the dam are given to the nearest seed in the same connected component. Specks under 200 px with no seed are absorbed into the nearest room. A larger unseeded piece of floor would stay its own cell; none of the three renders produced one.

**Results**

| Render | Seeds | Cells | Overlap | Footprint pixels left out |
| --- | --- | --- | --- | --- |
| heritage towers | 10 | 10 | 0 | 0 |
| highlandlux citadel | 13 | 13 | 0 | 0 |
| limestone ranch | 11 | 11 | 0 | 0 |

- Heritage keeps the open living, kitchen, and hall as one cell and gives the bedroom, closet, bathroom, tub, and balcony their own cells. The bed is still its own cell inside the bedroom, and a thin fragment remains in the hall, because those were separate Phase 1 masks.
- Highland keeps the kitchen and hall, the living and dining room, the bedroom, the balcony, and the closet. The bed is its own cell. The bathroom tile is still missing, and a few cells sit on the exterior wall face. Both come from Phase 1.
- Limestone keeps the living-room carpet, the bedroom carpet, the kitchen, the closet, and the bathroom. The sofa and the bed stay their own cells inside those rooms.

**Insights**

- **A centroid dot slices a connected floor in half.** The first watershed used only the centroid. Two seeds in one open plan met on the perpendicular bisector, so the heritage hall became a band cut through the living room. Painting the Phase 1 mask as the marker confines the flood to the gaps. Rooms that already had a boundary keep it.
- **The cell complex is exact on these three renders.** No cell shares a pixel with another, no cell leaves the footprint, and every footprint pixel belongs to one cell. Adjacent rooms meet on that shared cut. There is no overlap and no gap for Phase 3 to repair.
- **Phase 2 cannot invent a room Phase 1 never seeded.** Highland's bathroom and limestone's balcony are absent here because they were absent from the footprint masks. Extra Phase 1 masks, such as a bed or a sofa, become extra cells.

**Known limitations**

- A bed or sofa that survived Phase 1 as its own mask remains its own cell. The watershed will not merge it back into the surrounding floor.
- Open doorways stay open when Phase 1 returned one mask for the combined space. The watershed does not add a seed of its own.
- Exterior-face masks from Phase 1 are still cells. They share edges with nothing inside the apartment, but they are part of the complex.

### Phase 3: Manhattan line graph

**Deliverables**

- `src/v3_topology/geometry.py`: `ManhattanLineGraph.extract_segments(cells)` returns the simplified outline of each watershed cell. `orthogonalize(segments)` forces every edge onto 0° or 90° and gives nearby parallel walls one coordinate. `polygons(segments)` returns the closed faces, cutting a room that sits inside another out of its parent.
- `configs/config.yaml`: new `v3_geometry:` section.
- `scripts/v3_test_geometry.py`: writes `07_boundary_segments.png`, `08_orthogonal_lines.png`, and `09_room_polygons.png` under `data/output/debug_v3/<stem>/`.

**Pipeline**

1. Each cell contour is simplified with `approxPolyDP` (8 px). Edges shorter than 12 px are dropped from the raw segment list.
2. Consecutive edges that point the same way are merged, so the outline alternates horizontal and vertical. Each horizontal edge keeps one y, each vertical edge one x, and every corner is their intersection. The ring stays closed.
3. Parallel edges whose coordinates fall within 12 px share one value, weighted by length. The two sides of a shared wall collapse to the same line, and rooms that meet there use the same corner.
4. A face whose centroid lies inside a larger face is subtracted from that parent, so a bed or a sofa becomes a hole rather than a second copy of the floor. Any remaining sliver is given to the smaller room.

**Results**

| Render | Raw segments | Orthogonal segments | Polygons | Cells | Overlap | Polygon area / footprint |
| --- | --- | --- | --- | --- | --- | --- |
| heritage towers | 208 | 108 | 10 | 10 | 0 | 1.066 |
| highlandlux citadel | 204 | 104 | 13 | 13 | 0 | 0.989 |
| limestone ranch | 218 | 107 | 12 | 11 | 0 | 1.071 |

Every orthogonal segment is horizontal or vertical. The faces do not overlap. Their combined area stays within 7% of the walkable footprint.

- Heritage keeps the open living and kitchen as one rectilinear polygon. The bedroom is an L with the bed cut out of it. The closet, bathroom, tub, and balcony are separate polygons. A thin hall fragment remains its own polygon.
- Highland keeps the living room, the kitchen, the bedroom with the bed cut out, the balcony, and the closet. A few polygons still sit on the exterior wall face.
- Limestone keeps the living room with the sofa cut out, the bedroom with the bed cut out, the kitchen, the closet, and the bathroom. One extra face appears where a contour split.

**Insights**

- **Orthogonalizing loose segments left the rooms unclosed.** The first version snapped each edge on its own and then tried to polygonize the graph. Gaps at the corners meant the large rooms never became faces, and heritage kept only 65% of the footprint. Building a closed ring per cell first, then snapping coordinates, is what makes every room a polygon.
- **A shared corner is a shared coordinate, not a shared box.** Clustering parallel edges within 12 px puts both rooms on the same line. Widening that to 18 px started pulling distinct walls together and the overlap came back, so 12 px is the snap that keeps the faces disjoint.
- **Furniture that Phase 1 kept as its own cell becomes a hole in the parent room.** The bed and the sofa are still polygons, and the surrounding floor is the parent with that polygon removed. The areas do not get counted twice.

**Known limitations**

- Walls whose two sides land more than 12 px apart stay as a white gap between polygons. Heritage's central corridor and a few highland doorways show this. The gap is the perspective lean of the original render, which this phase does not warp away.
- A diagonal wall becomes whichever axis it is closer to. The orthogonal polygon is a Manhattan stand-in for the perspective outline, so the area moves by up to about 7%.
- Limestone produces one more polygon than it has cells, from a contour that split while it was being squared.
- Phase 1 fragments (the hall sliver, the exterior faces, the bed, the sofa) are still polygons. Orthogonalization does not merge them.

### Phase 4: Global area scale

**Deliverables**

- `src/v3_topology/scale.py`: `AreaScaleEstimator.square_feet(filename)` reads the last size in the name. `estimate(polygons, filename, image=None)` returns pixels per foot. `result` holds the pixel area, the square footage, the source, and any fixture anchors.
- `configs/config.yaml`: new `v3_scale:` section.
- `scripts/v3_test_scale.py`: prints the filename scale for each render, then calls the estimator again with a name that has no size.

**Pipeline**

1. The filename parser accepts `593 Sq. Ft.`, `623 Sq. Ft.`, and `625sq`.
2. Pixel area is the sum of the Phase 3 polygon areas. A hole cut out of a parent is already excluded from that parent, and the furniture polygon is added once.
3. The scale is \(S = \sqrt{\text{pixel area} / \text{square feet}}\), in pixels per foot. A polygon's square footage is its pixel area divided by \(S^2\).
4. When the name has no size, YOLOv8n looks for a toilet. The shorter side of the box is taken as a 40 cm bowl, and that length becomes pixels per foot. A bathtub class is used the same way at 150 cm when the weights include one. COCO YOLOv8n does not. Untextured gaps are not measured.

**Results**

| Render | Square feet | Pixel area | Scale (px/ft) | Source |
| --- | --- | --- | --- | --- |
| heritage towers | 593 | 236776 | 19.982 | filename |
| highlandlux citadel | 623 | 228237 | 19.140 | filename |
| limestone ranch | 625 | 284860 | 21.349 | filename |

Dividing each pixel area by \(S^2\) returns the filename size on all three renders. The three scales sit near 20 px/ft.

A second call with the filename `unit.webp` ran the fixture fallback on each render. YOLOv8n returned no toilet and no bathtub, so the fallback scale is absent. At a confidence of 0.05 the strongest hits are a microwave, a television, a clock, a plant, and a book.

**Insights**

- **The filename is the scale.** All three sample names carry a size, including limestone's compact `625sq`. The parser takes the last match, so a leading unit number does not become the area.
- **\(S\) is pixels per foot.** The square root converts an area ratio into a length ratio. Phase 5 can label a room as `polygon.area / S^2` square feet.
- **The three renders agree.** Heritage, highland, and limestone land between 19.1 and 21.3 px/ft. That spread is the difference in framing and in how much floor Phase 3 kept, not a parser error.

**Known limitations**

- Missing rooms shrink the pixel area and therefore \(S\). Highland's bathroom and limestone's balcony are still absent. Exterior-face polygons add area and push \(S\) the other way. The scale uses the polygons as Phase 3 left them.
- YOLOv8n, trained on photographs, does not recognize the toilets in these renders, and its class list has no bathtub. The fallback is wired and returns no scale rather than measuring a doorway or a curtain gap. On these three files the filename path is the one that produces a number.
- `models/yolov8n.pt` is downloaded on first use and stays out of git, the same way the FastSAM weights do.

### Phase 5: Blueprint rendering

**Deliverables**

- `src/v3_topology/visualizer.py`: `TopologyBlueprint.render(shape, polygons, scale_px_per_ft)` draws the orthogonal outlines on a white canvas and labels each face. `rooms()` returns the id, relative area, and square footage.
- `configs/config.yaml`: new `v3_blueprint:` section. Wall thickness is the stroke width.
- `main.py`: `python main.py --version v3` runs segmentation, partitioning, orthogonalization, scale, and the blueprint. It writes `blueprint_v3_<stem>.png` and `result_v3.json`.

**Pipeline**

1. Each polygon exterior, and each hole, is stroked in black at 5 px so a shared wall reads as thickness.
2. The label sits at the interior point farthest from the walls, then slides along that cross until the text rectangle is inside the face. An L-shaped room no longer puts its label in the narrow neck.
3. The three lines are the room id, the share of the combined polygon area, and the square footage `pixel area / S^2`. A tall thin room turns those lines on their side. A face that cannot hold the three lines keeps the room id.

**Results**

| Render | Rooms | Scale (px/ft) | Sum of labeled sq ft |
| --- | --- | --- | --- |
| heritage towers | 10 | 19.982 | 593.1 |
| highlandlux citadel | 13 | 19.140 | 622.8 |
| limestone ranch | 12 | 21.349 | 624.9 |

The labeled areas add back to the filename size, within a tenth from rounding. Heritage's open living room is 185.7 sq ft (31.3%). Highland's largest face is 193.3 sq ft. Limestone's two large faces are 163.0 and 157.8 sq ft.

**Insights**

- **Stroke width is the wall.** The faces already share corners, so drawing each outline at a fixed thickness produces the shared wall without a second geometry pass.
- **The geometric centroid is the wrong anchor.** On an L, or on a room with a bed cut out of it, the centroid falls outside the face. The label uses the point farthest from the boundary and then slides so the text stays inside the polygon.
- **Square footage is the global scale applied per face.** Relative area is that face's share of the polygons. The two numbers come from the same pixel areas Phase 4 used, so they add up to the filename.

**Known limitations**

- Gaps left by Phase 3, where the two sides of a perspective wall are more than 12 px apart, stay white. This phase does not close them.
- A bed, a sofa, a hall sliver, or an exterior-face polygon is still its own labeled face. The blueprint names whatever Phase 3 returned.
- The smallest faces have room only for an id. Highland's room 7 and room 9, and limestone's room 6, are in that group.
