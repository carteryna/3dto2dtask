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
| 2 | `topology.py` | Stub |
| 3 | `geometry.py` | Stub |
| 4 | `scale.py` | Stub |
| 5 | `visualizer.py` | Stub |

Implementation proceeds one phase at a time. Phase 2 starts only after Phase 1 is approved.

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
