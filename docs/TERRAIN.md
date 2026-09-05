# Terrain that changes

The README's last "Not yet" said the stockpiles were static, so a cut/fill or
volume-tracking pipeline had nothing to measure. It does now. The bucket takes
soil out of the ground, the ground remembers, the hauler's body fills over
several passes and drives the material off site, and every cubic metre of it is
published on a held-out topic so a pipeline that claims to have recovered the
earthwork can be scored rather than admired.

`--terrain deforming` is the default. `--terrain static` is the analytic plane
and cones the scene had before, kept because a byte-reproducible baseline is
worth more than the code it costs.

## The model, and what it is not

AGX Terrain and Vortex Studio solve this problem properly: a heightfield with a
few thousand particles under the tool, so material in front of the cutting edge
piles up, shears, and falls back. That is the right model when the question is
what the *machine* feels, and it is the wrong one here. The question sitegen
asks is what a LiDAR sees and what a volume-tracking pipeline can recover from
it, and for that a plain elevation grid answers both at a price the generator
can pay six hundred times a recording.

So: **a heightfield with mass conservation.**

| | |
| --- | --- |
| patch | ±25 m, the site's whole working area |
| grid | 201 × 201 nodes, **40,401**, at **0.25 m** |
| triangles | **80,000**, one BVH |
| angle of repose | **34°**, the value `terrain.py` already drew the cones at |
| bucket | **1.2 m³** heaped, a 1.0 m wide cutting edge, 1.28 m sole |
| hauler body | **7.2 m³**, six heaped buckets |
| surface commit rate | **1 Hz**, and only when something changed |
| simulation step | 20 Hz, independent of `--rate` |

Outside the patch the ground is still the analytic plane at z = 0, which is
exact and free. The border ring of the grid is pinned to zero, so the mesh and
the plane meet without a seam, and a ray that hits the plane *inside* the
patch footprint is discarded — otherwise the plane would floor over every cut.

**0.25 m is the bucket's number, not a round one.** The cutting edge is 1.0 m
across and the sole is 1.28 m long, so a bite covers about 4 × 5 cells: fine
enough that a bite is a shape rather than a single cell, coarse enough that the
patch is 80,000 triangles instead of the 1.3 million a 0.05 m grid would be.
The bucket's own curvature is a couple of centimetres of profile, which is
under the grid either way — hence the cut floor is the underside of the posed
bucket's bounding box rather than the scoop's actual surface.

### Cutting

Every 50 ms the bucket's sole — the four corners of the underside of its posed
bounding box — is rasterised onto the grid and the surface is lowered to the
plane through them, interpolated barycentrically because a posed quad's four
corners are not coplanar. The volume removed is measured off the grid and rides
in the bucket as a scalar. If the geometric bite is larger than what a bucket
holds, the depth is scaled down uniformly so the removed volume is exactly
1.2 m³: a partial bite, which is what an operator feathering the stick gets.

Nothing says *when* to dig. The bucket cuts where its sole is below the surface
and does not where it is not, and that single rule reproduces the cycle: the
bucket is in the ground from t = 12.1 s to 15.9 s of every fifteen-second loop
— the tail of the swing back, the settle over the cut, and the first second of
the next stroke — and the walk between dig stations digs no trench because the
scripted travel pose has the boom tucked and the teeth 4.6 m in the air.

### Where it goes

During the dump phase (0.47 to 0.60 of the cycle) the bucket empties. If a
hauler is standing at the loading spot the soil goes into its body; if the body
is full, or nothing is standing there, it goes to the nearest stockpile, which
is what an operator does with it. The body's load is a heightfield of its own,
0.20 m in the bed link's frame, relaxed by the same code with the grid border
standing in for the body sides — so the load heaps against the headboard at
the angle of repose instead of lying flat, and the visible heap and the number
in the oracle are one quantity rather than two that have to be kept in step.
At 1.0 m³ the heap stands 0.68 m proud of the floor.

When a hauler leaves the scene it takes its load with it: that volume moves to
`hauled`, and it is the one number no survey of the site could ever recover.

### The angle of repose

After every edit the affected window is slope-limited until no two neighbouring
nodes differ by more than one cell of run at 34°. The iteration is a set of
symmetric pairwise transfers, so the window's total is invariant — which is
what makes the mass balance an assertion rather than an estimate. The window
grows until the slump stops touching its border, so material never piles
against a wall that is not there.

A cross-section of the first dig station's cut at the end of a 60 s recording,
along y = -3.25 m, node by node:

```
0.00  0.00  -0.09  -0.26  -0.43  -0.60  -0.77  -0.60  -0.43  -0.26  -0.09  0.00
```

0.169 m of fall per 0.25 m of run, everywhere, which is 34.0° to three figures
— the bowl is *exactly* at the limit, because that is what a slope limiter
converges to. 2.25 m across, 0.77 m at the deepest. From a camera on the house
9 m away it reads as a sharp-walled trench; it is a cone.

The initial grid is sampled from the analytic cones, which were already drawn
at 34°, so it satisfies the limit the moment it exists and the first frame of a
recording is not a landslide. `tests/test_terrain.py` checks that rather than
assuming it.

**Determinism.** The relaxation has no random component at all, so it is
reproducible under any seed by construction. What `--seed` moves is where a
bucket lands on a stockpile: a 0.35 m draw either way, so the growing pile is
not perfectly axisymmetric. Nothing in the volumes depends on it.

## Mass conservation

At every instant, and to floating-point round-off rather than a tolerance:

```
cut == bucket + bed + hauled + (stockpiles - stockpiles at t0)
net == (stockpiles - stockpiles at t0) - cut
```

The first is bookkeeping against bookkeeping. The second is bookkeeping against
the grid — `net` is measured by summing elevations and multiplying by cell
area, so the two agree only if the relaxation moves soil rather than rounding
it away, which is the failure mode a slope limiter written the obvious way has.

Worst residual over a 120 s recording, 2,401 simulation steps:
**1.0 × 10⁻¹³ m³.** The test's bar is 10⁻⁹ m³ — a cubic millimetre — which is
four orders of magnitude of headroom over round-off and would catch a leak of
one cell's worth of soil anywhere in the run.

## What the two sensors see

Both. The LiDAR intersects the patch mesh through embree and Cycles renders the
same vertex array, handed over as one `.npy` of committed elevations because
the grid's topology never changes and only its z does. `docs/RENDERING.md`
used to say "terrain stayed analytic, and Blender builds the identical plane
and 128-gon cones"; that is now only true of `--terrain static`.

The BVH is the reason for the 1 Hz commit rate. Rebuilding 80,000 triangles
costs **39 ms** — affordable once a second, and four times the whole frame
budget twenty times a second. So the ground the sensors intersect is the most
recent *committed* snapshot, and a cut appears up to one second after the
bucket made it. Reducing the update rate was the right lever rather than
coarsening the grid: the grid resolution is what makes a bite a shape, and a
one-second lag on a sixty-second earthwork is not a thing a labeller can see.

Querying it costs **2.2 ms** for a 14,400-ray sweep, against roughly the same
for the two analytic cones it replaced.

**Agreement did not move.** `tests/test_same_surface.py` on the same 2 s
recording, static and deforming:

```
static     same-surface agreement 0.9980 over 14657 returns
  front_left 0.9933 (3856)  front_right 1.0000 (3827)
  left 0.9991 (3500)        right 1.0000 (3474)

deforming  same-surface agreement 0.9980 over 14657 returns
  front_left 0.9933 (3856)  front_right 1.0000 (3827)
  left 0.9991 (3500)        right 1.0000 (3474)
```

Identical to four figures, per camera. That is not luck: ground, pile, sky and
now the cut all carry instance id 0, so a disagreement between the two
renderers about a terrain silhouette is invisible in the masks by construction.
The 0.67% `front_left` disagreement is the pre-existing one — parallax between
a LiDAR on the mast and a camera at the house corner, at an actor's edge.

## The contract

### `/terrain/heightmap` — observable, now republished

Was one `foxglove.PointCloud` at t₀. Still one at t₀, and now one more each
time the ground changes, bounded by the commit rate. A consumer that reads the
first message and stops is exactly where it was.

A 60 s `--seed 1` recording publishes **12** of them, at t = 0, 1, 13, 14, 15,
16, 43, 44, 45, 46, 58 and 59 s — which is the dig windows and nothing else.
Each is 10,201 points: the patch decimated to **0.5 m**, because that is what a
drone survey of a site actually delivers and it keeps the topic near the 14,400
points it published as a one-off before. 1.96 MB of payload across the run,
94 KB on disk after chunk compression, on a 90 MB file.

The exact surface is 0.25 m and the oracle integrates that one, so the
published product is coarser than the truth by design. Worth being blunt about
the other half of it: this topic is the answer to the easy version of the
question. A pipeline scored on `/ground_truth/volumes` should be building its
surface out of `/lidar/points` — sparse, noisy, self-occluded, and dusty when
the truck pulls away — because integrating a published heightmap is not
evidence of anything.

### `/ground_truth/volumes` — held out

`foxglove.JointStates`, once a second, seven named doubles:

| channel | |
| --- | --- |
| `cut` | cumulative m³ taken out of the ground |
| `bucket` | what is in the bucket right now |
| `bed` | what is in the hauler standing on site right now |
| `hauled` | cumulative m³ that has left the site |
| `stockpile_0`, `stockpile_1` | m³ standing in each pile's footprint |
| `net` | the ground's own volume change since t₀ |

`JointStates` for a volume record needs a word. Foxglove has no generic
named-scalar schema; `KeyValuePair` carries no timestamp and is not plottable,
and a `SceneUpdate`'s metadata is strings. `JointStates` *is* a timestamped
vector of named doubles, the Plot panel already reads it, and the README
already tells people to open one on `/ego/joint_states`. Using a well-known
schema so that nothing has to be installed is the point of the contract, and
this is the well-known schema that fits the shape.

Only the scorer reads it. `sitegen volumes site.mcap --out volumes.csv` exports
it as `t` plus one column per channel, in the order the recording wrote them.

### How a cut/fill pipeline would be scored against it

Build a surface per time window out of `/lidar/points` and `/tf`; difference it
against the surface from the first window; integrate. That estimate is
comparable with `net` directly. Three things make it harder than it sounds, and
they are the reason the topic is worth having:

- **The machine occludes its own cut.** The sensor rides the house 4.4 m above
  a bowl it is standing next to, and 27% of every sweep is the ego's own boom.
  The floor of the cut is visible for part of each swing and not for the rest.
- **`net` is not `cut`.** Material that went to a stockpile is still on site
  and cancels in the difference; material that left in a hauler does not. A
  pipeline that reports `net` is answering the survey question; one that
  reports `cut` and `hauled` separately has had to track the trucks as well as
  the ground, which is the whole reason both sensors are in the same file.
- **The stockpiles move too, by very little.** Pile 1 gains 0.047 m³ over
  120 s on a standing 21.3 m³ — two parts in a thousand, well under the
  surface noise a sparse sweep reconstructs at. Recovering that is a harder
  claim than recovering the cut, and the oracle says exactly how much there
  was to find.

## What the scene actually moves, which is less than the cycle implies

Honest numbers, `--seed 1`, per dig pass:

| pass | station | m³ |
| --- | --- | ---: |
| t = 0.0–0.8 s | 1 | 0.36 (the recording starts mid-stroke) |
| t = 12.0–15.4 s | 1 | 0.45 |
| t = 42.0–45.6 s | **2, virgin ground** | **1.03** |
| t = 57.1–60.0 s | 2 | 0.05 |
| every pass after that | 2 | 0.01–0.03 |
| **60 s total** | | **2.13** |
| **120 s total** | | **2.25** |

The one full bite is the first one at the second dig station, and that is the
whole story: **the scripted trajectory drags the cutting edge through the same
1.5 m of face every fifteen seconds without advancing the cut.** After the
first bite each pass recovers only what slumped back in, the bowl widens, and
the yield converges toward zero. That is not a bug in the model — it is what
digging repeatedly into one spot in cohesionless material does, and it is
exactly why a real operator advances the face or tracks the machine back.

The scene tracks the machine back once, at t = 26–34 s, and the pass after it
is worth a bucket. Everything else is the same hole.

Two consequences worth stating rather than tuning away:

- **The capacity cap almost never binds.** 1.2 m³ is the heaped bucket of a
  20-tonne machine and the cut is bounded by it, but only the t = 42 s pass
  gets close.
- **A body does not fill in a 60 s recording.** It reaches 1.03 m³ of 7.2, and
  the second hauler's does the same. `--duration 300` gets further; six passes
  filling a body needs a dig stroke that advances the face, and that is a
  change to the published joint trajectory rather than to the terrain model.

Making the cut productive would mean changing `Scene.ego_at`, which is
published truth: `/ego/joint_states` is the free supervision this whole dataset
is built around, `docs/GOOSE-EX.md`'s sensor statistics were measured through
it, and `--terrain static --actors boxes --sensor legacy` reproduces the
pre-terrain recordings byte for byte only while it stands. That is a scene
change, and it is the first thing to do if this ever needs to move more soil.

The same constraint explains a second thing a viewer will notice. The dump pose
carries the cutting edge to (0.0, 7.6, 8.8) while the body sits at
(7.8, 7.8, 2.0) — the bucket is beside and above the hauler, not over it. Where
the material goes is decided by the cycle phase, not by whether the bucket is
geometrically over the body. Same reason.

## Cost

`--seed 1 --duration 60`, M-series Mac, no camera:

| | |
| --- | --- |
| `--terrain static` (and every recording before this) | **8.1 s** |
| `--terrain deforming` | **10.4 s** |
| of which: the earthworks pre-pass | 0.5 s |
| of which: 12 terrain BVH rebuilds at 39 ms | 0.5 s |
| of which: 600 terrain ray queries at 2.2 ms | 1.3 s |
| file size | 89.9 MB, +94 KB |

**1.28×**, against a budget of 2×.

The pre-pass is why the numbers are this small. The scene is otherwise a pure
function of `t` and terrain cannot be, because a hole is history — so the whole
run is simulated once, up front, at 20 Hz, and both the writer's main loop and
the Cycles shot list read the same indexable timeline afterwards. Two callers
asking for t = 41.3 s still get the same answer for the same reason they always
did.

Cycles pays a smaller price than it looks: the patch is one mesh datablock for
the whole render, and a changed snapshot rewrites its z coordinates rather than
rebuilding 80,000 triangles.

## Deliberately not modelled

- **Particles.** No material in front of the cutting edge, no shear plane, no
  spillage off the sides of the bucket. The bite is the swept sole, capped at
  capacity.
- **Compaction and swell.** Soil bulks 25–30% when you dig it up; here a cubic
  metre in the ground is a cubic metre in the body. Modelling swell would mean
  giving up exact conservation, or tracking in-situ and loose volumes as two
  quantities — worth doing the day a pipeline is scored on tonnes.
- **Soil types.** One material, one angle of repose. The scene's two stockpiles
  are labelled `soil` and `gravel` and behave identically, and the cut wall
  slumps like a spoil pile because the model has no cohesion — a trench in clay
  would stand much steeper, and briefly.
- **Water, freeze, traffic.** Nothing compacts the haul road, nothing runs off.
- **The load in the bucket.** The soil is a scalar between the cut and the
  dump; the bucket renders empty. It is 1.2 m³ inside a scoop that is 1.2 m³,
  visible from a mast 4 m above it for two seconds a cycle.
- **The stockpiles as separate objects.** They are part of the one ground
  surface, which is why they now render in the ground material rather than the
  darker pile one: material scraped off the floor and material dumped on a pile
  is the same soil, and the grid cannot tell them apart.
