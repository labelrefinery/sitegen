# sitegen

A synthetic construction-site MCAP generator with **held-out ground truth**,
built to evaluate auto-labeling pipelines that start with no labels at all.

There is no public synthetic construction MCAP. There are CAD model sets
([3D-ConHE](https://www.mdpi.com/2076-3417/14/9/3599)) and static real scans
([Rohbau3D](https://www.nature.com/articles/s41597-025-05827-7)), but nothing
with motion, a sensor, and truth. That last part is the reason to generate
rather than find: *"start unlabeled and improve"* is only a measurable claim if
something knows the right answer and refuses to tell the labeler.

## The contract

One file, two groups of topics, and a rule.

| topic | schema | |
| --- | --- | --- |
| `/lidar/points` | `foxglove.PointCloud` | the sweep, `x y z intensity` |
| `/ego/joint_states` | `foxglove.JointStates` | swing, boom, stick, bucket |
| `/tf` | `foxglove.FrameTransform` | `map` → `lidar` |
| `/gnss` | `foxglove.LocationFix` | ego global pose |
| `/terrain/heightmap` | `foxglove.PointCloud` | published once, at t₀ |
| `/camera/<name>/image` | `foxglove.CompressedImage` | optional, `--camera-hz`; four cameras |
| `/camera/<name>/calibration` | `foxglove.CameraCalibration` | pinhole K and P |
| `/ground_truth/actors` | `foxglove.SceneUpdate` | **held out** — per-part cuboids |
| `/ground_truth/points` | `foxglove.PointCloud` | **held out** — per-point instance id |
| `/ground_truth/camera_instances/<name>` | `foxglove.CompressedImage` | **held out** — per-pixel instance id |

**A labeler reads the first group. Only the scorer reads `/ground_truth/*`.**
Both live in one file rather than two so the truth cannot drift from the data
it describes — the same pass wrote them.

## The scene

A 20-tonne excavator running a dig-swing-dump-return cycle (15 s, five or six
passes to fill a bed), loading an articulated hauler that backs in, is loaded,
and hauls off before a second truck takes its place. Part-way through, the
machine walks 6.5 m along the trench line to a second dig station -- house
squared up over the tracks, boom tucked -- and the hauler repositions with it.
A spotter holds station; a second worker crosses the swing radius around
t=25 s. Two stockpiles sit at the angle of repose, a row of grade stakes runs
along the north edge, and dust kicks up when the loaded truck pulls away.

Five things are deliberate:

**The excavator is articulated.** Base pose plus swing, boom, stick, and bucket
— four DOF on top of the body. Every link is its own ground-truth box, so a
scorer can ask whether a labeler got the *bucket* right, which is the part that
will actually hit something, rather than only whether it drew a plausible blob
around the machine.

**Proprioception is a published topic.** The ego's joint angles are free, exact
labels available offline. They are the seed an unlabeled pipeline bootstraps
from — the same trick STONE uses with a driven trajectory. First thing they buy
you: the sensor rides the house, so the ego's own boom and roof are in the
cloud (about 12% of returns), and nothing removes them for you.

**The difficulty is in the sensor, not the scene.** Points on a target fall off
as 1/r², range noise grows with distance, dropout rises quadratically, and dust
eats returns over a region. A worker at 14 m gets about 20 points; a truck at
45 m gets a handful. That is where naive clustering stops finding things, and
where a smoother that carried the track forward from when it was close starts
to pay.

**The machine travels.** The walk between dig stations is not decoration. A
fixed sensor origin quietly removes three things worth testing: the map never
grows, occlusion behind the stockpile never resolves, and there is no driven
trajectory for an annotation-free terrain labeler to calibrate against --
which is the free supervision STONE-style methods depend on, exactly as joint
angles are for the boom chain.

**Nothing is balanced.** Terrain is ~83% of returns, the two workers together
are ~0.3%. Real class imbalance, not a curated benchmark.

## Use

```sh
make install
uv run sitegen generate --out site.mcap --seed 1 --duration 60
```

```
--seed             scene and noise are deterministic in this
--duration         seconds
--rate             sensor Hz (default 10)
--truth-points-hz  per-point truth rate (default 2); boxes are always at --rate
--difficulty       scales range noise, dropout and dust severity
--azimuth-steps    horizontal resolution (default 450)
```

Roughly 1.4 MB of scene per second at defaults, so a 60 s run is ~85 MB.
Samples are published as release assets rather than committed — pin one by URL
and checksum so two pipelines under comparison consume byte-identical input.

## Reading it in Foxglove

Every topic uses a well-known Foxglove schema, so there is nothing to install
and no custom panel to write.

1. Open [Foxglove](https://foxglove.dev) and drag `site.mcap` in (or `Cmd/Ctrl-O`).
2. Add a **3D** panel. Set **Display frame** to `map`.
3. In the panel's topic list turn on:
   - `/lidar/points` — set *Color by* to `intensity` for a legible sweep
   - `/ground_truth/actors` — the oracle cuboids, coloured by class
   - `/terrain/heightmap` — the static ground and stockpiles
4. Add a **Plot** panel on `/ego/joint_states` to watch the dig cycle —
   `swing`, `boom`, `stick` and `bucket` trace one cycle every 15 s, and the
   flat stretch at t≈26–34 s is the machine walking to its second station.

`/ground_truth/*` is the oracle. It is there so you can *see* what a labeler
was up against; a labeler must never read it.

## Seeing a pipeline's labels against the truth

Predictions go in their own file, so the scene stays immutable and one 86 MB
recording serves any number of pipeline runs:

```sh
sitegen overlay round0.csv round1.csv round2.csv \
    --out labels.mcap --scene site.mcap
```

That writes one `/pred/<name>` topic per input CSV — cuboids plus billboarded
track ids, one categorical colour each — and comes to a few hundred KB.
`--scene` reads `t = 0` from the recording so the two line up exactly.

Then **open both files together**. Foxglove merges local files into a single
playback timeline, which is exactly the "a main recording plus a related file"
case it documents:

```sh
foxglove site.mcap labels.mcap      # or drag both in at once
```

Now the 3D panel has `/ground_truth/actors` and every `/pred/*` set on one
timeline, each independently toggleable. Scrub to t≈25 s and watch a stage
invent objects on the stockpile, or turn two rounds on at once to see which
one stops. If you would rather keep the sources visually separate than merged,
Foxglove's [comparison mode](https://docs.foxglove.dev/docs/visualization/comparison-mode)
does that instead.

From Daft:

```python
df = daft.read_mcap("site.mcap", topics=["/lidar/points"])
```

One wrinkle worth knowing before building on it: Daft returns the payload in a
`String` column holding the Python `repr` of the bytes (`"b'\n\x06...'"`), not
the bytes themselves. It round-trips exactly, but you have to unwrap it:

```python
import ast
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud

raw = ast.literal_eval(row["data"])     # verified byte-identical to mcap.reader
pc = PointCloud()
pc.ParseFromString(raw)
```

## Two pipelines this is for

- **Cold start** — geometry only. Ground fit, cluster, associate, smooth, emit
  pseudo-labels. Score against the oracle.
- **Bootstrap loop** — train a student on that output, run it, feed
  disagreement and low-confidence frames back, re-label, retrain.

The number that makes the case is one curve: label quality against round, with
the oracle as the ceiling.

## The camera, and what it is not

`--camera-hz 2` renders a forward-looking camera on the cab mast, using the
same raycaster as the LiDAR — so a return in the cloud and a pixel in the image
describe the same surface, because the same ray produced both. Per-pixel
instance ids come out free, since the raycaster already knows which box each
ray hit: segmentation ground truth nobody had to draw.

### The rig is four cameras, and the placement matters more than it sounds

A single centre-forward camera spends **22.8% of every frame looking at its own
boom** — and when Grounding DINO was pointed at one, that boom captured the
only detection it made. Real machines mount surround-view at the house corners
for exactly this reason, so `SURROUND_RIG` does too: front-left and front-right
at the corners panned 35° out, left and right on the sides panned 90°.

Measured across seven swing angles, all four cameras: **0.0% ego occlusion,
everywhere.** The cameras ride the house, so as the machine swings they sweep
the whole site, and which one has the clear view changes with the dig cycle.

### What an open-vocabulary detector makes of it

Ten unobstructed views, Grounding DINO, prompt
`"excavator . haul truck . worker . person ."`:

| | |
| --- | --- |
| real construction photograph | `excavator` **0.858**, `haul truck` **0.771** |
| sitegen renders, 9 of 10 | `haul truck` **0.351 – 0.421** — correct label, every time |
| sitegen renders, 1 of 10 | nothing |
| workers, visible in 3 views | **never detected, not once** |
| grade stakes | never detected |

The control proves the model, weights and prompt are fine. What the renders
lack is texture and context, so confidence lands at ~0.38 against ~0.8 on a
photograph — and a person, which is the safety-critical class, is invisible to
it entirely.

**So: enough to build a naming pipeline on, not enough to trust its labels.**
A 9-in-10 hit rate with a consistent correct label is plenty of signal to
develop and test camera → detection → 3D-instance association, because the
truck's real box is known. It is not a basis for *training* on those names, and
anything built this way would name machines while silently leaving people
unnamed.

The two halves turn out to be complementary, which is the useful part:
**geometry finds the worker and cannot name the machine; the detector names the
machine and cannot find the worker.** The 97.5% size split covers the person
case without a model at all. Raycasting the
[3D-ConHE](https://www.mdpi.com/2076-3417/14/9/3599) meshes would narrow the
gap; only real imagery closes it.

That last claim is now measured rather than assumed.
[docs/PROBE-WORKER.md](docs/PROBE-WORKER.md) re-renders these same ten views in
Cycles under an outdoor HDRI with exactly one thing changed — the worker
cuboid replaced by a rigged human in a hi-vis vest and hard hat. The worker
goes from never detected to **11 of 11 sightings, mean 0.544**, labelled
`worker person` every time. The truck, left as the same pair of boxes, does not
move (0.469 to 0.463), and keeping the cuboid under the identical lighting and
ground still detects nothing at all. **What the detector cannot see is the box,
not the scene.**

## Not yet

- Mesh geometry. Actors are oriented boxes; raycasting the 3D-ConHE meshes
  would make the clouds look like real equipment and make size estimation
  honest. The worker probe puts a number on what it would buy on the camera
  side.
- Terrain that changes. The stockpiles are static, so a cut/fill or
  volume-tracking pipeline has nothing to measure yet.

## License

Apache 2.0.
