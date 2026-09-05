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
| `/terrain/heightmap` | `foxglove.PointCloud` | the ground, at t₀ and whenever it changes |
| `/camera/<name>/image` | `foxglove.CompressedImage` | optional, `--camera-hz`; four cameras |
| `/camera/<name>/calibration` | `foxglove.CameraCalibration` | pinhole K and P |
| `/ground_truth/actors` | `foxglove.SceneUpdate` | **held out** — per-part cuboids |
| `/ground_truth/points` | `foxglove.PointCloud` | **held out** — per-point instance id |
| `/ground_truth/camera_instances/<name>` | `foxglove.CompressedImage` | **held out** — per-pixel instance id |
| `/ground_truth/volumes` | `foxglove.JointStates` | **held out** — cut, bucket, bed, hauled, per-pile |

**A labeler reads the first group. Only the scorer reads `/ground_truth/*`.**
Both live in one file rather than two so the truth cannot drift from the data
it describes — the same pass wrote them.

## The scene

A 20-tonne excavator running a dig-swing-dump-return cycle (15 s), loading an
articulated hauler that backs in, is loaded, and hauls off before a second
truck takes its place. Part-way through, the machine walks 6.5 m along the
trench line to a second dig station -- house squared up over the tracks, boom
tucked -- and the hauler repositions with it. A spotter holds station; a second
worker crosses the swing radius around t=25 s. Two stockpiles sit at the angle
of repose, a row of grade stakes runs along the north edge, and dust kicks up
when the loaded truck pulls away.

The ground is a heightfield the bucket cuts into. Each pass takes soil out of
the cut, the wall slumps back to the angle of repose, the material heaps in the
hauler's body, and it leaves the site when the hauler does -- so a 60 s
recording ends with two bowls in ground that started flat, and 2.1 m³ of
material to account for.

Six things are deliberate:

**The excavator is articulated.** Base pose plus swing, boom, stick, and bucket
— four DOF on top of the body. Every link is its own ground-truth box, so a
scorer can ask whether a labeler got the *bucket* right, which is the part that
will actually hit something, rather than only whether it drew a plausible blob
around the machine.

**Proprioception is a published topic.** The ego's joint angles are free, exact
labels available offline. They are the seed an unlabeled pipeline bootstraps
from — the same trick STONE uses with a driven trajectory. First thing they buy
you: the sensor rides the house, so the ego's own boom and roof are in the
cloud — **27% of returns**, against 28.0% measured on a real excavator-mounted
rig — and nothing removes them for you.

**The difficulty is in the sensor, not the scene.** Points on a target fall off
as 1/r², range noise grows with distance, dropout rises quadratically, and dust
eats returns over a region. The numbers are not invented: the elevation band,
the range, the ray budget and the per-class intensity were all measured against
[GOOSE-Ex](docs/GOOSE-EX.md), 2,164 labelled sweeps from four Ousters on a
working Liebherr R924. At the default 32 × 450 ray budget a worker at 14 m gets
**6 returns** and a truck at 45 m about 22; at `--density real`, which spends
the ray budget the real rig has, the worker gets **40** against a real 141.
That is where naive clustering stops finding things, and where a smoother that
carried the track forward from when it was close starts to pay.

**The machine travels.** The walk between dig stations is not decoration. A
fixed sensor origin quietly removes three things worth testing: the map never
grows, occlusion behind the stockpile never resolves, and there is no driven
trajectory for an annotation-free terrain labeler to calibrate against --
which is the free supervision STONE-style methods depend on, exactly as joint
angles are for the boom chain.

**Nothing is balanced.** Terrain is ~70% of returns, the ego's own machine is
**27%**, and the two workers together are **0.08%**. Real class imbalance, not
a curated benchmark — and it got four times as severe over two changes: half
the returns the old worker collected were off a cuboid a person does not fill,
and widening the elevation band to the real machine's put the other half into
the ground.

**The ground remembers.** Every other actor is a pure function of `t`. Terrain
cannot be, because a hole is a record of what happened to it — and that is the
point: a cut/fill pipeline has to *accumulate* rather than recognise. Soil
comes out of the ground at the cutting edge, slumps to the angle of repose,
heaps in the hauler's body and leaves the site with it, and the balance closes
to 10⁻¹³ m³. So `/ground_truth/volumes` is an oracle in the same sense the
cuboids are, and `net` — what differencing two surveys would give you — is
deliberately not `cut`, because material that went to a stockpile is still on
site and material that went out the gate is not.

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
--beams            vertical channels (default 32)
--azimuth-steps    horizontal resolution (default 450)
--density          sample (default) or real: 64 x 1440, the real rig's budget
--sensor           calibrated (default) or legacy, the pre-GOOSE-Ex sensor
--actors           meshes (default) or boxes, the geometry both sensors see
--terrain          deforming (default) or static; see docs/TERRAIN.md
--camera-hz        render the four-camera rig; see docs/RENDERING.md
```

Roughly 1.5 MB of scene per second at defaults, so a 60 s run is ~90 MB in
10 seconds — 8 of them with `--terrain static`, the rest being the ground.
`--density real` is 92,160 rays a sweep instead of 14,400: **590 MB and
25 seconds** for the same 60 s, and the only setting that puts a worker at 14 m
within a factor of three of the real rig. Samples are published as release
assets rather than committed — pin one by URL and checksum so two pipelines
under comparison consume byte-identical input.

## Reading it in Foxglove

Every topic uses a well-known Foxglove schema, so there is nothing to install
and no custom panel to write.

1. Open [Foxglove](https://foxglove.dev) and drag `site.mcap` in (or `Cmd/Ctrl-O`).
2. Add a **3D** panel. Set **Display frame** to `map`.
3. In the panel's topic list turn on:
   - `/lidar/points` — set *Color by* to `intensity`. It is a per-class
     albedo now rather than a stand-in for range, so the workers light up:
     hi-vis PPE returns nearly four times what soil does, which is what it does
     on a real site
   - `/ground_truth/actors` — the oracle cuboids, coloured by class
   - `/terrain/heightmap` — the ground and stockpiles, republished each time
     the bucket changes them: twelve messages in a 60 s recording, every one of
     them inside a dig window
4. Add a **Plot** panel on `/ego/joint_states` to watch the dig cycle —
   `swing`, `boom`, `stick` and `bucket` trace one cycle every 15 s, and the
   flat stretch at t≈26–34 s is the machine walking to its second station.

`/ground_truth/*` is the oracle. It is there so you can *see* what a labeler
was up against; a labeler must never read it.

## Seeing a pipeline's labels against the truth

Predictions go in their own file, so the scene stays immutable and one 90 MB
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

## The camera

`--camera-hz 2` renders the four-camera rig in Cycles from the same actor
meshes the LiDAR just intersected — same triangles, same poses, only the
shading differs. Per-pixel instance ids come out of a second one-sample pass
where every surface emits its own id: segmentation ground truth nobody had to
draw.

A return in the cloud and a pixel in the image still describe the same surface.
That used to be true by construction, because one raycaster drew both; with two
renderers it is a claim about intrinsics, extrinsics and two optical
conventions agreeing, so it is now measured. `tests/test_same_surface.py`
projects LiDAR returns into every camera using only what the recording
publishes and checks the held-out mask at that pixel: **99.84% land on their own
instance, three of the four cameras exactly.**

[docs/RENDERING.md](docs/RENDERING.md) has the whole of it — what each sensor
sees, asset provenance, timings, and the acceptance table below in full.
`--actors boxes` and `--camera-renderer raycast` still select the original
cuboid geometry and flat-shaded renderer, and `--sensor legacy` the
pre-calibration LiDAR; `--actors boxes --sensor legacy` reproduces the old
recordings byte for byte.

### The rig is four cameras, and the placement matters more than it sounds

A single centre-forward camera spends **22.8% of every frame looking at its own
boom** — and when Grounding DINO was pointed at one, that boom captured the
only detection it made. Real machines mount surround-view at the house corners
for exactly this reason, so `SURROUND_RIG` does too: front-left and front-right
at the corners panned 35° out, left and right on the sides panned 90°.

Across all 480 instance masks of a 60 s recording, all four cameras: **the ego
machine occupies 0 pixels.** The cameras ride the house, so as the machine
swings they sweep the whole site, and which one has the clear view changes with
the dig cycle.

### What an open-vocabulary detector makes of it

Ten views chosen by the held-out masks, Grounding DINO, prompt
`"excavator . haul truck . worker . person ."`, threshold 0.35. The baseline
column is the *same ten views* re-rendered as boxes, so this is paired:

| | boxes + raycast | meshes + Cycles |
| --- | --- | --- |
| **worker**, 11 sightings | 1 of 11, best 0.352 | **11 of 11**, 0.585–0.711, mean 0.660 |
| **haul truck**, 5 sightings | 5 of 5, 0.370–0.443, mean 0.397 | **5 of 5**, 0.686–0.710, mean 0.699 |
| real construction photograph | | `excavator` 0.858, `haul truck` 0.771 |

**The complementary-halves asymmetry is gone.** The README used to say that
geometry finds the worker and cannot name the machine while the detector names
the machine and cannot find the worker; that was a property of the box
renderer, and [the probe](docs/PROBE-WORKER.md) proved it by changing one thing.
Now the detector finds every worker, names it `worker` or `person` every time,
and scores the haul truck within a hundredth of what it scores a real one in a
photograph. The wheels did most of that: the box truck was two floating
cuboids.

**What that is now enough for**, and what it still is not. Camera → detection →
3D-instance association can be built and *evaluated* on this, including for the
safety-critical class, because every detection has a known 3D box behind it.
It is still not evidence that a student trained on these renders transfers to
real imagery — a detector recognising a stylised human is a much weaker claim
than its features being useful training signal. Only real imagery settles that.

Two costs came with it. The detector now puts a second, confident `excavator`
label on the haul truck in half the views (0.442–0.486, IoU 0.97 with the
truck's own pixels); argmax still resolves it, but a pipeline consuming raw
detections has to. And the *excavator itself cannot be tested from this rig at
all* — there is one excavator on the site and it is the one carrying the
cameras. Asked from a camera placed off the machine, the same weights score the
excavator mesh **0.807** and **0.809**, against 0.858 for a real one.

## The ground

`--terrain deforming`, the default, makes the site a 0.25 m elevation grid over
the working area, cut by the bucket's own sole and relaxed to the 34° angle of
repose after every bite, with the removed volume conserved all the way to the
hauler driving it off site. Both sensors see it — the LiDAR through a BVH
rebuilt only when the ground changes, Cycles from the same vertex array — and
`tests/test_same_surface.py` reports **0.9980**, identical to four figures with
the terrain static, because ground, pile and cut all carry instance id 0.

The material balance is held out on `/ground_truth/volumes` and exports with
`sitegen volumes`. It costs **1.28×** the generation time: 10.4 s for a 60 s
scene against 8.1 s, of which half a second is the earthworks pre-pass and half
a second is twelve BVH rebuilds at 39 ms each. `--terrain static` is the
analytic plane and cones, and with `--actors boxes --sensor legacy` still
reproduces the pre-terrain recordings byte for byte.

[docs/TERRAIN.md](docs/TERRAIN.md) is the whole of it, including the part that
is not flattering: the scripted dig stroke works the same 1.5 m of face every
cycle without advancing the cut, so after the first bite each pass recovers
only what slumped back in. A 60 s recording moves 2.1 m³ rather than the six
heaped buckets the cycle description implies, and the one near-full bucket in
it is the first pass at the second dig station. Fixing that means changing the
published joint trajectory, which is a scene change and not a terrain one.

## License

Apache 2.0.
