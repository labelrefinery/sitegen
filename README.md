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
| `/ground_truth/actors` | `foxglove.SceneUpdate` | **held out** — per-part cuboids |
| `/ground_truth/points` | `foxglove.PointCloud` | **held out** — per-point instance id |

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

## Reading it

Drag the file into [Foxglove](https://foxglove.dev) — every topic renders
natively, including the ground-truth cuboids, so you can watch a pipeline's
labels drift against truth.

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

## Not yet

- Cameras. The schemas exist (`RawImage`, `CameraCalibration`); no renderer.
- Mesh geometry. Actors are oriented boxes; raycasting the 3D-ConHE meshes
  would make the clouds look like real equipment and make size estimation
  honest.
- Terrain that changes. The stockpiles are static, so a cut/fill or
  volume-tracking pipeline has nothing to measure yet.

## License

Apache 2.0.
