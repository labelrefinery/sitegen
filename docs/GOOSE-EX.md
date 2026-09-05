# GOOSE-Ex: sitegen's sensor model, measured against a real excavator

sitegen's README makes a claim it never checked:

> Points on a target fall off as 1/r², range noise grows with distance,
> dropout rises quadratically, and dust eats returns over a region. A worker
> at 14 m gets about 20 points; a truck at 45 m gets a handful.

Every number there was chosen to make clustering fail at an interesting
distance. None of it came from a sensor. This document is what happened when
the same quantities were measured on
[GOOSE-Ex](https://goose-dataset.de/) — a Liebherr R924 tracked excavator
carrying four Ouster scanners, working real landfill, quarry and construction
sites, with 5,000 hand-labelled frames carrying **pointwise class *and*
instance** annotations.

The instance ids are what make the comparison possible. With them you can ask
the only question that decides whether an auto-labeler has anything to work
with — *how many returns land on a person at 20 m* — and get an answer from a
machine that exists.

**The short version: sitegen's falloff *shape* is right for people and wrong
for vehicles, its absolute density is 29× too low, and it under-states the ego
machine by half.** Details and specific parameter changes below.

## Where the data is

`../data/goose-ex/` — outside this repo, and staying there: GOOSE-Ex is
CC BY-SA 4.0, and ShareAlike would attach to anything derived from it that
this Apache-2.0 repo redistributed. Provenance, URLs, sizes, checksums and the
attribution requirement are recorded in
[`data/goose-ex/SOURCE.md`](../../data/goose-ex/SOURCE.md).

All six official archives were taken, 31.3 GB compressed:

| | train | val | test |
| --- | ---: | ---: | ---: |
| `gooseEx_3d_*.zip` | 9.49 GB | 853 MB | 1.70 GB |
| `gooseEx_2d_*.zip` | 15.36 GB | 1.46 GB | 2.46 GB |

Only 3D is used here. **Test labels are withheld** for the public Codabench
benchmark, so train and val are the measurable set.

## What it contains

5,000 labelled frames — a pointwise-annotated LiDAR cloud paired 1:1 with a
pixel-annotated image — over 100 sequences recorded across Germany in about a
year:

* **2,800 frames from ALICE**, the drive-by-wire Liebherr R924 tracked
  excavator. This is the platform that matters here; a ~24-tonne tracked
  excavator is within a few tonnes of the 20-tonne machine sitegen simulates.
  The loader finds 2,164 train + 192 val + 444 test = 2,800, which is exactly
  the paper's figure — a useful check that nothing is being silently missed.
* **2,200 frames from a Boston Dynamics Spot** quadruped — a different sensor
  height and a different world, ignored below.

By environment: generic 1,560, landfill 1,572, quarry 1,226, construction site
642. Split 3,989 / 407 / 604 train / val / test.

## Sensor facts

### LiDAR

The excavator carries **four Ouster scanners**, and the labelled data ships
only their **merged** cloud:

| | paper (arXiv:2409.18788) | platform docs (goose-dataset.de/docs/alice) |
| --- | --- | --- |
| house-mounted | 3 × Ouster **OS1-64** Rev6 | 3 × Ouster **OS0-64** Rev C |
| boom-mounted | 1 × Ouster **OS1-128** Rev6 | 1 × Ouster **OS0-128** |

**The two sources disagree**, and it matters: OS1 is a 45° vertical FOV /
~120 m sensor, OS0 a 90° / ~50 m one. The data settles it in the paper's
favour — returns reach **104.2 m**, well past any OS0's usable range — so the
OS1 reading is the one used below. Treat the exact model as unresolved.

* **Channels**: 3 × 64 + 1 × 128 = **320 beams** across the rig.
* **Merged cloud rate**: 10 Hz (`/pcl_merging/merged_cloud`). The labelled
  frames are sampled far more sparsely than that — roughly one every 5 s.
* **Mounting**: three on the **house** (cabin), one on the **boom**. The boom
  unit moves with the arm, so its extrinsic is not static; the authors merge
  all four into the **INS/IMU frame** for exactly this reason, which is why
  only a merged cloud is published. Figure 3 of the paper gives sensor heights
  of 0.73–1.26 m *relative to the IMU frame*, on a machine 12.82 m long and
  2.99 m wide.
* **Rotation rate, per-channel elevation pattern**: **not recoverable from
  this release.** Four differently-oriented sensors merged into one frame have
  no single ray pattern, and the paper cites the Ouster datasheet rather than
  quoting figures. Getting a per-scanner elevation pattern means the raw ROS
  bags from <https://goosedb.uni-koblenz.de/>, not the labelled ZIPs.
* **Measured range envelope** (ALICE, 2,164 train frames): median **4.5 m**,
  p99 **18.2 m**, max **104.2 m**; excluding the ego machine's own returns,
  median **5.1 m**, p90 **11.3 m**, p99 **20.7 m**. See the histogram below —
  this sensor spends almost all of itself inside 10 m.
* **Intensity encoding**: raw Ouster **remission**, stored as `float32` but
  valued 0–255 with a long tail. Not normalised, not calibrated, not
  comparable across sensors.

### Cameras

* 2 × **JAI FSFE-3200D** prism cameras — 3.2 MP, **RGB and NIR
  simultaneously**, 6 mm Fujinon lens, 5 Hz, 59° hFOV, front-left on the cabin.
* 4 × **Alvium G1-240C** — 2.4 MP RGB, global shutter, 5 Hz, **110° hFOV**,
  at the cabin corners. sitegen's four-camera `SURROUND_RIG` at the house
  corners turns out to match real practice closely.
* Intrinsics: pinhole + checkerboard, published on `camera_info` in OpenCV
  form. Extrinsics: AprilTag target with three circular holes, matched in both
  cloud and image; recalibrated between recordings, so it drifts slightly.

### Coordinate frames and sync

Everything is expressed relative to the **IMU/INS frame** of an SBG Ekinox-D,
which is also the PTP grandmaster for the whole sensor network at 200 Hz. The
merged cloud's origin therefore sits at the INS, near the machine base and
close to ground level — the clouds bottom out around z ≈ 0 and the ego
machine's own returns rise to z ≈ +7.8 m. No `base_link`-style frame names are
published beyond "IMU frame".

**Every range in this document is measured from that machine-base origin**,
not from a scanner. sitegen measures from its sensor, ~3 m up. At the
distances being compared the discrepancy is under 2%, and machine-relative
range is the more meaningful quantity for a working machine anyway.

### How much of the sweep is the machine itself

**28.0%.** Well over a quarter of every merged sweep is `ego_vehicle` — the
excavator's own house, boom and bucket. It reaches 8.2 m from the origin and
7.8 m up, which is the boom. (25.9% on the val split; 28.0% across the 2,164
train frames.)

sitegen's README says "about 12%". Its actual output is 13–15%, and the real
machine is **roughly double that**. Note the paper's Fig. 2 omits
`ego_vehicle` from its point-cloud statistics — the class is present and
labelled in the data, just excluded from that figure — so this number does not
appear anywhere upstream.

### Label ontology

64 classes, shared unchanged with the original GOOSE dataset, shipped as
`goose_label_mapping.csv` (`class_name,label_key,has_instance,hex`) in every
archive. 36 occur in GOOSE-Ex. Mapped onto sitegen's four:

| sitegen | GOOSE-Ex | key | notes |
| --- | --- | ---: | --- |
| `excavator` | `heavy_machinery` | 57 | no excavator class; also covers loaders, dozers |
| `haul_truck` | `truck`, `trailer` | 34, 37 | `car` (12) is separate |
| `worker` | `person` | 14 | `rider` (32) is separate |
| `grade_stake` | `pole` | 45 | nearest instanced match; not exact |
| terrain | `soil`, `gravel`, `asphalt`, `cobble`, `low_grass` | 31, 24, 23, 3, 50 | a real site is not one material |
| ego | `ego_vehicle` | 8 | |

`has_instance` marks which classes carry instance ids — all the object classes
do, none of the terrain or vegetation classes do.

### File format

SemanticKITTI convention, in two mirrored trees:

```
lidar/<split>/<platform>_scenario<NN>/<...>_pcl.bin      float32 (N,4)  x y z remission
labels/<split>/<platform>_scenario<NN>/<...>_goose.label uint32  (N,)   sem = v & 0xFFFF
                                                                        inst = v >> 16
```

Filenames carry the capture timestamp in nanoseconds, so playback timing is
recoverable. Scenario directories are prefixed `alice_` or `spot_`.

## Running the loader

`tools/goose/` is a standalone uv project — it does not import `sitegen`, and
`sitegen` does not import it.

```sh
cd tools/goose

# The comparison: profile, sweep-level table, and both points-on-target curves.
uv run python compare.py \
    --goose ../../../data/goose-ex/gooseEx_3d_val \
    --scene /tmp/site.mcap \
    --plot ../../docs/points-on-target.png

# Optional: wrap GOOSE-Ex frames in sitegen's own MCAP contract, so the two
# recordings open in one Foxglove window on one timeline.
uv run python to_mcap.py \
    --goose ../../../data/goose-ex/gooseEx_3d_val \
    --scenario alice_scenario02 --limit 20 --out goose_alice.mcap
```

The sitegen side is generated with:

```sh
uv run sitegen generate --out /tmp/site.mcap --seed 1 --duration 60
```

`--platform spot` switches to the quadruped; `--limit N` caps the frames read.

`compare.py` reduces both datasets to the same thing — one
`goose.Observation` per instance per frame, carrying a range and a point
count — so the two are compared on identical axes despite arriving by
completely different routes. sitegen knows which box each return came from
because it fired the ray; GOOSE-Ex knows because somebody labelled it.

Points on target are counted **per object, not per part**: sitegen's
articulated excavator is five boxes, but a labeler has to find *the machine*.
The ego machine is excluded from both curves — it is always there, and
proprioception gives it away for free.

## The comparison

**2,164 labelled ALICE frames** (the whole train split, 605 M returns) against
a 60 s sitegen scene at defaults (seed 1, 600 sweeps, 120 truth clouds). The
val split (192 ALICE frames, 51.5 M returns) was run separately and agrees
throughout to within a few percent, so nothing below rests on one scenario.

### Sweep level

| | GOOSE-Ex ALICE | sitegen | sitegen / real |
| --- | ---: | ---: | ---: |
| returns per sweep | 279,555 | 9,607 | **0.03×** |
| ego machine share | 28.0% | 15.0% | 0.54× |
| median return range | 4.5 m | 14.5 m | 3.21× |
| p99 return range | 18.2 m | 52.8 m | 2.89× |
| max return range | 104.2 m | 54.2 m | 0.52× |

### Where the returns actually are

| bin | share | cumulative |
| --- | ---: | ---: |
| 0–2 m | 3.40% | 3.40% |
| 2–5 m | 55.93% | 59.33% |
| 5–10 m | 29.82% | **89.15%** |
| 10–15 m | 7.85% | 96.99% |
| 15–20 m | 2.26% | 99.25% |
| 20–30 m | 0.68% | 99.93% |
| 30–40 m | 0.06% | 99.98% |
| 40–60 m | 0.01% | 100.00% |
| 60–120 m | 0.00% | 100.00% |

**89% of a real excavator sweep lands within 10 m**, and 97% within 15 m.
sitegen's median return is at 14.5 m. This is the single largest structural
difference between the two.

### Points on target — person

| range | GOOSE-Ex median (IQR) | n | sitegen median (IQR) | n | sitegen / real |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–5 m | 1,076 (671–1,609) | 78 | — | — | — |
| 5–10 m | 298 (220–414) | 650 | 38 (22–44) | 7 | 0.13× |
| 10–15 m | 141 (97–195) | 1,012 | 23 (21–24) | 63 | **0.16×** |
| 15–20 m | 63 (43–92) | 862 | 12 (11–15) | 106 | 0.19× |
| 20–25 m | 34 (24–47) | 493 | 5 (5–10) | 38 | 0.15× |
| 25–30 m | 19 (12–26) | 220 | — | — | — |
| 30–40 m | 12 (7–17) | 364 | — | — | — |
| 40–50 m | 8 (5–12) | 229 | — | — | — |
| 50–80 m | 4 (3–7) | 208 | — | — | — |

4,116 person observations. The `sitegen / real` column is 0.13, 0.16, 0.19,
0.15 across four consecutive bins — flat, which is the whole finding.

### Points on target — vehicle

| range | GOOSE-Ex median (IQR) | n | sitegen median (IQR) | n | sitegen / real |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–5 m | 9,719 (8,758–14,432) | 41 | — | — | — |
| 5–10 m | 6,435 (3,552–11,929) | 436 | 792 (786–889) | 78 | 0.12× |
| 10–15 m | 1,737 (836–2,945) | 796 | 484 (415–580) | 15 | 0.28× |
| 15–20 m | 39 (20–208) | 1,572 | 225 (220–291) | 7 | **5.77×** |
| 20–25 m | 76 (17–235) | 1,431 | 132 (76–174) | 4 | 1.74× |
| 25–30 m | 17 (7–58) | 1,503 | 108 (68–127) | 6 | **6.38×** |
| 30–40 m | 9 (4–21) | 1,694 | 70 (62–78) | 4 | **7.72×** |
| 40–50 m | 4 (2–9) | 779 | 32 (31–34) | 4 | 7.88× |
| 50–80 m | 3 (2–6) | 1,851 | 20 (18–23) | 2 | 6.83× |

10,103 vehicle observations. The real column is not monotonic — the 15–20 m
bin (39) sits below 20–25 m (76) — because which machines are parked at which
distance varies by scenario, and the wide IQRs (20–208, 17–235) say the same
thing: at a given range a real vehicle is either in clear view or mostly
behind something. That bimodality is itself a finding; see below.

![points on target vs range](points-on-target.png)

### Remission is a class signal, and sitegen throws it away

| class | remission mean | median | p95 |
| --- | ---: | ---: | ---: |
| **person** | **62.1** | 28.0 | **243.0** |
| container | 58.6 | 41.0 | 172.0 |
| car | 48.6 | 11.0 | 243.0 |
| building | 29.6 | 23.0 | 78.0 |
| pole | 21.4 | 10.0 | 123.2 |
| tree_trunk | 20.6 | 18.0 | 41.0 |
| ego_vehicle | 20.4 | 9.0 | 100.0 |
| truck | 19.9 | 15.0 | 51.0 |
| heavy_machinery | 19.9 | 15.0 | 50.0 |
| low_grass | 13.5 | 12.0 | 28.0 |
| asphalt | 10.9 | 10.0 | 24.0 |
| **soil** | **8.3** | 7.0 | 20.0 |
| gravel | 7.6 | 7.0 | 18.0 |
| water | 5.6 | 5.0 | 13.0 |

**A person is the brightest class on a construction site** — 7.5× soil on the
mean, 3× the heavy machinery beside them, and a p95 of 243 that pins the top
of the scale. That is hi-vis PPE and retroreflective banding doing exactly
what it was designed to do, and poles (grade stakes, p95 123) show the same
signature for the same reason. Note the medians are far below the means:
these distributions are heavily right-skewed, because a return is only bright
when it lands on the retroreflective band rather than the fabric.

## What sitegen gets right, and what it gets wrong

**Right: the 1/r² falloff, for people.** This is the load-bearing claim and it
survives almost exactly. Real person counts fall 298 → 8 between the 5–10 m and
40–50 m bins, a factor of **37.3** across a 6× range increase; pure 1/r²
predicts **36×**. And the `sitegen / real` column for people is 0.13, 0.16,
0.19, 0.15 across four consecutive bins — a **flat ratio**, meaning the curve
has the right shape and only the wrong scale. sitegen is uniformly ~6–7× too
sparse on people, not wrongly shaped.

**Right: the ego machine is a real problem, and the README under-sells it.**
"About 12%" was conservative; a real excavator-mounted rig spends **28.0%** of
every sweep on itself. Masking it from proprioception is even more valuable
than sitegen claims.

**Right: the surround camera rig.** Four cameras at the house corners is what
ALICE actually does, with 110° lenses.

**Wrong: absolute density, by 29×.** 9,607 returns against 279,555. Nothing
about a 14,400-ray sensor resembles a 320-beam four-scanner rig, and this
propagates into every point-count claim. "A worker at 14 m gets about 20
points" is the README's headline number; the real figure is **141**.

**Wrong: the range distribution.** 89% of a real sweep is inside 10 m; sitegen
puts its median return at 14.5 m. sitegen's narrow −25°…+5° elevation band
aims most beams near the horizon, where they travel until they hit something
far away or nothing at all. A real rig with wide vertical coverage, mounted
low on a machine standing in a pit, dumps most of its beams into the ground
within a few metres.

**Wrong: vehicle falloff at range, badly.** Real vehicle counts collapse
1,737 → 4 between the 10–15 m and 40–50 m bins — a factor of **434** where
1/r² predicts **13**, leaving a factor of ~33 unexplained by geometry.
sitegen holds 70 points on a truck at 30–40 m where reality gives 9, and its
`--difficulty` dropout term (`0.35 × (r/60)²`, just 12% at 35 m) is nowhere
near steep enough to close that. Part of this is honest scene difference —
real distant machines sit behind spoil piles and other machines, and any
instance labelled at all at 50 m is by construction one of the barely-visible
ones, while sitegen's single truck is unoccluded on flat ground. But that *is*
the finding: **sitegen has no occlusion, and beyond ~15 m occlusion dominates
1/r².** The real IQRs say it outright — 17–235 points at 20–25 m is not noise
around a mean, it is two populations, one in clear view and one mostly hidden.

**Wrong: intensity carries no information.** sitegen computes intensity as
`1 − r/max_range` — a pure function of range, identical for a worker's vest
and the dirt behind them. In reality a person returns 7.5× the mean remission
of soil and 3× that of the machine beside them. sitegen's README observes that
"geometry finds the worker and cannot name the machine"; on real data,
*intensity alone* very nearly names the worker, and the synthetic data cannot
express that at all.

## Recommended parameter changes

Tested by re-running sitegen's own `sweep()` with modified `Lidar` settings on
seed 1 and re-measuring, so these are measured rather than argued. Note that
`beams`, `elevation_*` and `max_range` are not currently reachable from the
CLI — only `--azimuth-steps` is — so this needs flags as well as defaults.

| config | pts/sweep | ego% | median range | person @10–15 m |
| --- | ---: | ---: | ---: | ---: |
| **real GOOSE-Ex** | **279,555** | **28.0%** | **4.5 m** | **141** |
| current: 32 beams, 450 az, −25…+5°, 60 m | 9,572 | 13.4% | 14.5 m | 23 |
| A: 64 beams, 1440 az, −25…+5°, 60 m | 61,323 | 13.3% | 14.7 m | 133 |
| B: 64 beams, 1440 az, −45…+15°, 100 m | 64,431 | **29.8%** | **8.9 m** | 71 |
| C: 64 beams, 2048 az, −45…+15°, 100 m | 91,630 | **29.8%** | **8.9 m** | 96 |

1. **`beams: 32 → 64`, `azimuth_steps: 450 → 1440`.** A ~6.4× ray budget.
   This alone takes a worker at 10–15 m from 23 points to 133 against a real
   141 — the single change that most improves fidelity. It costs 6.4× the
   generation time and file size (a 60 s scene goes from 86 MB to roughly
   550 MB), so it belongs behind a flag with a documented cheap default rather
   than silently imposed.

2. **`elevation_min_deg: −25 → −45`, `elevation_max_deg: +5 → +15`.** Two
   independent statistics move the right way from this one change: median
   return range falls 14.7 m → 8.9 m (real 4.5 m) and the ego share rises
   13.3% → 29.8%, landing within two points of the real 28.0%. It costs person density, which is
   why it wants to be paired with (1) — config C is the best joint match
   found.

3. **`max_range: 60 → 100 m`.** Real returns reach 104.2 m. Nearly free and
   nearly cosmetic: 99.99% of real returns are inside 60 m, so this only
   affects whether far-field trucks exist at all.

4. **Make dropout much steeper at range, or add occlusion.** The real
   10–15 m → 40–50 m vehicle collapse is 434×; 1/r² accounts for 13× and the
   current dropout term for essentially none of the rest. Cranking
   `dropout_at_max_range` is the cheap approximation; the honest fix is
   terrain and inter-object occlusion, since that is what actually produces
   the effect and it is also what would make the far bins *bimodal* the way
   the real IQRs are (2–17 points at 30–40 m spans "mostly hidden" to "clear
   view").

5. **Give intensity a per-class albedo.** Replace `1 − r/max_range` with a
   range term times a class reflectivity, seeded from the measured medians:
   person ≈ 28 (p95 243, so sample it right-skewed rather than constant —
   the bright returns are the retroreflective bands), tree_trunk ≈ 18,
   heavy_machinery/truck ≈ 15, low_grass ≈ 12, pole ≈ 10, asphalt ≈ 10,
   soil ≈ 7, gravel ≈ 7, water ≈ 5, on the same 0–255 scale.
   This is the cheapest high-value change on the list: it costs one lookup per
   point and it creates the retroreflective-PPE signal that a real
   safety-critical pipeline would lean on, which sitegen currently makes
   impossible to exercise.

6. **Correct the README's "about 12%" to ~28%**, and quote a worker at 14 m as
   ~140 points rather than ~20, once (1) lands.

### What this does not license

sitegen's scene composition should *not* be tuned to GOOSE-Ex class shares.
Real `heavy_machinery` is 0.61% of returns and `person` 0.09%, against
sitegen's 6.3% truck and 0.29% worker — but that is a difference in what is
parked near the machine, not in how the sensor behaves, and sitegen's tighter
scene is a deliberate choice. Only the sensor model is being calibrated here.

## Caveats

* Ranges are measured from the machine base, not a scanner (see above); under
  2% at the distances compared.
* GOOSE-Ex's far-range instance bins are survivorship-biased — an instance
  needs at least one labelled return to appear at all — so real counts beyond
  ~40 m are a floor, not a mean.
* The merged cloud cannot yield a per-scanner elevation pattern or rotation
  rate. Those need the raw bags from <https://goosedb.uni-koblenz.de/>.
* `pole` is an imperfect stand-in for `grade_stake`, and GOOSE has no
  excavator class distinct from other heavy machinery.
* The sitegen side is one 60 s scene at one seed — 120 truth clouds, and only
  4 vehicle observations in some far bins. The real side is 2,164 frames across
  seven scenarios. Treat sitegen's far-range medians as indicative.
