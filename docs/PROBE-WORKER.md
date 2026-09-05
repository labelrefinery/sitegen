# The worker probe

sitegen's own renderer draws every actor as a flat-shaded oriented box, and
when Grounding DINO was pointed at ten of those images it named the haul truck
nine times out of ten and **never once found a worker** — the one class where a
missed label is a safety failure rather than a metric.

That result has two possible causes and the README could not tell them apart:

- the *scene* is hard — a 40 x 110 pixel figure at 9–16 m in flat light, and no
  detector would find it either; or
- the *renderer* is unconvincing — an orange cuboid is not a person, and the
  detector is right to refuse.

This is the experiment that separates them. Ten views, the same camera poses,
the same worker positions, one thing changed.

## What was rendered

`sitegen cameras` exports the ten views out of the recording rather than
recomputing them, so the geometry under test is the geometry that was written:
intrinsics from `/camera/<name>/calibration`, extrinsics from the `/tf` entry
for `camera_<name>`, and actor poses from `/ground_truth/actors`. The held-out
per-pixel instance masks pick the views — every distinct one with a worker over
60 pixels, then the clearest views of the truck as a control — and give each
actor a pixel bounding box, which is what later decides whether a detection
landed on the worker or on something else.

`tools/probe/render.py` rebuilds each of those ten views in Cycles:

| | baseline (sitegen) | probe (Cycles) |
| --- | --- | --- |
| camera | pinhole raycast, 960 x 540 | identical K and pose, sensor/lens derived from fx |
| light | one directional term, sky gradient | `construction_yard` HDRI, 2K |
| ground | flat albedo on an analytic plane | same plane, gravel PBR (albedo, roughness, normal) |
| stockpiles | analytic cones | the same two cones, matte dirt |
| haul truck | flat-albedo cuboids | the same cuboids, PBR grey — **unchanged on purpose** |
| worker | flat orange cuboid, 0.6 x 0.45 x 1.75 m | **rigged textured human in hi-vis and hard hat** |

The truck staying a pair of boxes is the control. If its score survived the
change of renderer then the experiment moved one variable; if it had jumped
too, the comparison would have been measuring "everything at once".

The human is scaled to the 1.75 m box it replaces, stood on the same ground
height, and turned to the same yaw. The spotter gets the asset's `Idle` action
and the worker crossing the swing radius gets `Walk`, because the scene says
one holds station and the other is walking.

### Asset and licence

Full detail in `tools/probe/CREDITS`. In short, all CC0:

- **human** — "Worker" by Quaternius, CC0, via poly.pizza. Rigged and skinned,
  hard hat and hi-vis vest as separate materials. It is *stylised low-poly and
  flat-coloured, not photoreal and not image-textured* — which matters for how
  far the result generalises, and is discussed under Verdict.
- **lighting** — "Construction Yard" HDRI by Sergej Majboroda, Poly Haven, CC0.
- **ground** — "Gravel" 2K by Dimitrios Savva, Poly Haven, CC0.

## Reproducing it

```sh
uv run sitegen generate --out site.mcap --seed 1 --duration 60 --camera-hz 2
uv run sitegen cameras site.mcap --out probe/views

# Blender 5.2.1, as the PyPI module rather than the app -- see Deviations.
uv run --python 3.13 --with bpy==5.2.1 python tools/probe/render.py \
    --views probe/views --assets probe/assets --out probe/renders
uv run --python 3.13 --with bpy==5.2.1 python tools/probe/render.py \
    --views probe/views --assets probe/assets --out probe/renders_box \
    --worker-as-box

# grounding-dino-tiny, the same weights and prompt the README quotes
for d in views renders renders_box; do
    uv run --with torch --with transformers --with pillow \
        python tools/probe/dino.py --dir probe/$d --out probe/scores_$d.json \
        --glob "$([ $d = views ] && echo '*.jpg' || echo '*.png')"
done

uv run python tools/probe/compare.py --views probe/views \
    --baseline probe/scores_views.json --ablation probe/scores_renders_box.json \
    --render probe/scores_renders.json --renders probe/renders \
    --figures docs/probe
```

Prompt, throughout: `excavator . haul truck . worker . person .`
Reported at the reference script's default `--box-threshold 0.35`; scores below
it are shown in parentheses, because "never detected" is a claim about a number
and 0.00 and 0.31 are different kinds of miss.

## The baseline, reproduced

Ten views out of `--seed 1`, sitegen's own renderer:

| view | detection | score |
| --- | --- | ---: |
| front_right t=9.5 | haul truck | 0.475 |
| front_left t=11.5 | haul truck | 0.497 |
| front_left t=21.5 | haul truck | 0.335 |
| front_left t=23.5 | excavator haul truck | 0.367 |
| left t=23.5 | excavator haul truck | 0.363 |
| front_right t=24.5 | haul truck | 0.458 |
| front_left t=27.0 | haul truck | 0.445 |
| front_right t=30.0 | excavator haul truck | 0.373 |
| front_right t=43.0 | haul truck | 0.323 |
| front_right t=45.0 | haul truck | 0.323 |
| **any worker, any view** | — | **none above 0.35** |

Truck scores 0.32–0.50 against 0.771 on a real photograph, which is the
README's finding to within the noise of a different ten frames. Two details the
original table did not have room for:

- the single worker score anywhere in the baseline is **0.254**, on the closest
  worker (9.3 m, 6797 px), below threshold.
- at t=43 and t=45 the detector *did* put a box on the worker's pixels — and
  called it `exca haul truck` (0.307). It is not that the figure is invisible;
  it is that a coloured cuboid reads as machinery.

## The comparison

Worker rows are per actor per view, so the eleven rows are eleven worker
sightings across the ten images. A score counts only if the detection box
overlaps that actor's ground-truth pixels (IoU ≥ 0.1) — otherwise a stray
`worker 0.25` on the horizon would score as a hit.

| view | worker px | range | baseline worker | box in Cycles | Cycles worker | label | baseline truck | Cycles truck |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| front_right t=9.5 worker_0 | 3497 | 12.8 m | -- | -- | 0.573 | worker person | 0.475 | 0.477 |
| front_left t=11.5 worker_0 | 4026 | 12.8 m | -- | -- | 0.568 | worker person | 0.497 | 0.443 |
| front_left t=21.5 worker_1 | 3940 | 13.8 m | -- | -- | 0.516 | worker person | -- | -- |
| front_left t=23.5 worker_1 | 6797 | 9.3 m | (0.254) | -- | 0.482 | worker person | -- | -- |
| left t=23.5 worker_1 | 6272 | 9.9 m | -- | -- | 0.536 | worker person | -- | -- |
| front_right t=24.5 worker_0 | 4471 | 12.8 m | -- | -- | 0.562 | worker person | 0.458 | 0.472 |
| front_left t=27.0 worker_0 | 3473 | 13.2 m | -- | -- | 0.582 | worker person | 0.445 | 0.459 |
| front_right t=30.0 worker_1 | 5763 | 8.1 m | -- | -- | 0.513 | worker person | -- | -- |
| front_right t=30.0 worker_0 | 705 | 16.2 m | -- | -- | 0.590 | worker person | -- | -- |
| front_right t=43.0 worker_1 | 3456 | 15.5 m | -- | -- | 0.533 | worker person | -- | -- |
| front_right t=45.0 worker_1 | 3456 | 15.5 m | -- | -- | 0.533 | worker person | -- | -- |

**11 of 11.** Mean 0.544, range 0.482–0.590, every one labelled `worker person`
— the detector matched both prompt phrases to the same box rather than picking
one, so the label is unambiguous in a way the baseline's `excavator haul truck`
was not.

The control held: truck 0.469 mean before, 0.463 after, unmoved to within a
hundredth. The truck is the same pair of boxes in both images; only its shading
changed, and the detector did not care.

The `box in Cycles` column is the ablation, and it is the load-bearing one.
Same Cycles, same HDRI, same gravel, same everything — with sitegen's orange
cuboid still standing in for the worker. **Nothing.** Not one detection landed
on a worker box in any of the ten. The `worker 0.25`-ish scores that do appear
in those images sit at y ≈ 256–283, which is the HDRI backdrop, not the site.

So it is not the lighting, not the tone mapping, not the ground texture, and
not the context. It is the shape of the thing.

Side-by-side images, baseline left and Cycles right, are in `docs/probe/`.

## Verdict

**The worker moved off 0.000 and it was not close.** Eleven of eleven sightings
detected where the baseline had zero, mean 0.544 against a baseline best of
0.254, and every one correctly named. That lands the synthetic worker roughly
where a real photograph puts a real machine (`excavator` 0.858, `haul truck`
0.771 on the control photo) — not equal to it, but the same order of confidence
rather than a different regime. The ablation says the credit belongs to the
human mesh specifically and not to the renderer: putting the identical camera,
sun, sky and ground behind an orange cuboid still detects nothing at all. What
an open-vocabulary detector wants from a person is a person-shaped silhouette,
and it will take one that is low-poly and flat-coloured. What it will not take
is a box, however well lit.

The practical consequence for sitegen is that the README's asymmetry —
"geometry finds the worker and cannot name the machine, the detector names the
machine and cannot find the worker" — is a property of the box renderer, not of
the scene or of the detector. Replacing the worker cuboid with a mesh is enough
to make the camera half of a naming pipeline cover the safety-critical class,
which is the half that mattered.

## What this does not settle

- **The asset is stylised.** The human is low-poly and flat-coloured, so this
  is a *lower bound*: a scanned or photoreal human should do at least as well.
  It also means the result says nothing about how far a *student trained* on
  these renders would transfer to real imagery — a detector recognising a
  cartoon is a much weaker claim than a detector's features being useful
  training signal.
- **The HDRI backdrop is visible, and it contains real people and vehicles.**
  They generate their own sub-threshold detections above the horizon
  (`excavator haul` 0.48 on a building at t=43 is the worst of them). The IoU
  gate keeps them out of the table, but a pipeline consuming raw detections off
  these images would have to reject them, and the baseline's empty sky never
  posed that problem. An HDRI with a clear horizon would isolate the variable
  better.
- **Ten frames, one seed, one detector, one checkpoint.** Everything here is
  `grounding-dino-tiny` at 0.35 on `--seed 1`.
- **This is not a renderer for sitegen.** Nothing in `tools/probe/` is wired
  into `sitegen generate`; the probe answers a question about what a mesh would
  buy, and the answer is an argument for the "Mesh geometry" item under *Not
  yet*, not an implementation of it.

## Deviations from the plan

The probe was specified to drive `blender -b -P script.py`. Blender 5.2.1 was
installed with `brew install --cask blender` as intended, but a cask-installed
app carries macOS's `com.apple.quarantine` flag, and every executable inside a
quarantined bundle blocks at launch waiting for a first-run consent dialog that
a headless session cannot show — the process sits at 0% CPU indefinitely, and
the bundled Python hangs the same way. Clearing the attribute was not
permitted here, so the renders were made with the official `bpy` PyPI module
instead: the same Blender 5.2.1 build and the same Cycles, imported as a Python
module rather than launched as an app. `render.py` is a plain script either
way; run under `blender -b -P` it would need only its arguments passed after
`--`.
