"""Where the soil goes: one pass over the clock, and the timeline it leaves.

The scene is otherwise a pure function of `t` -- ask it for a pose at 41.3 s
and it computes one, with no memory of 41.2 s. Terrain cannot be written that
way, because a hole is history. So the whole run is simulated once, up front,
at `SIM_HZ`, and what the rest of the program reads afterwards is an indexable
timeline: which committed ground snapshot is current at frame *i*, which load
mesh is in the hauler, and what every volume was. Two callers that both ask for
`t = 41.3` still get the same answer for the same reason they always did.

The material balance is the point of the exercise. At every instant

    cut == bucket + bed + hauled + (stockpiles - stockpiles at t0)

and separately the ground's own volume moves by exactly `deposited - cut`.
Both are checked in `tests/test_terrain.py` to 1e-9 m3, which is round-off on a
40,401-node grid rather than a modelling tolerance, because nothing here
creates or destroys material: a cut reports what it removed, a deposit places
exactly what it was given, and the repose relaxation only ever moves soil
between neighbouring nodes in symmetric pairs.

The cycle this reads is the one that was already there, unchanged. The bucket
is in the ground from t = 12.1 s to 15.9 s of every cycle -- the tail of the
swing back, the settle, and the first second of the next dig, which is when the
scripted trajectory actually drags the cutting edge through soil -- and out of
it the rest of the time. Nothing decides that but the geometry: the bucket cuts
where its sole is below the surface and does not where it is not, which is also
why the walk between dig stations digs no trench, with no rule saying so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .actors import excavator_parts
from .geometry import Array
from .heightfield import (
    BedLoad,
    GroundSurface,
    Heightfield,
    bed_mesh,
    ground_field,
)
from .meshes import register

if TYPE_CHECKING:  # only for the annotation; scene.py imports this module
    from .scene import Scene

#: Simulation step. Twice the default sensor rate, so the bucket never moves
#: more than about 0.15 m between bites and the swept cut is continuous, and
#: independent of `--rate` so that the terrain a scene contains does not change
#: when the sensor is asked to spin faster.
SIM_HZ = 20.0

#: How often a snapshot of the ground is taken. This is the bounded rate the
#: `/terrain/heightmap` topic and the ray BVH both run at: a rebuild over
#: 80,000 triangles costs 39 ms, which is affordable once a second and is the
#: whole scene's budget twenty times a second. The visible cost is that a cut
#: appears up to one second after the bucket made it.
TERRAIN_COMMIT_HZ = 1.0

#: Heaped bucket capacity for a 20-tonne machine, in cubic metres.
BUCKET_CAPACITY_M3 = 1.2

#: What the hauler's body holds before the operator starts sending buckets to
#: the pile instead. Six passes, which is the cycle the scene has always
#: described. The modelled body is geometrically larger than that -- see
#: docs/TERRAIN.md -- and this is the number the scene's own timing implies.
BED_CAPACITY_M3 = 6.0 * BUCKET_CAPACITY_M3


def _bucket_sole() -> Array:
    """The bucket's contact footprint: the underside of its own bounding box.

    Taken from the mesh rather than written down, so it stays the bucket if the
    bucket is ever rebuilt -- 1.28 m from the back of the scoop to the tips of
    the teeth, 1.0 m across, at the depth of the teeth. Everything the posed
    box's sole passes below has been displaced into the bucket, which is what
    "a bucket-shaped bite" means at this resolution: the scoop's curvature is
    two centimetres of shape on a 25 cm grid.
    """
    from .meshes import bounds

    lo, hi = bounds("excavator.bucket")
    return np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
        ]
    )


BUCKET_SOLE = _bucket_sole()

#: Fraction of the cycle spent tipping the bucket out.
DUMP_PHASE = (0.47, 0.60)

#: Where a bucket lands on a stockpile, relative to its centre. Drawn once per
#: dump from the scene's seed, so `--seed` moves the shape of a growing pile
#: without touching the volumes: the relaxation itself has no random component
#: at all, and nor does anything else here.
STOCKPILE_SCATTER_M = 0.35


@dataclass(frozen=True)
class Volumes:
    """Every cubic metre on the site, accounted for, at one instant.

    `net` is the one an outside observer could measure: the change in the
    ground's own volume since t0, which is what differencing two surveys gives
    you. The rest is what a survey cannot see -- what is in the bucket, what is
    in the body, and what has already been driven off the site.
    """

    t: float
    cut: float
    bucket: float
    bed: float
    hauled: float
    stockpiles: tuple[float, ...]
    net: float

    def channels(self) -> list[tuple[str, float]]:
        """Ordered name/value pairs, as the topic and the CSV both publish."""
        return [
            ("cut", self.cut),
            ("bucket", self.bucket),
            ("bed", self.bed),
            ("hauled", self.hauled),
            *((f"stockpile_{i}", v) for i, v in enumerate(self.stockpiles)),
            ("net", self.net),
        ]


@dataclass
class Timeline:
    """The simulated run, indexed by frame at `SIM_HZ`."""

    surfaces: list[GroundSurface]
    surface_at: NDArray[np.int32]
    load_meshes: list[str | None]
    """Registered mesh names for the hauler's load; index 0 is an empty body."""
    load_bounds: list[tuple[Array, Array] | None]
    load_at: NDArray[np.int32]
    volumes: list[Volumes]

    def frame(self, t: float) -> int:
        return int(np.clip(round(t * SIM_HZ), 0, len(self.volumes) - 1))

    def surface(self, t: float) -> GroundSurface:
        return self.surfaces[int(self.surface_at[self.frame(t)])]

    def load(self, t: float) -> tuple[str, tuple[Array, Array]] | None:
        index = int(self.load_at[self.frame(t)])
        name = self.load_meshes[index]
        extent = self.load_bounds[index]
        if name is None or extent is None:
            return None
        return name, extent

    def volumes_at(self, t: float) -> Volumes:
        return self.volumes[self.frame(t)]


def simulate(scene: Scene, duration_s: float) -> Timeline:
    """Step the site forward and record what the sensors will be shown."""
    ground = ground_field(scene.terrain)
    start_volume = ground.volume()
    discs = _stockpile_discs(ground, scene)
    rng = np.random.default_rng(scene.seed + 2000)

    bed = BedLoad()
    bed_owner: str | None = None
    carrying = 0.0
    cut_total = 0.0
    hauled_total = 0.0
    dump_start_load = 0.0

    surfaces = [GroundSurface.commit(ground)]
    committed = ground.z.copy()
    load_meshes: list[str | None] = [None]
    load_bounds: list[tuple[Array, Array] | None] = [None]
    committed_load = 0.0

    frames = int(round(duration_s * SIM_HZ)) + 1
    commit_every = max(1, int(round(SIM_HZ / TERRAIN_COMMIT_HZ)))
    surface_at = np.zeros(frames, dtype=np.int32)
    load_at = np.zeros(frames, dtype=np.int32)
    volumes: list[Volumes] = []

    for i in range(frames):
        t = i / SIM_HZ

        # -- the bucket in the ground ---------------------------------------
        phase = (t % scene.cycle_s) / scene.cycle_s
        dumping = _dumping(scene, t)
        if not dumping and carrying < BUCKET_CAPACITY_M3:
            bucket = excavator_parts(scene.ego_at(t), "ego")[-1]
            corners = BUCKET_SOLE @ bucket.rotation.T + bucket.translation
            taken = ground.carve(corners, BUCKET_CAPACITY_M3 - carrying)
            carrying += taken
            cut_total += taken

        # -- which hauler is standing there ---------------------------------
        cycle = scene.truck_cycle(t)
        owner = cycle[2] if cycle is not None else None
        at_spot = cycle is not None and cycle[0] <= t <= cycle[1]
        if owner != bed_owner:
            hauled_total += bed.empty()
            bed_owner = owner

        # -- tipping out ----------------------------------------------------
        if dumping:
            if phase - DUMP_PHASE[0] < 0.5 / (SIM_HZ * scene.cycle_s):
                dump_start_load = carrying
            span = (phase - DUMP_PHASE[0]) / (DUMP_PHASE[1] - DUMP_PHASE[0])
            last = not _dumping(scene, t + 1.0 / SIM_HZ)
            wanted = 0.0 if last else dump_start_load * (1.0 - min(span, 1.0))
            tipped = max(carrying - wanted, 0.0)
            carrying -= tipped
            spilled = bed.add(tipped, BED_CAPACITY_M3) if at_spot else tipped
            if spilled > 0.0:
                x, y = _nearest_stockpile(scene, t)
                ground.deposit(
                    x + float(rng.normal(0.0, STOCKPILE_SCATTER_M)),
                    y + float(rng.normal(0.0, STOCKPILE_SCATTER_M)),
                    spilled,
                )
        else:
            dump_start_load = carrying

        # -- snapshots, at a bounded rate -----------------------------------
        if i and i % commit_every == 0:
            if not np.array_equal(committed, ground.z):
                surfaces.append(GroundSurface.commit(ground))
                committed = ground.z.copy()
            if abs(bed.volume() - committed_load) > 1e-12:
                committed_load = bed.volume()
                load_meshes.append(_register_load(bed, len(load_meshes)))
                load_bounds.append(
                    None
                    if load_meshes[-1] is None
                    else _bounds_of(load_meshes[-1])
                )
        surface_at[i] = len(surfaces) - 1
        load_at[i] = len(load_meshes) - 1

        volumes.append(
            Volumes(
                t=t,
                cut=cut_total,
                bucket=carrying,
                bed=bed.volume(),
                hauled=hauled_total,
                stockpiles=tuple(
                    float(ground.z[mask].sum()) * ground.pitch**2 for mask in discs
                ),
                net=ground.volume() - start_volume,
            )
        )

    return Timeline(
        surfaces=surfaces,
        surface_at=surface_at,
        load_meshes=load_meshes,
        load_bounds=load_bounds,
        load_at=load_at,
        volumes=volumes,
    )


def _dumping(scene: Scene, t: float) -> bool:
    phase = (t % scene.cycle_s) / scene.cycle_s
    return DUMP_PHASE[0] <= phase < DUMP_PHASE[1]


def _register_load(bed: BedLoad, index: int) -> str | None:
    """Bake the current load surface into a named, cacheable mesh.

    A name per committed load rather than a mutable mesh, because the Cycles
    bridge addresses geometry by name and writes one PLY per name. The load
    only changes while the bucket is tipping, so a 60 s scene registers a
    handful of these, not six hundred.
    """
    if bed.volume() <= 1e-9:
        return None
    return register(f"haul_truck.load.{index:04d}", bed_mesh(bed.surface))


def _bounds_of(name: str) -> tuple[Array, Array]:
    from .meshes import bounds

    lo, hi = bounds(name)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _stockpile_discs(ground: Heightfield, scene: Scene) -> list[NDArray[np.bool_]]:
    """One boolean mask per stockpile, generous enough to hold a grown pile.

    The oracle's per-pile volume is the ground inside this disc, which is what
    a survey would report for "the pile": a fixed footprint, integrated. 1.6
    times the starting radius leaves room for several hundred cubic metres of
    growth before the disc starts clipping the toe.
    """
    ax, ay = ground.axes()
    gx, gy = np.meshgrid(ax, ay, indexing="ij")
    return [
        np.hypot(gx - pile.x, gy - pile.y) <= 1.6 * pile.radius
        for pile in scene.terrain.stockpiles
    ]


def _nearest_stockpile(scene: Scene, t: float) -> tuple[float, float]:
    """The pile the operator would swing to: the closest one to the machine."""
    bx, by = scene.base_at(t)
    pile = min(
        scene.terrain.stockpiles, key=lambda p: float(np.hypot(p.x - bx, p.y - by))
    )
    return pile.x, pile.y
