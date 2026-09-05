"""Reading GOOSE-Ex, the closest thing to ground truth for sitegen's sensor.

sitegen's LiDAR model asserts its numbers. Points fall off as 1/r^2, dropout
rises quadratically, the ego machine is "about 12%" of the sweep -- all of it
plausible, none of it measured. GOOSE-Ex is a real Ouster rig on a real
Liebherr R924 tracked excavator working real landfill, quarry and construction
sites, with pointwise semantic *and instance* labels. That last part is what
makes the comparison possible: with instance ids you can ask the only question
that decides whether an auto-labeler works -- **how many points land on a
person at 20 m** -- and get an answer from a machine that exists.

The format is the SemanticKITTI convention, which GOOSE inherits:

  * ``*_pcl.bin``    float32, ``(N, 4)``, ``x y z remission``
  * ``*_goose.label`` uint32, ``(N,)``, ``sem = label & 0xFFFF``,
    ``inst = label >> 16``

Two things about the geometry are worth knowing before trusting any range
computed here. The cloud is *merged* from four Ouster OS0 units (three OS0-64
on the house, one OS0-128 on the boom), so it is not one sensor's sweep and
has no single origin ray pattern. And the merged frame sits at the machine
base, roughly ground level, not at a sensor -- so a range measured from the
origin is range-from-the-machine, which differs from range-from-a-sensor by
the couple of metres of mounting height. At the distances that matter here
that is under 2%, and it is the honest quantity to compare against anyway.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Array = np.ndarray

# The GOOSE ontology, mapped onto the four things sitegen simulates. GOOSE has
# no `excavator` class -- a tracked excavator is `heavy_machinery`, which also
# covers wheel loaders and dozers. `grade_stake` has no exact match either;
# `pole` is the nearest thing with instances. Everything a sitegen scene calls
# terrain lives across five GOOSE ground classes, because a real site is not
# one material.
SITEGEN_EQUIVALENT: dict[str, tuple[str, ...]] = {
    "excavator": ("heavy_machinery",),
    "haul_truck": ("truck", "trailer"),
    "worker": ("person",),
    "grade_stake": ("pole",),
    "terrain": ("soil", "gravel", "asphalt", "cobble", "low_grass"),
    "ego": ("ego_vehicle",),
}

# The two groups the points-on-target curve is drawn for. Vehicles are pooled
# because sitegen's haul truck and excavator are the same problem to a
# clusterer: a large rigid body, tens of square metres of facing surface.
PERSON_CLASSES = ("person",)
VEHICLE_CLASSES = ("heavy_machinery", "truck", "car", "bus", "trailer", "caravan")


@dataclass(frozen=True)
class Ontology:
    """The `goose_label_mapping.csv` that ships beside every split."""

    name_of: dict[int, str]
    has_instance: dict[int, bool]

    @classmethod
    def load(cls, path: Path) -> Ontology:
        name_of: dict[int, str] = {}
        has_instance: dict[int, bool] = {}
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                key = int(row["label_key"])
                name_of[key] = row["class_name"]
                has_instance[key] = row["has_instance"] == "1"
        return cls(name_of=name_of, has_instance=has_instance)

    def keys_for(self, names: tuple[str, ...]) -> set[int]:
        wanted = set(names)
        return {k for k, n in self.name_of.items() if n in wanted}


@dataclass
class Frame:
    """One labelled merged sweep: geometry, remission, and per-point truth."""

    path: Path
    xyz: Array
    """(N, 3) float32, metres, in the platform's merged-cloud frame."""
    remission: Array
    """(N,) float32. Ouster reports this as a 16-bit signal count, but the
    values here run 0-255 with a long tail, so it behaves like an 8-bit
    intensity. It is *not* normalised and *not* comparable across sensors."""
    semantic: Array
    """(N,) uint16 class key, indexes into `Ontology.name_of`."""
    instance: Array
    """(N,) uint16 instance id, unique per frame, 0 where the class carries
    no instances (all the terrain and vegetation classes)."""

    @property
    def platform(self) -> str:
        """`alice` (the excavator) or `spot` (the quadruped)."""
        return self.path.name.split("_", 1)[0]

    @property
    def range_m(self) -> Array:
        return np.linalg.norm(self.xyz, axis=1)

    def __len__(self) -> int:
        return len(self.xyz)


def load_frame(cloud: Path) -> Frame:
    """Read one `*_pcl.bin` and the `*_goose.label` that belongs to it.

    The two live in parallel trees (`lidar/<split>/...` and
    `labels/<split>/...`) under matching names, so the label path is derived
    rather than searched for.
    """
    label = Path(
        str(cloud).replace("/lidar/", "/labels/").replace("_pcl.bin", "_goose.label")
    )
    scan = np.fromfile(cloud, dtype=np.float32).reshape(-1, 4)
    packed = np.fromfile(label, dtype=np.uint32)
    if len(packed) != len(scan):
        raise ValueError(f"{cloud.name}: {len(scan)} points but {len(packed)} labels")
    return Frame(
        path=cloud,
        xyz=scan[:, :3],
        remission=scan[:, 3],
        # The low half is the class, the high half the instance. Anything that
        # reads only the uint32 gets nonsense for every labelled object.
        semantic=(packed & 0xFFFF).astype(np.uint16),
        instance=(packed >> 16).astype(np.uint16),
    )


def find_frames(root: Path, platform: str = "alice") -> list[Path]:
    """Every labelled cloud for one platform, in sequence order.

    `root` is an extracted split directory, e.g. `.../gooseEx_3d_val`.
    """
    return sorted(root.glob(f"lidar/*/{platform}_*/*_pcl.bin"))


def iter_frames(paths: list[Path]) -> Iterator[Frame]:
    for p in paths:
        yield load_frame(p)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """One instance, seen in one frame: how far away, and how many returns.

    This is the shared currency between the real data and sitegen. Both sides
    reduce to a list of these, and the comparison is then just two histograms
    on the same axes.
    """

    group: str
    """`person` or `vehicle`."""
    range_m: float
    """Median range of the instance's own points -- robust to a long vehicle
    presenting 12 m of flank, where a centroid would be pulled around by which
    end happens to be visible."""
    points: int


def class_histogram(frames: Iterator[Frame], onto: Ontology) -> dict[str, int]:
    """Total returns per class name, summed over frames."""
    total: dict[str, int] = {}
    for frame in frames:
        keys, counts = np.unique(frame.semantic, return_counts=True)
        for key, count in zip(keys, counts):
            name = onto.name_of.get(int(key), f"unknown_{key}")
            total[name] = total.get(name, 0) + int(count)
    return total


def remission_by_class(
    frame: Frame, onto: Ontology
) -> dict[str, tuple[int, float, float, float]]:
    """Per class in one frame: (n, mean, median, p95) of remission."""
    out: dict[str, tuple[int, float, float, float]] = {}
    for key in np.unique(frame.semantic):
        mask = frame.semantic == key
        values = frame.remission[mask]
        out[onto.name_of.get(int(key), f"unknown_{key}")] = (
            int(mask.sum()),
            float(values.mean()),
            float(np.median(values)),
            float(np.percentile(values, 95)),
        )
    return out


def observations(frame: Frame, onto: Ontology) -> list[Observation]:
    """Split one frame into per-instance points-on-target observations.

    Instance id 0 is the catch-all for "this class carries instances but this
    point was not assigned to one", so it is dropped rather than treated as a
    single enormous object.
    """
    groups = {
        "person": onto.keys_for(PERSON_CLASSES),
        "vehicle": onto.keys_for(VEHICLE_CLASSES),
    }
    ranges = frame.range_m
    out: list[Observation] = []
    for group, keys in groups.items():
        in_group = np.isin(frame.semantic, list(keys))
        if not in_group.any():
            continue
        # An instance id is only unique within a class, so key on the pair.
        tagged = frame.semantic.astype(np.int64) << 20 | frame.instance
        for tag in np.unique(tagged[in_group & (frame.instance > 0)]):
            mask = tagged == tag
            out.append(
                Observation(
                    group=group,
                    range_m=float(np.median(ranges[mask])),
                    points=int(mask.sum()),
                )
            )
    return out


# Bin edges shared by both sides of the comparison. They are narrow where the
# curve is steep and interesting (10-30 m, the range at which a clusterer
# still has a chance) and wide out where everything is a handful of points.
RANGE_BINS: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 80.0)


@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    """Number of instance observations that fell in this bin."""
    median: float
    p25: float
    p75: float


def bin_observations(obs: list[Observation], group: str) -> list[Bin]:
    """Median points-on-target per range bin, for one group."""
    picked = np.array(
        [(o.range_m, o.points) for o in obs if o.group == group], dtype=np.float64
    )
    out: list[Bin] = []
    if not len(picked):
        return out
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        mask = (picked[:, 0] >= lo) & (picked[:, 0] < hi)
        if not mask.any():
            continue
        counts = picked[mask, 1]
        out.append(
            Bin(
                lo=lo,
                hi=hi,
                n=int(mask.sum()),
                median=float(np.median(counts)),
                p25=float(np.percentile(counts, 25)),
                p75=float(np.percentile(counts, 75)),
            )
        )
    return out
