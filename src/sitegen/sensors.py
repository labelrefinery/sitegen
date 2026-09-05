"""A spinning LiDAR, and the ways it lies to you.

Perfect returns make a pipeline look good for the wrong reasons. The
degradations here are the ones that actually decide whether an auto-labeler
works:

  - **Range-dependent density.** Points on a target fall off as 1/r^2. A truck
    at 45 m is a dozen returns, which is where naive clustering quietly stops
    finding it -- and where a smoother that carries the track forward from when
    it was close earns its keep.
  - **Range-dependent noise.** Position error grows with distance, so a box
    fitted far away is both sparse and wrong.
  - **Dust.** Scripted events attenuate returns over a region. Dust is the
    construction-specific failure: it is not weather, it is caused by the
    machines themselves, so it correlates with exactly the moments something
    interesting is happening.
  - **Self-returns.** The sensor sits on the ego's house, so the ego's own boom
    and bucket are in the cloud. Nothing removes them for you. Masking them
    from proprioception is the first thing an unlabeled pipeline gets to do
    with its free labels.

The numbers below are no longer asserted. Every default here was measured
against GOOSE-Ex -- 2,164 labelled sweeps from four Ousters on a Liebherr R924
working real sites -- and `docs/GOOSE-EX.md` records which statistic each one
bought. `Lidar.legacy()` is the sensor as it was before that, kept so the
measurements taken against it stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from .geometry import Array, IndexArray

if TYPE_CHECKING:  # only for the annotation; raycast imports this module back
    from .raycast import Raycaster


#: Raw Ouster remission by class, as (median, p95), measured on the 2,164
#: labelled ALICE sweeps of GOOSE-Ex. The pair is enough to fix a lognormal:
#: these distributions are strongly right-skewed, because a return is bright
#: only when it lands on a retroreflective band rather than on the fabric
#: beside it, and a constant per class would throw exactly that away. A person
#: is the brightest thing on a real site -- hi-vis PPE doing what it is for --
#: and grade stakes carry the same banding for the same reason.
REMISSION: dict[str, tuple[float, float]] = {
    "worker": (28.0, 243.0),
    "grade_stake": (10.0, 123.2),
    "haul_truck": (15.0, 51.0),
    "excavator": (15.0, 50.0),
    "ego": (9.0, 100.0),
    "terrain": (7.0, 20.0),
}

#: Ouster reports remission 0-255 with a long tail; sitegen's `intensity`
#: field stays in [0, 1], so this is the divisor between the two.
REMISSION_FULL_SCALE = 255.0

#: The p95 of a lognormal sits 1.645 sigma above its median.
_P95_SIGMAS = 1.6448536269514722


@dataclass
class Lidar:
    """A spinning sensor, with GOOSE-Ex's excavator rig for its geometry.

    The elevation band and range are the real machine's, not a datasheet's:
    a rig mounted low on a machine standing in a pit throws most of its beams
    into the ground within a few metres, which is why 89% of a real sweep
    lands inside 10 m. `beams` and `azimuth_steps` are *not* the real 320
    channels -- that is a ray budget the default deliberately does not spend;
    see `--density real`.
    """

    beams: int = 32
    azimuth_steps: int = 720
    elevation_min_deg: float = -45.0
    elevation_max_deg: float = 15.0
    max_range: float = 100.0
    range_noise_per_m: float = 0.0015
    range_noise_floor: float = 0.02
    dropout_far: float = 0.35
    """Probability a return at `dropout_range_m` is lost outright."""
    dropout_range_m: float = 60.0
    """The distance `dropout_far` is quoted at. Held fixed rather than tied to
    `max_range`, because whether a return survives 45 m of air is a fact about
    45 m: quoting it as a fraction of the catalogue range would have meant
    raising `max_range` 60 -> 100 m silently cut far-field dropout by two
    thirds, which is a degradation changing behind a range change's back."""
    per_class_intensity: bool = True
    """False restores `1 - r/max_range`, which is a pure function of range and
    says nothing about what the beam hit."""
    _dirs: Array | None = field(default=None, repr=False)

    def legacy(self) -> "Lidar":
        """This sensor as it was before GOOSE-Ex was measured.

        Every number in `docs/PROBE-WORKER.md` and every pre-calibration table
        was taken through these settings, and `--actors boxes --sensor legacy`
        still reproduces those recordings byte for byte -- which it can only do
        if the intensity path does not draw from the shared generator either,
        which is what `per_class_intensity` switches off.
        """
        return replace(
            self,
            elevation_min_deg=-25.0,
            elevation_max_deg=5.0,
            max_range=60.0,
            per_class_intensity=False,
        )

    def directions(self) -> Array:
        """Unit ray directions in the sensor frame, (beams * azimuth, 3)."""
        if self._dirs is None:
            el = np.radians(
                np.linspace(self.elevation_min_deg, self.elevation_max_deg, self.beams)
            )
            az = np.linspace(0.0, 2.0 * np.pi, self.azimuth_steps, endpoint=False)
            e, a = np.meshgrid(el, az, indexing="ij")
            self._dirs = np.stack(
                [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1
            ).reshape(-1, 3)
        return self._dirs


def intensity(
    lidar: Lidar,
    points: Array,
    source: IndexArray,
    classes: list[str],
    rng: np.random.Generator,
) -> Array:
    """Per-return intensity in [0, 1]: a class albedo times a range term.

    `classes` is the per-source-index key into `REMISSION`, and -1 is terrain.
    The albedo is drawn per point rather than looked up, from a lognormal
    pinned to the measured median and p95 -- so a worker is usually about as
    bright as the dirt behind them and occasionally four times brighter, which
    is what the real distribution does and what makes intensity a class signal
    instead of a second copy of range.
    """
    ranges = np.linalg.norm(points, axis=1)
    fade = np.clip(1.0 - ranges / lidar.max_range, 0.0, 1.0)
    if not lidar.per_class_intensity:
        return fade

    keys = ["terrain", *classes]
    median = np.array([REMISSION[k][0] for k in keys])
    sigma = np.array(
        [np.log(REMISSION[k][1] / REMISSION[k][0]) / _P95_SIGMAS for k in keys]
    )
    index = source + 1  # -1 (terrain) becomes 0, which is `keys[0]`
    albedo = median[index] * np.exp(sigma[index] * rng.standard_normal(len(index)))
    return np.clip(albedo / REMISSION_FULL_SCALE * fade, 0.0, 1.0)


@dataclass
class DustEvent:
    """A cloud of airborne fines that eats returns passing through it."""

    start_s: float
    end_s: float
    x: float
    y: float
    radius: float
    severity: float = 0.7

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s


def sweep(
    lidar: Lidar,
    sensor_r: Array,
    sensor_t: Array,
    caster: "Raycaster",
    rng: np.random.Generator,
    dust: list[DustEvent],
    t_seconds: float,
) -> tuple[Array, IndexArray]:
    """One full rotation. Returns (points_in_sensor_frame, source_index).

    `source_index` is the index into the actor list each point came from, or -1
    for terrain. It is written only to the ground-truth topic -- it is
    per-point semantic truth, and handing it to a labeler would defeat the
    exercise.

    Nothing below the intersection changed when actors became meshes. Which
    triangle a ray hit is a different question from how a real sensor mangles
    the answer, and only the first one moved.
    """
    world_dirs = lidar.directions() @ sensor_r.T
    hits = caster.intersect(sensor_t, world_dirs)
    t, source = hits.t, hits.source

    valid = np.isfinite(t) & (t < lidar.max_range)
    if not np.any(valid):
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)

    t = t[valid]
    dirs = world_dirs[valid]
    source = source[valid]

    # Range noise grows with distance; a far box is sparse *and* smeared.
    sigma = lidar.range_noise_floor + lidar.range_noise_per_m * t
    t = t + rng.normal(0.0, sigma)

    keep = rng.random(t.shape) > lidar.dropout_far * (t / lidar.dropout_range_m) ** 2

    points_world = sensor_t + dirs * t[:, None]
    for event in dust:
        if not event.active(t_seconds):
            continue
        d = np.hypot(points_world[:, 0] - event.x, points_world[:, 1] - event.y)
        inside = d < event.radius
        keep &= ~(inside & (rng.random(t.shape) < event.severity))

    points_world = points_world[keep]
    source = source[keep]
    points_sensor = (points_world - sensor_t) @ sensor_r
    return points_sensor, source
