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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .geometry import Array, IndexArray

if TYPE_CHECKING:  # only for the annotation; raycast imports this module back
    from .raycast import Raycaster


@dataclass
class Lidar:
    """A 32-beam spinning sensor, roughly a mid-range mechanical unit."""

    beams: int = 32
    azimuth_steps: int = 720
    elevation_min_deg: float = -25.0
    elevation_max_deg: float = 5.0
    max_range: float = 60.0
    range_noise_per_m: float = 0.0015
    range_noise_floor: float = 0.02
    dropout_at_max_range: float = 0.35
    _dirs: Array | None = field(default=None, repr=False)

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

    keep = rng.random(t.shape) > lidar.dropout_at_max_range * (t / lidar.max_range) ** 2

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
