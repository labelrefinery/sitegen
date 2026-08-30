"""Ground and stockpiles, as analytically intersectable primitives.

A heightmap would need ray marching; a plane plus a few cones is exact, which
matters when four million rays go through this per scene. The cones are also
physically honest: loose material piles at its angle of repose, so the
radius-to-height ratio is a material property rather than a free parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Array

#: Angle of repose for typical aggregate. Sets the cone's slope.
REPOSE_DEG = 34.0


@dataclass
class Stockpile:
    """A conical pile of material, apex up, sitting on the ground plane."""

    x: float
    y: float
    height: float
    material: str = "soil"

    @property
    def radius(self) -> float:
        return self.height / np.tan(np.radians(REPOSE_DEG))

    def height_at(self, px: Array, py: Array) -> Array:
        """Surface elevation over a grid of (x, y), for the heightmap topic."""
        r = np.hypot(px - self.x, py - self.y)
        return np.maximum(0.0, self.height * (1.0 - r / max(self.radius, 1e-6)))

    def intersect(self, origin: Array, dirs: Array) -> Array:
        """Ray-cone distances, `inf` where a ray misses. `dirs` is (N, 3)."""
        k = self.radius / max(self.height, 1e-6)
        k2 = k * k
        o = origin - np.array([self.x, self.y, self.height])

        a = dirs[:, 0] ** 2 + dirs[:, 1] ** 2 - k2 * dirs[:, 2] ** 2
        b = 2.0 * (o[0] * dirs[:, 0] + o[1] * dirs[:, 1] - k2 * o[2] * dirs[:, 2])
        c = o[0] ** 2 + o[1] ** 2 - k2 * o[2] ** 2

        t = np.full(dirs.shape[0], np.inf)
        disc = b * b - 4.0 * a * c
        ok = (disc >= 0.0) & (np.abs(a) > 1e-12)
        if not np.any(ok):
            return t
        sq = np.sqrt(np.maximum(disc[ok], 0.0))
        for root in ((-b[ok] - sq) / (2.0 * a[ok]), (-b[ok] + sq) / (2.0 * a[ok])):
            z = origin[2] + root * dirs[ok, 2]
            # Only the lower nappe, and only between ground and apex.
            valid = (root > 0.1) & (z >= 0.0) & (z <= self.height)
            candidate = np.where(valid, root, np.inf)
            current = t[ok]
            t[ok] = np.minimum(current, candidate)
        return t


@dataclass
class Terrain:
    """The ground plane plus whatever is piled on it."""

    stockpiles: list[Stockpile]
    extent: float = 60.0

    def intersect(self, origin: Array, dirs: Array) -> Array:
        # Flat ground at z = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ground = np.where(dirs[:, 2] < -1e-9, -origin[2] / dirs[:, 2], np.inf)
        t_ground = np.where(np.isfinite(t_ground) & (t_ground > 0.0), t_ground, np.inf)
        t = t_ground
        for pile in self.stockpiles:
            t = np.minimum(t, pile.intersect(origin, dirs))
        return t

    def heightmap(self, cells: int = 120) -> tuple[Array, float]:
        """(cells, cells) elevation grid over [-extent, extent], and its pitch."""
        axis = np.linspace(-self.extent, self.extent, cells)
        gx, gy = np.meshgrid(axis, axis, indexing="xy")
        z = np.zeros_like(gx)
        for pile in self.stockpiles:
            z = np.maximum(z, pile.height_at(gx, gy))
        return z, float(2.0 * self.extent / cells)
