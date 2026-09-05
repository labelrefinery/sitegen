"""A regular elevation grid that material can be taken out of and put back.

This is the one piece of state in sitegen that is not a pure function of `t`.
Everything else -- joint angles, actor poses, the sweep -- is computed from the
clock; the ground is not, because a hole is the record of what happened to it.
So the grid is stepped forward once, up front, and the timeline that falls out
of it is what the rest of the program reads. `earthworks.py` does the stepping.

**Why a heightfield and not a particle bed.** AGX Terrain and Vortex Studio
model a bucket in soil as a heightfield with a few thousand particles under the
tool, so the material in front of the cutting edge piles, shears and falls
back. That is the right model for an operator-training simulator, where the
question is what the machine *feels*. Here the question is what a LiDAR sees
and what a volume-tracking pipeline can recover from it, and a 0.25 m grid
answers both at a cost the generator can pay six hundred times a scene. What is
lost is listed in `docs/TERRAIN.md` and it is real: no particles, no
compaction, no swell factor, one material.

**Mass is conserved exactly.** Every operation here either moves material
between cells or reports the volume it removed, and the relaxation is written
as symmetric pairwise transfers, so `sum(z)` is invariant to floating-point
round-off rather than to a tolerance. That is what makes
`cut == bucket + bed + hauled + stockpiles` an assertion rather than an
estimate, and the oracle worth publishing.

**The angle of repose does the shaping.** After any change the field is
relaxed until no two neighbouring nodes differ by more than one cell of run at
`REPOSE_DEG` -- the same 34 deg the analytic stockpiles were always drawn at, so
sampling the old cones onto the grid produces a field that is already relaxed
and the first frame is not a slump. Loose material has no cohesion in this
model, which is why a cut wall slumps like a spoil pile instead of standing the
way a trench in clay would.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import Array
from .meshes import Mesh
from .terrain import REPOSE_DEG, Terrain

#: Half-width of the deforming patch, in metres. Outside it the ground is the
#: analytic plane at z = 0, which is exact and free -- the site is 50 m across
#: and the machine, both stockpiles and every dumping point are inside it, so a
#: wider grid would be paying for cells that can never change.
GRID_EXTENT_M = 25.0

#: Cell pitch. A 20-tonne machine's bucket is 1.0 m wide and takes a bite about
#: 0.7 m long, so 0.25 m puts roughly 4 x 3 cells under the cutting edge: fine
#: enough that a bite is a shape rather than one cell, coarse enough that the
#: patch is 80,000 triangles instead of 1.3 million.
GRID_PITCH_M = 0.25

#: Relaxation transfer coefficient. Each of the four neighbour directions moves
#: half of this fraction of the excess over the repose limit per iteration, so
#: a node sheds at most 0.8 of its excess in one sweep and the iteration cannot
#: overshoot into oscillation.
_RELAX_OMEGA = 0.4

#: Below this the field counts as relaxed. A nanometre of slope violation over
#: a 0.25 m run is six orders of magnitude under anything a sensor could see.
_RELAX_TOL = 1e-9

_MAX_RELAX_ITERATIONS = 4000

_VERSION_COUNTER = 0


def _next_version() -> int:
    """Globally unique ids for committed snapshots.

    Unique across `Scene` instances rather than per scene, because the ray
    intersector caches by version and two scenes in one process must not be
    able to collide on a number.
    """
    global _VERSION_COUNTER
    _VERSION_COUNTER += 1
    return _VERSION_COUNTER


@dataclass
class Heightfield:
    """Node elevations on a regular grid, in metres, in some frame's x-y.

    Nodes rather than cells: `z[i, j]` is the surface at
    `(x0 + i * pitch, y0 + j * pitch)`, and each node owns one cell of area for
    the purposes of volume. The border ring of the ground field is held at
    zero, which is what lets the analytic plane outside the patch meet the mesh
    inside it without a seam.
    """

    z: Array
    x0: float
    y0: float
    pitch: float

    # -- geometry ----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.z.shape[0]), int(self.z.shape[1]))

    def axes(self) -> tuple[Array, Array]:
        nx, ny = self.shape
        return (
            (self.x0 + self.pitch * np.arange(nx)).astype(np.float64),
            (self.y0 + self.pitch * np.arange(ny)).astype(np.float64),
        )

    def volume(self) -> float:
        """Signed volume between the surface and z = 0, in cubic metres."""
        return float(self.z.sum()) * self.pitch * self.pitch

    def height_at(self, x: float, y: float) -> float:
        """Bilinear surface elevation, clamped to the patch."""
        nx, ny = self.shape
        u = float(np.clip((x - self.x0) / self.pitch, 0.0, nx - 1.0))
        v = float(np.clip((y - self.y0) / self.pitch, 0.0, ny - 1.0))
        i = min(int(u), nx - 2)
        j = min(int(v), ny - 2)
        fu, fv = u - i, v - j
        return float(
            self.z[i, j] * (1 - fu) * (1 - fv)
            + self.z[i + 1, j] * fu * (1 - fv)
            + self.z[i, j + 1] * (1 - fu) * fv
            + self.z[i + 1, j + 1] * fu * fv
        )

    def copy(self) -> Heightfield:
        return Heightfield(self.z.copy(), self.x0, self.y0, self.pitch)

    def mesh(self) -> tuple[Array, NDArray[np.int32]]:
        """Vertices and triangles for the patch, in the grid's own frame.

        Fixed topology: only z ever changes, which is what lets Blender keep
        one mesh datablock for the whole run and rewrite its coordinates when
        the version does.
        """
        nx, ny = self.shape
        ax, ay = self.axes()
        gx, gy = np.meshgrid(ax, ay, indexing="ij")
        vertices = np.stack([gx, gy, self.z], axis=-1).reshape(-1, 3)
        return vertices, grid_triangles(nx, ny)

    # -- editing -----------------------------------------------------------

    def carve(self, corners: Array, capacity: float) -> float:
        """Cut down to a plane inside a quadrilateral. Returns volume removed.

        `corners` is four world points in order around the quad -- the bucket's
        mouth, projected onto the grid -- and the surface is cut to the plane
        through their z values, interpolated barycentrically. The cutting edge
        and the bucket floor behind it are between them the only part of a
        bucket that decides where the new surface ends up.

        `capacity` caps the bite: what a bucket takes out of a face is bounded
        by what a bucket holds, not by how much soil happens to be in front of
        it. When the geometric cut is larger the depth is scaled down
        uniformly, so the removed volume is exactly `capacity` -- a partial
        bite, which is what an operator feathering the stick actually gets.
        """
        if capacity <= 0.0:
            return 0.0
        window = self._rasterise(corners)
        if window is None:
            return 0.0
        (i0, i1, j0, j1), inside, target = window
        patch = self.z[i0:i1, j0:j1]
        cut = np.where(inside, np.maximum(patch - target, 0.0), 0.0)
        total = float(cut.sum()) * self.pitch * self.pitch
        if total <= 0.0:
            return 0.0
        if total > capacity:
            cut *= capacity / total
            total = capacity
        self.z[i0:i1, j0:j1] = patch - cut
        self.relax(i0, i1, j0, j1)
        return total

    def deposit(self, x: float, y: float, volume: float, radius: float = 1.0) -> None:
        """Drop `volume` at (x, y) as a small cone, then let it find its angle.

        The cone is a placement, not a shape claim: it is normalised to the
        exact volume asked for, and the relaxation spreads whatever will not
        stand. Its radius is widened to the one a cone of that volume would
        have at the angle of repose, which is not cosmetic: slope limiting is a
        diffusion, so it takes O(n^2) sweeps to move material n cells, and
        starting a 12 m3 dump as an 11 m spike costs thousands of iterations to
        undo. Starting it at roughly the right footprint costs tens.
        """
        if volume <= 0.0:
            return
        radius = max(
            radius,
            float(
                (3.0 * volume / (np.pi * np.tan(np.radians(REPOSE_DEG)))) ** (1 / 3)
            ),
        )
        nx, ny = self.shape
        reach = radius + self.pitch
        i0 = max(int(np.floor((x - reach - self.x0) / self.pitch)), 0)
        i1 = min(int(np.ceil((x + reach - self.x0) / self.pitch)) + 1, nx)
        j0 = max(int(np.floor((y - reach - self.y0) / self.pitch)), 0)
        j1 = min(int(np.ceil((y + reach - self.y0) / self.pitch)) + 1, ny)
        if i1 <= i0 or j1 <= j0:
            return
        ax, ay = self.axes()
        gx, gy = np.meshgrid(ax[i0:i1], ay[j0:j1], indexing="ij")
        shape = np.maximum(0.0, 1.0 - np.hypot(gx - x, gy - y) / radius)
        weight = float(shape.sum()) * self.pitch * self.pitch
        if weight <= 0.0:
            return
        self.z[i0:i1, j0:j1] += shape * (volume / weight)
        self.relax(i0, i1, j0, j1)

    # -- the angle of repose -----------------------------------------------

    def relax(
        self, i0: int = 0, i1: int | None = None, j0: int = 0, j1: int | None = None
    ) -> int:
        """Slope-limit a window until nothing in it exceeds the repose angle.

        The window grows until the slump stops touching its border, so the
        result does not depend on how big a box the caller guessed: material
        that reached the edge would otherwise pile against a wall that is not
        there. Every transfer is a symmetric pair, so the window's total is
        invariant and the site's mass balance closes to round-off.

        Returns the iteration count, which is only interesting when tuning: a
        cut settles in a few dozen, a dump on a pile in a few hundred.
        """
        nx, ny = self.shape
        i1 = nx if i1 is None else i1
        j1 = ny if j1 is None else j1
        pad = max(int(round(3.0 / self.pitch)), 4)
        iterations = 0
        for _ in range(6):
            a0, a1 = max(i0 - pad, 0), min(i1 + pad, nx)
            b0, b1 = max(j0 - pad, 0), min(j1 + pad, ny)
            window = self.z[a0:a1, b0:b1]
            before = window.copy()
            iterations += _relax_in_place(window, self.pitch)
            if not _border_changed(before, window):
                return iterations
            if (a0, b0, a1, b1) == (0, 0, nx, ny):
                return iterations
            i0, i1, j0, j1 = a0, a1, b0, b1
        return iterations

    def slope_excess(self) -> float:
        """Worst violation of the repose limit anywhere, in metres of rise."""
        limit = self.pitch * float(np.tan(np.radians(REPOSE_DEG)))
        worst = 0.0
        for axis in (0, 1):
            d = np.abs(np.diff(self.z, axis=axis))
            worst = max(worst, float(d.max(initial=0.0)) - limit)
        return max(worst, 0.0)

    # -- internals ---------------------------------------------------------

    def _rasterise(
        self, corners: Array
    ) -> tuple[tuple[int, int, int, int], Array, Array] | None:
        """Cells whose centres fall inside the quad, and the quad's z there.

        Split into two triangles and interpolated barycentrically rather than
        fitted as a plane, because a quad whose four corners are not coplanar
        -- which is every posed bucket mouth -- has no plane, and a fit would
        put the cut floor somewhere neither triangle is.
        """
        nx, ny = self.shape
        xs, ys = corners[:, 0], corners[:, 1]
        i0 = max(int(np.floor((xs.min() - self.x0) / self.pitch)), 0)
        i1 = min(int(np.ceil((xs.max() - self.x0) / self.pitch)) + 1, nx)
        j0 = max(int(np.floor((ys.min() - self.y0) / self.pitch)), 0)
        j1 = min(int(np.ceil((ys.max() - self.y0) / self.pitch)) + 1, ny)
        if i1 <= i0 or j1 <= j0:
            return None

        ax, ay = self.axes()
        gx, gy = np.meshgrid(ax[i0:i1], ay[j0:j1], indexing="ij")
        inside = np.zeros(gx.shape, dtype=bool)
        target = np.full(gx.shape, np.inf)
        for tri in ((0, 1, 2), (0, 2, 3)):
            p, q, r = corners[tri[0]], corners[tri[1]], corners[tri[2]]
            det = (q[1] - r[1]) * (p[0] - r[0]) + (r[0] - q[0]) * (p[1] - r[1])
            if abs(det) < 1e-12:
                continue
            wp = ((q[1] - r[1]) * (gx - r[0]) + (r[0] - q[0]) * (gy - r[1])) / det
            wq = ((r[1] - p[1]) * (gx - r[0]) + (p[0] - r[0]) * (gy - r[1])) / det
            wr = 1.0 - wp - wq
            hit = (wp >= 0.0) & (wq >= 0.0) & (wr >= 0.0)
            if not hit.any():
                continue
            plane = wp * p[2] + wq * q[2] + wr * r[2]
            target = np.where(hit, np.minimum(target, plane), target)
            inside |= hit
        if not inside.any():
            return None
        return (i0, i1, j0, j1), inside, np.where(inside, target, 0.0)


def grid_triangles(nx: int, ny: int) -> NDArray[np.int32]:
    """Two triangles per cell, split along the same diagonal everywhere."""
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (ii * ny + jj).ravel()
    b = ((ii + 1) * ny + jj).ravel()
    c = ((ii + 1) * ny + jj + 1).ravel()
    d = (ii * ny + jj + 1).ravel()
    return np.concatenate(
        [np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)]
    ).astype(np.int32)


def _relax_in_place(z: Array, pitch: float) -> int:
    limit = pitch * float(np.tan(np.radians(REPOSE_DEG)))
    for iteration in range(_MAX_RELAX_ITERATIONS):
        delta = np.zeros_like(z)
        worst = 0.0
        for axis in (0, 1):
            d = np.diff(z, axis=axis)
            excess = np.clip(np.abs(d) - limit, 0.0, None)
            worst = max(worst, float(excess.max(initial=0.0)))
            move = np.sign(d) * excess * (0.5 * _RELAX_OMEGA)
            if axis == 0:
                delta[1:, :] -= move
                delta[:-1, :] += move
            else:
                delta[:, 1:] -= move
                delta[:, :-1] += move
        if worst <= _RELAX_TOL:
            return iteration
        z += delta
    return _MAX_RELAX_ITERATIONS


def _border_changed(before: Array, after: Array, tol: float = 1e-12) -> bool:
    edges = (
        (before[0, :], after[0, :]),
        (before[-1, :], after[-1, :]),
        (before[:, 0], after[:, 0]),
        (before[:, -1], after[:, -1]),
    )
    return any(float(np.abs(a - b).max(initial=0.0)) > tol for a, b in edges)


def ground_field(
    terrain: Terrain, extent: float = GRID_EXTENT_M, pitch: float = GRID_PITCH_M
) -> Heightfield:
    """Sample the analytic ground and stockpiles onto the grid.

    The cones were already drawn at the angle of repose, so the sampled field
    satisfies the slope limit the moment it exists -- `relax()` on it is a
    no-op, which `tests/test_terrain.py` checks rather than assumes.
    """
    n = int(round(2.0 * extent / pitch)) + 1
    axis = (-extent + pitch * np.arange(n)).astype(np.float64)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    z: Array = np.zeros_like(gx)
    for pile in terrain.stockpiles:
        z = np.maximum(z, pile.height_at(gx, gy))
    # The border ring meets the analytic plane outside the patch.
    z[0, :] = z[-1, :] = 0.0
    z[:, 0] = z[:, -1] = 0.0
    return Heightfield(z=z, x0=-extent, y0=-extent, pitch=pitch)


# -- what a raycaster and a renderer are handed ----------------------------


@dataclass(frozen=True)
class GroundSurface:
    """One committed snapshot of the ground: the patch mesh plus the plane.

    Committed, because the simulation moves soil every 50 ms and rebuilding a
    BVH over 80,000 triangles at that rate would cost more than the rest of the
    scene put together. What the sensors see is the most recent snapshot,
    taken at `earthworks.TERRAIN_COMMIT_HZ`.

    Rays are answered in two parts. Inside the patch the mesh does it, through
    embree. Outside it the ground is still the analytic plane, tested exactly
    as it always was -- and a plane hit landing *inside* the patch footprint is
    discarded, because otherwise the plane would floor over every cut.
    """

    version: int
    field: Heightfield
    extent: float

    @staticmethod
    def commit(field: Heightfield) -> GroundSurface:
        snapshot = field.copy()
        return GroundSurface(
            version=_next_version(), field=snapshot, extent=float(-snapshot.x0)
        )

    def hits(self, origin: Array, dirs: Array) -> tuple[Array, Array]:
        """(distance, normal) per ray; `inf` where the ray misses the ground."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t_plane = np.where(dirs[:, 2] < -1e-9, -origin[2] / dirs[:, 2], np.inf)
        t_plane = np.where(np.isfinite(t_plane) & (t_plane > 0.0), t_plane, np.inf)
        px = origin[0] + t_plane * dirs[:, 0]
        py = origin[1] + t_plane * dirs[:, 1]
        outside = (np.abs(px) >= self.extent) | (np.abs(py) >= self.extent)
        t = np.where(outside, t_plane, np.inf)
        normal = np.tile(np.array([0.0, 0.0, 1.0]), (len(dirs), 1))

        intersector, normals = _bvh(self)
        tri, ray, location = intersector.intersects_id(
            np.broadcast_to(origin, dirs.shape).astype(np.float64),
            dirs,
            multiple_hits=False,
            return_locations=True,
        )
        if len(ray):
            distance = np.linalg.norm(location - origin, axis=1)
            closer = distance < t[ray]
            ray, tri, distance = ray[closer], tri[closer], distance[closer]
            t[ray] = distance
            normal[ray] = normals[tri]
        return t, normal


#: One slot, because the writer, the tests and the Blender job all walk the
#: scene forward in time and never look back. A version that has scrolled off
#: is rebuilt if someone does look back, which costs 39 ms and is correct.
_BVH: dict[str, Any] = {"version": None, "intersector": None, "normals": None}


def _bvh(surface: GroundSurface) -> tuple[Any, Array]:
    if _BVH["version"] != surface.version:
        try:
            from trimesh import Trimesh
            from trimesh.ray.ray_pyembree import RayMeshIntersector
        except ImportError as exc:  # pragma: no cover -- a dependency, not a mode
            raise RuntimeError(
                "deforming terrain needs trimesh and embreex; `make install`"
            ) from exc

        vertices, triangles = surface.field.mesh()
        corners = vertices[triangles]
        face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
        # The patch is a graph over x-y, so the outward side is the one with
        # positive z. Forcing it is cheaper than trusting the winding.
        face *= np.sign(face[:, 2])[:, None]
        _BVH["version"] = surface.version
        _BVH["normals"] = face
        _BVH["intersector"] = RayMeshIntersector(
            Trimesh(vertices=vertices, faces=triangles, process=False)
        )
    return _BVH["intersector"], _BVH["normals"]


# -- the material in the bucket, and in the bed ----------------------------

#: Interior of the hauler's dump body, in the bed link's own frame: the floor
#: is at z = 1.24 and the walls stand at x = -4.40..1.80, y = +-1.28 (see
#: `meshes._truck_bed`). The load grid stops one cell short of each wall so a
#: heap leans on the body rather than poking through it.
BED_FLOOR_Z = 1.24
BED_X = (-4.30, 1.70)
BED_Y = (-1.18, 1.18)
BED_PITCH_M = 0.20


def bed_field() -> Heightfield:
    """An empty load surface, in the bed link's frame."""
    nx = int(round((BED_X[1] - BED_X[0]) / BED_PITCH_M)) + 1
    ny = int(round((BED_Y[1] - BED_Y[0]) / BED_PITCH_M)) + 1
    return Heightfield(
        z=np.zeros((nx, ny)), x0=BED_X[0], y0=BED_Y[0], pitch=BED_PITCH_M
    )


def bed_mesh(load: Heightfield, material: str = "soil") -> Mesh:
    """The load as a closed prism: heaped top, vertical skirt, flat floor.

    A skirt rather than an open sheet, because a LiDAR on a mast 15 m away sees
    the load at a grazing angle over the tailgate, and an open sheet is
    invisible edge-on from exactly there.
    """
    nx, ny = load.shape
    ax, ay = load.axes()
    gx, gy = np.meshgrid(ax, ay, indexing="ij")
    top = np.stack([gx, gy, BED_FLOOR_Z + load.z], axis=-1).reshape(-1, 3)
    floor = np.stack([gx, gy, np.full_like(gx, BED_FLOOR_Z)], axis=-1).reshape(-1, 3)
    vertices = np.vstack([top, floor])
    offset = nx * ny

    border: list[tuple[int, int]] = []
    border += [(i * ny, (i + 1) * ny) for i in range(nx - 1)]
    border += [((i + 1) * ny + ny - 1, i * ny + ny - 1) for i in range(nx - 1)]
    border += [((nx - 1) * ny + j, (nx - 1) * ny + j + 1) for j in range(ny - 1)]
    border += [(j + 1, j) for j in range(ny - 1)]
    skirt = np.array(
        [[a, b, b + offset] for a, b in border]
        + [[a, b + offset, a + offset] for a, b in border],
        dtype=np.int32,
    )
    triangles = np.vstack([grid_triangles(nx, ny), skirt]).astype(np.int32)
    return Mesh(
        vertices=vertices,
        triangles=triangles,
        tri_material=np.zeros(len(triangles), dtype=np.int32),
        materials=(material,),
    )


@dataclass
class BedLoad:
    """What is in one hauler's body, as a little heightfield of its own.

    The same relaxation runs on it, with the grid border standing in for the
    body sides, so the load heaps against the headboard and sits at the angle
    of repose exactly as the spoil outside does. It is the same physics at a
    fiftieth of the area, and it means the visible load and the number in the
    oracle are one quantity rather than two that have to be kept in step.
    """

    surface: Heightfield = dataclasses.field(default_factory=bed_field)

    def volume(self) -> float:
        return self.surface.volume()

    def add(self, volume: float, capacity: float) -> float:
        """Tip `volume` in. Returns whatever did not fit."""
        room = capacity - self.volume()
        taken = max(min(volume, room), 0.0)
        if taken > 0.0:
            self.surface.deposit(
                0.5 * (BED_X[0] + BED_X[1]), 0.0, taken, radius=1.2
            )
        return volume - taken

    def empty(self) -> float:
        gone = self.volume()
        self.surface.z[:] = 0.0
        return gone
