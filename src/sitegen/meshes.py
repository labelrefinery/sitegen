"""Triangle meshes for every actor link, and the file format they live in.

One mesh set, two sensors. The LiDAR intersects these triangles and Cycles
renders these same triangles, built from the same arrays -- so "a return in the
cloud and a pixel in the image describe the same surface" survives the move off
boxes. It is the reason the meshes are *arrays* rather than a `.blend` file:
anything Blender-only would be a second definition of the geometry, and a second
definition is a thing that can drift.

Every mesh is authored in its link's own frame, in metres, with the same origin
and axes the kinematic chain already used -- a boom runs along +x from its foot,
the house sits on the slew ring at z = 0. The ground-truth cuboid is then the
tight bounding box of the posed mesh rather than a hand-written envelope, which
is what keeps the oracle describing what the sensors actually saw. The truck
gains wheels this way, and its box grows down to the ground: the old box model
had an eight-tonne hauler floating 0.6 m in the air, and nothing noticed.

Shapes are built from y-extruded convex profiles and boxes. That sounds like a
limitation and mostly is not: a track frame, a counterweight, a bucket and a
wheel are all one convex outline swept sideways. The worker is the exception --
it is a rigged CC0 human baked to triangles by `tools/bake_worker.py`, because
a person is exactly the silhouette that cannot be faked with prisms, and
docs/PROBE-WORKER.md is the measurement that says so.

Triangles carry a material index, kept only for the renderer. The LiDAR ignores
it; Cycles needs it, because a hi-vis vest that renders the same grey as the
tracks throws away the one cue the probe showed was doing the work.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .geometry import Array

ASSETS = Path(__file__).parent / "assets"

TriArray = NDArray[np.int32]


@dataclass(frozen=True)
class Mesh:
    """A triangle soup in one link's local frame."""

    vertices: Array
    triangles: TriArray
    tri_material: TriArray
    materials: tuple[str, ...]

    def bounds(self) -> tuple[Array, Array]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


class _Builder:
    """Accumulates primitives into one mesh, tracking which material each
    triangle belongs to."""

    def __init__(self) -> None:
        self._v: list[Array] = []
        self._f: list[TriArray] = []
        self._m: list[TriArray] = []
        self._names: list[str] = []

    def _material(self, name: str) -> int:
        if name not in self._names:
            self._names.append(name)
        return self._names.index(name)

    def add(self, verts: Array, faces: TriArray, material: str) -> None:
        offset = sum(len(v) for v in self._v)
        self._v.append(np.asarray(verts, dtype=np.float64))
        self._f.append(np.asarray(faces, dtype=np.int32) + offset)
        self._m.append(np.full(len(faces), self._material(material), dtype=np.int32))

    def box(
        self,
        lo: tuple[float, float, float],
        hi: tuple[float, float, float],
        material: str,
    ) -> None:
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        verts = np.array(
            [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ]
        )
        faces = np.array(
            [
                [0, 2, 1], [0, 3, 2],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.int32,
        )
        self.add(verts, faces, material)

    def prism(
        self, profile: Array, y0: float, y1: float, material: str
    ) -> None:
        """A convex (x, z) outline swept from y0 to y1.

        Convex is what lets the end caps be a triangle fan from vertex 0, and
        every profile here -- track frame, boom, bucket, wheel -- is convex.
        """
        p = np.asarray(profile, dtype=np.float64)
        n = len(p)
        verts = np.vstack(
            [
                np.column_stack([p[:, 0], np.full(n, y0), p[:, 1]]),
                np.column_stack([p[:, 0], np.full(n, y1), p[:, 1]]),
            ]
        )
        faces: list[list[int]] = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])
            faces.append([i, n + j, n + i])
        for i in range(1, n - 1):
            faces.append([0, i + 1, i])
            faces.append([n, n + i, n + i + 1])
        self.add(verts, np.array(faces, dtype=np.int32), material)

    def wheel(
        self,
        x: float,
        z: float,
        radius: float,
        y0: float,
        y1: float,
        material: str = "rubber",
        segments: int = 14,
    ) -> None:
        a = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        self.prism(
            np.column_stack([x + radius * np.cos(a), z + radius * np.sin(a)]),
            y0,
            y1,
            material,
        )

    def mirrored_wheel(
        self, x: float, z: float, radius: float, inner: float, outer: float, **kw
    ) -> None:
        self.wheel(x, z, radius, inner, outer, **kw)
        self.wheel(x, z, radius, -outer, -inner, **kw)

    def build(self) -> Mesh:
        return Mesh(
            vertices=np.vstack(self._v),
            triangles=np.vstack(self._f).astype(np.int32),
            tri_material=np.concatenate(self._m).astype(np.int32),
            materials=tuple(self._names),
        )


# -- the machines ----------------------------------------------------------
#
# Dimensions follow actors.py, so a mesh fills the envelope the box occupied
# and `--actors boxes` keeps reproducing the old numbers. Where a mesh sticks
# out of that envelope it is because the box was wrong: wheels, a bent boom.


def _undercarriage() -> Mesh:
    """Two track frames and the car body between them, 4.5 x 2.8 x 0.9."""
    b = _Builder()
    # Classic track outline: flat on the ground, sloped up at both ends over
    # the idler and the sprocket.
    track = np.array(
        [
            [-2.25, 0.12], [-1.95, 0.0], [1.95, 0.0], [2.25, 0.12],
            [2.25, 0.62], [1.75, 0.90], [-1.75, 0.90], [-2.25, 0.62],
        ]
    )
    for sign in (1.0, -1.0):
        outer, inner = sign * 1.40, sign * 1.02
        b.prism(track, min(outer, inner), max(outer, inner), "track")
    b.box((-1.35, -1.02, 0.38), (1.35, 1.02, 0.86), "steel")
    return b.build()


def _house() -> Mesh:
    """Counterweight, engine deck, operator cab and boom pedestal.

    The cab sits on the left and the boom foot on the right, which is what an
    excavator looks like from any angle a camera on the machine will see it.
    """
    b = _Builder()
    b.box((-2.10, -1.35, 0.00), (-1.45, 1.35, 1.95), "counterweight")  # rear slab
    b.box((-1.45, -1.35, 0.00), (0.55, 1.35, 1.55), "body")  # engine deck
    b.box((-2.10, -1.35, 0.00), (1.50, 1.35, 0.22), "steel")  # house floor
    # Operator cab: a tapered box with a raked windscreen, on the left.
    b.prism(
        np.array([[0.18, 0.22], [1.42, 0.22], [1.42, 1.55], [1.15, 2.15], [0.18, 2.15]]),
        0.18,
        1.28,
        "cab",
    )
    b.prism(
        np.array([[1.30, 0.30], [1.44, 0.30], [1.44, 1.50], [1.16, 2.05], [1.06, 2.05]]),
        0.22,
        1.24,
        "glass",
    )
    b.box((0.30, -1.30, 0.22), (1.50, -0.15, 1.05), "body")  # boom pedestal
    return b.build()


def _beam(length: float, rise: float, near: float, far: float) -> Mesh:
    """A tapered structural member along +x, bowed up by `rise` in the middle.

    Booms and sticks are welded box sections, deeper at the pinned end and
    bent so the bucket can reach past the tracks. The centreline returns to
    z = 0 at the tip, which is where the next joint is.
    """
    b = _Builder()
    xs = np.linspace(0.0, length, 7)
    mid = np.sin(np.pi * xs / length) * rise
    half = np.linspace(near, far, 7) / 2.0
    top = np.column_stack([xs, mid + half])
    bottom = np.column_stack([xs[::-1], (mid - half)[::-1]])
    b.prism(np.vstack([top, bottom]), -0.28, 0.28, "body")
    return b.build()


def _bucket() -> Mesh:
    """A curved scoop with five teeth, hung from the stick pin at the origin.

    The outline is an arc that wraps most of the way round plus one point out
    at the cutting edge, which is convex read in order and so still sweeps as
    a prism.
    """
    b = _Builder()
    angles = np.linspace(-0.45, 2.98, 9)
    back = np.column_stack([0.65 + 0.60 * np.cos(angles), -0.02 + 0.55 * np.sin(angles)])
    profile = np.vstack([back, [1.20, -0.45]])
    b.prism(profile, -0.50, 0.50, "steel")
    for y in np.linspace(-0.42, 0.42, 5):
        b.prism(
            np.array([[1.10, -0.48], [1.34, -0.44], [1.10, -0.26]]),
            y - 0.045,
            y + 0.045,
            "steel",
        )
    return b.build()


#: An articulated hauler is two units on a centre hinge, and the split here is
#: that hinge: `cab` is the tractor -- engine, cab, steer axle -- and `bed` is
#: the rear unit it pulls, chassis and tandem axles under the dump body. Both
#: cuboids then describe something that exists, which the old pair did not: the
#: box model had no wheels at all and started 0.6 m off the ground.
def _truck_cab() -> Mesh:
    """Tractor unit: bonnet, raked cab, steer axle, hitch stub."""
    b = _Builder()
    b.box((1.60, -0.50, 0.72), (2.30, 0.50, 1.10), "chassis")  # hitch
    b.box((2.20, -1.25, 0.62), (4.40, 1.25, 1.10), "chassis")
    b.prism(  # bonnet, then the cab with a raked screen
        np.array(
            [
                [2.15, 1.05], [4.35, 1.05], [4.35, 2.10],
                [3.55, 2.20], [3.30, 3.45], [2.15, 3.45],
            ]
        ),
        -1.20,
        1.20,
        "body",
    )
    b.prism(
        np.array([[3.32, 2.30], [3.58, 2.28], [3.42, 3.40], [3.20, 3.40]]),
        -1.14,
        1.14,
        "glass",
    )
    b.mirrored_wheel(3.55, 0.74, 0.74, 0.95, 1.45)
    return b.build()


def _truck_bed() -> Mesh:
    """Rear unit: chassis, tandem axles, and an open dump body over them."""
    b = _Builder()
    b.box((-4.40, -0.62, 0.62), (1.90, 0.62, 1.05), "chassis")
    for x in (-0.80, -2.70):
        b.mirrored_wheel(x, 0.78, 0.78, 0.95, 1.45)
    b.box((-4.40, -1.45, 1.05), (2.00, 1.45, 1.24), "bed")  # floor
    for y0, y1 in ((-1.45, -1.28), (1.28, 1.45)):
        b.prism(
            np.array([[-4.40, 1.24], [2.00, 1.24], [2.00, 3.20], [-4.40, 2.55]]),
            y0,
            y1,
            "bed",
        )
    b.box((1.80, -1.45, 1.24), (2.00, 1.45, 3.20), "bed")  # headboard
    return b.build()


def _grade_stake() -> Mesh:
    """A 50 mm square post with a sharpened foot and a painted head."""
    b = _Builder()
    b.prism(
        np.array([[-0.025, 0.10], [0.0, 0.0], [0.025, 0.10], [0.025, 0.80], [-0.025, 0.80]]),
        -0.025,
        0.025,
        "timber",
    )
    b.box((-0.025, -0.025, 0.80), (0.025, 0.025, 1.00), "marker")
    return b.build()


# -- PLY, for the meshes that were not built here --------------------------


def read_ply(path: Path) -> Mesh:
    """Binary little-endian PLY with a per-face material index.

    Written rather than pulled in from a library because it is forty lines and
    the alternative is a dependency that exists to parse one file format for
    three files. Material names ride in a header comment.
    """
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    counts: dict[str, int] = {}
    materials: list[str] = []
    for line in header:
        if line.startswith("element "):
            _, name, n = line.split()
            counts[name] = int(n)
        elif line.startswith("comment material "):
            materials.append(line.split(" ", 2)[2])
    n_v, n_f = counts["vertex"], counts["face"]
    verts = np.frombuffer(data, dtype="<f4", count=n_v * 3, offset=end)
    verts = verts.reshape(n_v, 3).astype(np.float64)
    face_dtype = np.dtype([("n", "u1"), ("i", "<i4", 3), ("m", "u1")])
    faces = np.frombuffer(data, dtype=face_dtype, count=n_f, offset=end + n_v * 12)
    return Mesh(
        vertices=verts,
        triangles=faces["i"].astype(np.int32),
        tri_material=faces["m"].astype(np.int32),
        materials=tuple(materials),
    )


def write_ply(mesh: Mesh, path: Path) -> None:
    """The inverse of `read_ply`. Used by the asset bake, and by the Cycles
    bridge to hand Blender the exact triangles the raycaster intersected."""
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment written by sitegen.meshes",
        *(f"comment material {name}" for name in mesh.materials),
        f"element vertex {len(mesh.vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(mesh.triangles)}",
        "property list uchar int vertex_indices",
        "property uchar material",
        "end_header",
        "",
    ]
    face_dtype = np.dtype([("n", "u1"), ("i", "<i4", 3), ("m", "u1")])
    faces = np.zeros(len(mesh.triangles), dtype=face_dtype)
    faces["n"] = 3
    faces["i"] = mesh.triangles
    faces["m"] = mesh.tri_material
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(mesh.vertices.astype("<f4").tobytes())
        f.write(faces.tobytes())


#: Link name -> how to build it. Keys match the `class_name` of the box each
#: mesh replaces, except the worker, whose two entries are two poses of one
#: class: the spotter stands and the crosser walks.
_BUILDERS = {
    "excavator.undercarriage": _undercarriage,
    "excavator.house": _house,
    "excavator.boom": lambda: _beam(5.7, 0.55, 0.78, 0.58),
    "excavator.stick": lambda: _beam(2.9, 0.18, 0.66, 0.48),
    "excavator.bucket": _bucket,
    "haul_truck.cab": _truck_cab,
    "haul_truck.bed": _truck_bed,
    "grade_stake": _grade_stake,
    "worker.stand": lambda: read_ply(ASSETS / "worker_stand.ply"),
    "worker.walk": lambda: read_ply(ASSETS / "worker_walk.ply"),
}


@functools.cache
def mesh(name: str) -> Mesh:
    """The mesh for one link, built once and shared."""
    return _BUILDERS[name]()


@functools.cache
def bounds(name: str) -> tuple[Array, Array]:
    """Local AABB, which is where the ground-truth cuboid comes from."""
    return mesh(name).bounds()


def names() -> tuple[str, ...]:
    return tuple(_BUILDERS)
