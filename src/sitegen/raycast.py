"""The one intersection layer both sensors go through.

A camera is a spinning sensor whose rays happen to go through a pixel grid
instead of around an axis, so there is no reason for the two to intersect
different geometry -- and every reason for them not to. What changed when the
actors became meshes is only what sits behind this interface: a slab test per
oriented box became a BVH over triangles.

Terrain used to stay analytic, for the same reason: a plane and two cones are
exact where a mesh is an approximation. It stopped being an option when the
ground had to change. `--terrain deforming` hands this layer a
`heightfield.GroundSurface` instead -- 80,000 triangles under a BVH of their
own, plus the analytic plane for everything outside the patch -- and the
terrain BVH is rebuilt only when the ground actually changes, once a second at
most, rather than per frame with the actors. Blender is given the same vertex
array, so the two sensors still agree about the ground; where they can differ
is a silhouette, and since ground, pile and sky all carry instance id 0, a
disagreement there is invisible in the instance masks by construction.

`embreex` is the ray engine, through trimesh. Open3D would have been the other
option and has no wheels for this interpreter; embree does the same job at
about 5 ms for a 14 400-ray sweep against the whole site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .geometry import Array, Box, IndexArray, Part
from .meshes import mesh as mesh_for


@dataclass(frozen=True)
class Hits:
    """Per ray: distance, which actor, and the surface normal there.

    `source` indexes the actor list the caster was built from, or -1 for
    terrain and for rays that hit nothing (which `t` distinguishes: a miss is
    infinite). It is per-point semantic truth and only ever reaches
    `/ground_truth/points`.
    """

    t: Array
    source: IndexArray
    normal: Array


class Raycaster(Protocol):
    def intersect(self, origin: Array, dirs: Array) -> Hits: ...


class Ground(Protocol):
    """Whatever is under the actors: `terrain.Terrain` or a committed patch."""

    def hits(self, origin: Array, dirs: Array) -> tuple[Array, Array]: ...


def _terrain_hits(ground: Ground, origin: Array, dirs: Array) -> Hits:
    t, normal = ground.hits(origin, dirs)
    return Hits(t=t, source=np.full(t.shape, -1, dtype=np.int32), normal=normal)


@dataclass
class BoxCaster:
    """Oriented cuboids, the geometry sitegen started with.

    Kept whole rather than deleted: `--actors boxes` is what makes the numbers
    in the README's detector table -- and every measurement taken before the
    meshes existed -- still reproducible from this tree.
    """

    boxes: list[Box]
    terrain: Ground

    def intersect(self, origin: Array, dirs: Array) -> Hits:
        hits = _terrain_hits(self.terrain, origin, dirs)
        t, source, normal = hits.t, hits.source, hits.normal
        for i, box in enumerate(self.boxes):
            t_box = intersect_box(origin, dirs, box)
            closer = t_box < t
            if not np.any(closer):
                continue
            t = np.where(closer, t_box, t)
            source = np.where(closer, i, source)
            normal = np.where(closer[:, None], _box_normal(origin, dirs, box), normal)
        return Hits(t=t, source=source.astype(np.int32), normal=normal)


def intersect_box(origin: Array, dirs: Array, box: Box) -> Array:
    """Slab-method ray-box distances; `inf` where the ray misses."""
    o = box.rotation.T @ (origin - box.center)
    d = dirs @ box.rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (-box.half_extents - o) * inv
        t2 = (box.half_extents - o) * inv
    t_near = np.max(np.minimum(t1, t2), axis=1)
    t_far = np.min(np.maximum(t1, t2), axis=1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near > 0.0)
    return np.where(hit, t_near, np.inf)


def _box_normal(origin: Array, dirs: Array, box: Box) -> Array:
    """The face a slab test entered through, in world axes."""
    o = box.rotation.T @ (origin - box.center)
    d = dirs @ box.rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        lo = np.minimum((-box.half_extents - o) * inv, (box.half_extents - o) * inv)
    axis = np.argmax(lo, axis=1)
    rows = np.arange(len(dirs))
    local = np.zeros_like(dirs)
    local[rows, axis] = -np.sign(d[rows, axis])
    return local @ box.rotation.T


class MeshCaster:
    """Every actor's triangles, posed into the world, under one BVH.

    The BVH is rebuilt per frame rather than kept per actor with rays
    transformed into each local frame. Rebuilding eleven thousand triangles
    costs about as much as one extra sweep and keeps occlusion between actors a
    single query, which is the property that makes the instance ids trustworthy
    where the boom passes in front of the truck.
    """

    def __init__(self, parts: list[Part], terrain: Ground) -> None:
        try:
            from trimesh import Trimesh
            from trimesh.ray.ray_pyembree import RayMeshIntersector
        except ImportError as exc:  # pragma: no cover -- a dependency, not a mode
            raise RuntimeError(
                "mesh actors need trimesh and embreex; `make install`"
            ) from exc

        self.terrain = terrain
        self.parts = parts
        vertices: list[Array] = []
        triangles: list[NDArray[np.int32]] = []
        owner: list[NDArray[np.int32]] = []
        offset = 0
        for i, part in enumerate(parts):
            m = mesh_for(part.mesh)
            vertices.append(m.vertices @ part.rotation.T + part.translation)
            triangles.append(m.triangles + offset)
            owner.append(np.full(len(m.triangles), i, dtype=np.int32))
            offset += len(m.vertices)

        self.vertices = np.vstack(vertices) if vertices else np.zeros((0, 3))
        self.triangles = (
            np.vstack(triangles).astype(np.int32) if triangles else np.zeros((0, 3), np.int32)
        )
        self.owner = (
            np.concatenate(owner).astype(np.int32) if owner else np.zeros(0, np.int32)
        )
        self._intersector = (
            RayMeshIntersector(Trimesh(vertices=self.vertices, faces=self.triangles, process=False))
            if len(self.triangles)
            else None
        )

    def intersect(self, origin: Array, dirs: Array) -> Hits:
        hits = _terrain_hits(self.terrain, origin, dirs)
        if self._intersector is None:
            return hits

        origins = np.broadcast_to(origin, dirs.shape).astype(np.float64)
        tri, ray, location = self._intersector.intersects_id(
            origins, dirs, multiple_hits=False, return_locations=True
        )
        if not len(ray):
            return hits

        distance = np.linalg.norm(location - origin, axis=1)
        closer = distance < hits.t[ray]
        ray, tri, distance = ray[closer], tri[closer], distance[closer]

        t = hits.t.copy()
        source = hits.source.copy()
        normal = hits.normal.copy()
        t[ray] = distance
        source[ray] = self.owner[tri]

        corners = self.vertices[self.triangles[tri]]
        face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
        # Winding is not guaranteed across procedural and baked meshes alike,
        # so face the normal back down the ray rather than trusting it.
        face *= -np.sign(np.sum(face * dirs[ray], axis=1))[:, None]
        normal[ray] = face
        return Hits(t=t, source=source, normal=normal)


def caster(parts: list[Part], boxes: list[Box], terrain: Ground, meshes: bool) -> Raycaster:
    return MeshCaster(parts, terrain) if meshes else BoxCaster(boxes, terrain)
