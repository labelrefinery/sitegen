#!/usr/bin/env python3
"""Bake the rigged CC0 worker into the two static poses sitegen ships.

The other actor meshes are built from primitives at import time, which is fine
for a machine and hopeless for a person -- docs/PROBE-WORKER.md is the
measurement that says a detector wants a person-shaped silhouette and will not
take a box. So the worker is a real asset, and this is where it stops being a
rig and becomes triangles.

Baking rather than shipping the GLB buys three things. The LiDAR gets a mesh it
can intersect without a skinning implementation; Cycles gets exactly the same
triangles the LiDAR saw, which is the whole same-surface property; and the
package stays self-contained, with no glTF importer in the runtime dependency
set. The cost is that the pose is frozen at bake time, which is why there are
two: the spotter stands and the crosser walks.

Needs Blender's Python module, which does not exist for the project's
interpreter:

    uv run --isolated --no-project --python 3.13 --with bpy==5.2.1 \
        python tools/bake_worker.py --glb <worker_quaternius.glb>

Writes src/sitegen/assets/worker_stand.ply and worker_walk.ply. Provenance and
licence go in src/sitegen/assets/CREDITS.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bpy  # type: ignore[import-not-found]  # noqa: E402
import addon_utils  # type: ignore[import-not-found]  # noqa: E402
from mathutils import Vector  # type: ignore[import-not-found]  # noqa: E402

from sitegen.meshes import Mesh, write_ply  # noqa: E402

#: From actors.WORKER -- the asset is scaled to the height it stands in for.
HEIGHT_M = 1.75

#: The asset faces -y after the glTF import; sitegen measures yaw from +x.
FACING_OFFSET_DEG = 90.0

#: Action name in the GLB -> the pose file it becomes. One holds station and
#: one is walking, which is what scene.workers_at() says the two of them do.
POSES = {"Idle": "worker_stand", "Walk": "worker_walk"}

#: Frame to sample. Rest pose is a T-pose-ish stance that reads as neither.
FRAME = 12


def load(glb: Path, action_name: str) -> list:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_gltf2", default_set=True, persistent=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    added = list(bpy.context.scene.objects)

    # The pack ships a shadow-blob sphere with the character. In sitegen it
    # would be a floating ball the LiDAR returns points from.
    for obj in list(added):
        if obj.type == "MESH" and obj.name.startswith("Icosphere"):
            bpy.data.objects.remove(obj, do_unlink=True)
            added.remove(obj)

    armature = next(o for o in added if o.type == "ARMATURE")
    action = bpy.data.actions.get(f"CharacterArmature|{action_name}")
    if action is None:
        raise SystemExit(f"no action {action_name!r} in {glb}")
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    if getattr(armature.animation_data, "action_slot", None) is None and action.slots:
        armature.animation_data.action_slot = action.slots[0]
    bpy.context.scene.frame_set(FRAME)
    return added


def triangles(objects: list) -> Mesh:
    """Evaluated, world-space triangles, grouped by material name."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    tri_material: list[np.ndarray] = []
    materials: list[str] = []

    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        data = evaluated.to_mesh()
        data.calc_loop_triangles()
        matrix = evaluated.matrix_world

        offset = sum(len(v) for v in verts)
        verts.append(
            np.array([tuple(matrix @ v.co) for v in data.vertices], dtype=np.float64)
        )
        slots = [
            (slot.material.name if slot.material else "worker") for slot in obj.material_slots
        ] or ["worker"]
        for name in slots:
            if name not in materials:
                materials.append(name)
        index = [materials.index(name) for name in slots]

        tris.append(
            np.array([tuple(t.vertices) for t in data.loop_triangles], dtype=np.int32)
            + offset
        )
        tri_material.append(
            np.array(
                [index[min(t.material_index, len(index) - 1)] for t in data.loop_triangles],
                dtype=np.int32,
            )
        )
        evaluated.to_mesh_clear()

    return Mesh(
        vertices=np.vstack(verts),
        triangles=np.vstack(tris),
        tri_material=np.concatenate(tri_material),
        materials=tuple(materials),
    )


def normalise(mesh: Mesh) -> Mesh:
    """Stand the figure on z = 0 at the origin, HEIGHT_M tall, facing +x."""
    a = math.radians(FACING_OFFSET_DEG)
    rotation = np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )
    v = mesh.vertices @ rotation.T
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = v * (HEIGHT_M / (hi[2] - lo[2]))
    lo, hi = v.min(axis=0), v.max(axis=0)
    v -= np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
    return Mesh(v, mesh.triangles, mesh.tri_material, mesh.materials)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", type=Path, required=True, help="worker_quaternius.glb")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "sitegen" / "assets",
    )
    args = ap.parse_args()

    for action, stem in POSES.items():
        mesh = normalise(triangles(load(args.glb, action)))
        path = args.out / f"{stem}.ply"
        write_ply(mesh, path)
        lo, hi = mesh.bounds()
        print(
            f"{path.name}: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris, "
            f"materials {list(mesh.materials)}, "
            f"extent {np.round(hi - lo, 3).tolist()} m, {path.stat().st_size} B"
        )


if __name__ == "__main__":
    main()
