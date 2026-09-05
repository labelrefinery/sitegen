#!/usr/bin/env python3
"""Re-render the exported views in Cycles, with a real human where the box was.

The question this answers is narrow: sitegen draws a worker as a flat-shaded
orange cuboid and Grounding DINO never once finds it. Is that the *scene* being
hard, or the *renderer* being unconvincing? So everything here is held fixed
except the thing under test -- same intrinsics, same extrinsics, same worker
position and heading, same ground plane -- and the cuboid is replaced by a
rigged, textured human in a hi-vis vest and hard hat, lit by an outdoor HDRI
and shaded with physically based materials.

The truck stays a pair of boxes on purpose. It is the control: the baseline
detector already finds it, so if its score survives the change of renderer the
comparison is measuring the worker and not a change of everything at once.

The world, ground, camera and engine setup now come from `sitegen.cycles`,
which is where they ended up when the probe's answer turned into sitegen's
renderer. This script keeps only what is specific to the experiment: the boxes
it deliberately leaves as boxes, and the rigged GLB it poses on the fly rather
than using the baked meshes the package ships.

Run it against the Blender Python module, which is the same build as the app:

    uv run --isolated --no-project --python 3.13 --with bpy==5.2.1 --with numpy \
        python tools/probe/render.py --views <dir> --assets <dir> --out <dir>

`sitegen cameras` writes <dir>/views.json; see tools/probe/CREDITS for where
the HDRI, the ground texture and the human come from.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import bpy  # type: ignore[import-not-found]  # noqa: E402
import addon_utils  # type: ignore[import-not-found]  # noqa: E402
import mathutils  # type: ignore[import-not-found]  # noqa: E402
from mathutils import Vector  # type: ignore[import-not-found]  # noqa: E402

from sitegen.cycles import camera as place_camera, engine, ground, world_hdri  # noqa: E402

#: The stockpiles, from sitegen.scene: cones at the angle of repose. They are
#: 20 m out and read as low mounds, but leaving them out would change the
#: horizon the detector sees, and the point is to change one thing.
STOCKPILES = [(18.0, -11.0, 3.4), (-14.0, 14.0, 2.1)]

#: Metres, from actors.WORKER -- the asset is scaled to the box it replaces.
WORKER_HEIGHT_M = 1.75

#: The asset faces -y in Blender after the glTF import; sitegen yaw is measured
#: from +x. Rotating by yaw + 90 degrees points it the way the box points.
ASSET_FACING_OFFSET_DEG = 90.0

#: A spotter holds station and a crosser walks. The GLB ships both.
POSE_BY_INSTANCE = {"worker_0": "Idle", "worker_1": "Walk"}


def clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def box(center, size, yaw: float, color, roughness: float = 0.45) -> None:
    """One oriented cuboid -- what sitegen draws for the machines."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(center))
    obj = bpy.context.object
    obj.scale = tuple(size)
    obj.rotation_euler = (0.0, 0.0, yaw)
    material = bpy.data.materials.new("machine")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.35
    obj.data.materials.append(material)


def _set(bsdf, name: str, value) -> None:
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = value


def worker(glb: Path, center, yaw: float, ground_z: float, pose: str) -> None:
    """The asset under test: one rigged human in PPE, standing where the box was."""
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    added = [o for o in bpy.context.scene.objects if o not in before]

    # The pack ships a shadow-blob sphere with the character; it would sit in
    # mid-air here, because this scene has a real ground casting real shadows.
    for obj in list(added):
        if obj.type == "MESH" and obj.name.startswith("Icosphere"):
            bpy.data.objects.remove(obj, do_unlink=True)
            added.remove(obj)

    armature = next(o for o in added if o.type == "ARMATURE")
    root = next(o for o in added if o.parent is None)

    action = bpy.data.actions.get(f"CharacterArmature|{pose}")
    if action is not None:
        if armature.animation_data is None:
            armature.animation_data_create()
        armature.animation_data.action = action
        # One frame in, so the pose is a stride or a stance rather than rest.
        slots = getattr(armature.animation_data, "action_slot", None)
        if slots is None and action.slots:
            armature.animation_data.action_slot = action.slots[0]
        bpy.context.scene.frame_set(12)

    lo, hi = 1e9, -1e9
    for obj in added:
        if obj.type == "MESH":
            for corner in obj.bound_box:
                z = (obj.matrix_world @ Vector(corner)).z
                lo, hi = min(lo, z), max(hi, z)
    scale = WORKER_HEIGHT_M / max(hi - lo, 1e-6)

    root.scale = (scale, scale, scale)
    root.rotation_mode = "XYZ"
    root.rotation_euler = (0.0, 0.0, yaw + math.radians(ASSET_FACING_OFFSET_DEG))
    root.location = (center[0], center[1], ground_z - lo * scale)

    for material in bpy.data.materials:
        if not material.use_nodes or "Principled BSDF" not in material.node_tree.nodes:
            continue
        bsdf = material.node_tree.nodes["Principled BSDF"]
        _set(bsdf, "Metallic", 0.0)
        _set(bsdf, "Roughness", 0.6)
        if material.name.startswith("Worker_Vest"):
            # Hi-vis: saturated, slightly sheened, and brighter than anything
            # else on the site. That is the whole point of the garment.
            _set(bsdf, "Base Color", (0.95, 0.32, 0.02, 1.0))
            _set(bsdf, "Roughness", 0.45)
            _set(bsdf, "Sheen Weight", 0.3)
        elif material.name.startswith("Worker_Yellow"):
            _set(bsdf, "Base Color", (0.93, 0.72, 0.04, 1.0))
            _set(bsdf, "Roughness", 0.3)
        elif material.name.startswith("Skin"):
            _set(bsdf, "Roughness", 0.55)


def camera(view: dict) -> None:
    """The exported pinhole, through the package's converter."""
    k = view["K"]
    place_camera(
        bpy,
        mathutils,
        {
            "width": view["width"],
            "height": view["height"],
            "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
        },
        view["rotation"],
        view["translation"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", type=Path, required=True, help="dir with views.json")
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--hdri", default="construction_yard_2k.hdr")
    ap.add_argument("--human", default="worker_quaternius.glb")
    ap.add_argument("--only", help="render just this view name")
    ap.add_argument(
        "--worker-as-box",
        action="store_true",
        help="keep sitegen's orange cuboid for the worker. This is the "
        "ablation: same Cycles, same HDRI, same ground, so whatever it scores "
        "is what the renderer bought and the rest is what the human bought",
    )
    args = ap.parse_args()

    addon_utils.enable("io_scene_gltf2", default_set=True, persistent=True)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.views / "views.json").read_text())

    for view in manifest:
        if args.only and view["name"] != args.only:
            continue
        clear()
        engine(bpy, addon_utils, args.samples)
        world_hdri(bpy, args.assets / args.hdri)
        ground(bpy, args.assets, STOCKPILES)

        for actor in view["actors"]:
            cls = actor["class"].split(".")[0]
            if cls == "worker" and args.worker_as_box:
                # ALBEDO["worker"] from sitegen.camera, the colour the baseline
                # image drew.
                box(actor["center"], actor["size"], actor["yaw"], (0.92, 0.36, 0.16))
            elif cls == "worker":
                worker(
                    args.assets / args.human,
                    actor["center"],
                    actor["yaw"],
                    actor["ground_z"],
                    POSE_BY_INSTANCE.get(actor["instance"], "Idle"),
                )
            elif cls == "haul_truck":
                box(actor["center"], actor["size"], actor["yaw"], (0.42, 0.40, 0.38))

        camera(view)
        path = args.out / f"{view['name']}.png"
        bpy.context.scene.render.filepath = str(path)
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
