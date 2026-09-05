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

Run it against the Blender Python module, which is the same build as the app:

    uv run --python 3.13 --with bpy==5.2.1 \
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

import bpy  # type: ignore[import-not-found]
import addon_utils  # type: ignore[import-not-found]
from mathutils import Matrix, Vector  # type: ignore[import-not-found]

#: Blender's default 36 mm sensor; the lens is derived from it and fx.
SENSOR_MM = 36.0

#: The stockpiles, from sitegen.scene: cones at the angle of repose. They are
#: 20 m out and read as low mounds, but leaving them out would change the
#: horizon the detector sees, and the point is to change one thing.
STOCKPILES = [(18.0, -11.0, 3.4), (-14.0, 14.0, 2.1)]
REPOSE_DEG = 34.0

#: Metres, from actors.WORKER -- the asset is scaled to the box it replaces.
WORKER_HEIGHT_M = 1.75

#: The asset faces -y in Blender after the glTF import; sitegen yaw is measured
#: from +x. Rotating by yaw + 90 degrees points it the way the box points.
ASSET_FACING_OFFSET_DEG = 90.0

#: A spotter holds station and a crosser walks. The GLB ships both.
POSE_BY_INSTANCE = {"worker_0": "Idle", "worker_1": "Walk"}


def clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def world_hdri(path: Path, strength: float = 1.0, rotation_deg: float = 0.0) -> None:
    """An outdoor environment, which is both the key light and the sky."""
    world = bpy.data.worlds.new("site")
    world.use_nodes = True
    bpy.context.scene.world = world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(path))
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_deg)
    coord = nodes.new("ShaderNodeTexCoord")
    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def ground(assets: Path, size: float = 400.0, tile_m: float = 2.0) -> None:
    """A plane at z = 0 -- sitegen's ground -- with a gravel PBR material."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object

    material = bpy.data.materials.new("gravel")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes["Principled BSDF"]

    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    # Object coordinates are metres here, because the plane is unscaled.
    mapping.inputs["Scale"].default_value = (1 / tile_m,) * 3
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def image(name: str, colorspace: str):
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(assets / name))
        node.image.colorspace_settings.name = colorspace
        node.extension = "REPEAT"
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    links.new(image("gravel_diff_2k.jpg", "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
    links.new(image("gravel_rough_2k.jpg", "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
    normal = nodes.new("ShaderNodeNormalMap")
    links.new(image("gravel_nor_gl_2k.jpg", "Non-Color").outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
    plane.data.materials.append(material)

    dirt = bpy.data.materials.new("pile")
    dirt.use_nodes = True
    dirt.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.14, 0.10, 0.07, 1.0
    )
    dirt.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
    for x, y, height in STOCKPILES:
        bpy.ops.mesh.primitive_cone_add(
            radius1=height / math.tan(math.radians(REPOSE_DEG)),
            depth=height,
            location=(x, y, height / 2.0),
            vertices=48,
        )
        bpy.context.object.data.materials.append(dirt)


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
    """The exported pinhole, converted to Blender's -z-forward convention."""
    data = bpy.data.cameras.new("cam")
    obj = bpy.data.objects.new("cam", data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj

    fx, cx = view["K"][0], view["K"][2]
    fy, cy = view["K"][4], view["K"][5]
    width, height = view["width"], view["height"]
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = SENSOR_MM
    data.lens = fx * SENSOR_MM / width
    data.shift_x = (width / 2.0 - cx) / width
    data.shift_y = (cy - height / 2.0) / width
    assert abs(fx - fy) < 1e-6, "square pixels assumed; the exporter writes them"

    # sitegen's rotation maps optical (+x right, +y down, +z forward) to world;
    # Blender's camera is +x right, +y up, -z forward.
    r = Matrix(view["rotation"]).to_4x4() @ Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    r.translation = Vector(view["translation"])
    obj.matrix_world = r

    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100


def engine(samples: int) -> None:
    addon_utils.enable("cycles", default_set=True, persistent=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:  # Metal where it exists, CPU where it does not.
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
        scene.cycles.device = "GPU"
    except Exception as exc:  # noqa: BLE001 -- any failure just means CPU
        print(f"  (no GPU: {exc}; rendering on CPU)", file=sys.stderr)


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
        engine(args.samples)
        world_hdri(args.assets / args.hdri)
        ground(args.assets)

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
