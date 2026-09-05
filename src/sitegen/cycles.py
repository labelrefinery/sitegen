"""The camera, rendered in Cycles from the meshes the LiDAR just intersected.

Both halves of the Blender bridge live here: the job description a normal
sitegen process writes, and the worker that reads it inside Blender. They are
one file because they are one contract, and a contract with two copies is a
contract that drifts.

Why a subprocess at all. `bpy` is a Python module, but a *versioned* one -- the
5.2.1 wheel is built for CPython 3.13 and the project runs on 3.14 -- so it
cannot simply be a dependency. uv makes the gap cheap to cross: an ephemeral
`--with bpy` environment, a JSON job and an NPZ of vertex arrays over the wall,
PNGs and instance masks back. The alternative, pinning the whole project to
whatever interpreter Blender ships this year, would put a renderer in charge of
the version every other module is written against.

What crosses the wall is the *same triangle arrays* the raycaster used, not a
description of how to rebuild them. That is the point of the whole exercise:
one mesh set, two sensors, and the only difference between them is shading.

Geometry conventions, which are where this sort of thing goes wrong:

  - sitegen's camera rotation maps optical axes (+x right, +y down, +z forward)
    into the world. Blender's camera is +x right, +y up, -z forward, so the
    matrix is post-multiplied by diag(1, -1, -1).
  - the lens comes from fx and a nominal 36 mm sensor; principal-point offset
    becomes shift_x/shift_y, both normalised by *width*, which is Blender's
    convention and not the obvious one.
  - `tests/test_same_surface.py` is what proves the above two paragraphs are
    right, by projecting LiDAR returns into the rendered instance masks.

Instance ids are a second render rather than the object-index render pass. The
plan was the pass; Blender 5.2's compositor is a rewritten, node-group-based
one whose Render Layers node exposes only Image and Alpha and whose File Output
node has no `file_slots`, and routing IndexOB out of it is neither stable nor
obvious. What replaces it needs no compositor at all: a black world, every
actor's material overridden by an emission shader carrying `id / 255`, one
sample, the pixel filter collapsed, straight to a linear EXR. Ground, piles and
sky emit nothing and read back as 0, which is the id they already had.

The one-sample, collapsed-filter part is not an optimisation. An object index
averaged across a pixel's samples is not an object index -- it is the mean of
two unrelated integers, and at every silhouette it would name a third actor
that is not in the scene. One sample at the pixel centre is also exactly where
`CameraIntrinsics.ray_directions` puts its ray, which is what lets the two
sensors be compared pixel for pixel.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Array

#: Same build the worker probe used, so its numbers stay comparable.
BPY_SPEC = "bpy==5.2.1"
BPY_PYTHON = "3.13"

#: A clean horizon on purpose. The probe used `construction_yard`, whose
#: backdrop contains real people and vehicles -- and those produced their own
#: sub-threshold detections above the skyline, which a pipeline consuming raw
#: boxes would have had to reject. A pure-sky HDRI lights the scene the same way
#: and puts nothing in it.
DEFAULT_HDRI = "kloofendal_43d_clear_puresky_2k.hdr"
GRAVEL = ("gravel_diff_2k.jpg", "gravel_rough_2k.jpg", "gravel_nor_gl_2k.jpg")

#: Blender's default sensor; the lens is derived from it and fx.
SENSOR_MM = 36.0

#: Angle of repose, from terrain.py. The cone is drawn with enough segments
#: that its silhouette sits close under the analytic cone the LiDAR uses.
REPOSE_DEG = 34.0
CONE_SEGMENTS = 128

#: Base colour, roughness, metallic per material name. Keys are matched as
#: "<mesh>:<material>" first and then bare, so an excavator's `body` can be
#: machine yellow while a truck's is hauler yellow.
PALETTE: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "excavator.house:body": ((0.72, 0.42, 0.03), 0.45, 0.3),
    "excavator.house:counterweight": ((0.68, 0.39, 0.03), 0.5, 0.3),
    "excavator.house:cab": ((0.70, 0.41, 0.03), 0.4, 0.3),
    "excavator.boom:body": ((0.72, 0.42, 0.03), 0.45, 0.3),
    "excavator.stick:body": ((0.72, 0.42, 0.03), 0.45, 0.3),
    "haul_truck.cab:body": ((0.76, 0.52, 0.04), 0.4, 0.3),
    "track": ((0.045, 0.045, 0.05), 0.75, 0.6),
    "steel": ((0.11, 0.11, 0.12), 0.5, 0.85),
    "chassis": ((0.07, 0.07, 0.08), 0.6, 0.7),
    "bed": ((0.20, 0.20, 0.21), 0.55, 0.8),
    "rubber": ((0.02, 0.02, 0.02), 0.85, 0.0),
    "glass": ((0.04, 0.06, 0.08), 0.08, 0.0),
    "timber": ((0.36, 0.24, 0.12), 0.85, 0.0),
    "marker": ((0.92, 0.30, 0.05), 0.6, 0.0),
    # The worker's slots, as the GLB named them. Hi-vis is saturated and
    # slightly sheened because that is the whole point of the garment, and
    # docs/PROBE-WORKER.md says the silhouette plus that colour is what an
    # open-vocabulary detector is actually responding to.
    "Worker_Vest": ((0.95, 0.32, 0.02), 0.45, 0.0),
    "Worker_Yellow": ((0.93, 0.72, 0.04), 0.30, 0.0),
    "Skin": ((0.72, 0.51, 0.38), 0.55, 0.0),
    "LightBrown": ((0.45, 0.30, 0.17), 0.7, 0.0),
    "Brown": ((0.30, 0.19, 0.10), 0.7, 0.0),
    "Brown2": ((0.24, 0.15, 0.08), 0.7, 0.0),
    "Grey": ((0.28, 0.29, 0.31), 0.7, 0.0),
    "Black": ((0.03, 0.03, 0.03), 0.6, 0.0),
    "Eye": ((0.02, 0.02, 0.02), 0.25, 0.0),
    "Eyebrows": ((0.05, 0.04, 0.03), 0.6, 0.0),
    "Moustache": ((0.09, 0.07, 0.05), 0.7, 0.0),
}
FALLBACK = ((0.55, 0.55, 0.55), 0.6, 0.0)


@dataclass
class Placement:
    """One actor link in one frame: which mesh, where, and its instance id."""

    mesh: str
    id: int
    rotation: list[list[float]]
    translation: list[float]


@dataclass
class Shot:
    """One rendered image: a camera pose and everything visible from it."""

    name: str
    rotation: list[list[float]]
    translation: list[float]
    objects: list[Placement] = field(default_factory=list)


def placements(parts: list, ids: list[int]) -> list[Placement]:
    return [
        Placement(
            mesh=part.mesh,
            id=instance,
            rotation=[[float(v) for v in row] for row in part.rotation],
            translation=[float(v) for v in part.translation],
        )
        for part, instance in zip(parts, ids)
    ]


def write_job(
    directory: Path,
    shots: list[Shot],
    *,
    intrinsics: Any,
    assets: Path,
    stockpiles: list[tuple[float, float, float]],
    samples: int = 48,
    hdri: str = DEFAULT_HDRI,
    ground_extent: float = 400.0,
) -> Path:
    """Serialise a render job and the meshes it refers to.

    The meshes go over as PLY, the same format the baked worker already ships
    in, so the thing Blender loads is a file anyone can open and check against
    what the LiDAR hit.
    """
    from .meshes import mesh as mesh_for, write_ply

    directory.mkdir(parents=True, exist_ok=True)
    used = sorted({p.mesh for shot in shots for p in shot.objects})
    for name in used:
        write_ply(mesh_for(name), directory / "meshes" / f"{name}.ply")

    job = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "samples": samples,
        "assets": str(assets),
        "hdri": hdri,
        "ground_extent": ground_extent,
        "stockpiles": [list(p) for p in stockpiles],
        "meshes": used,
        "shots": [
            {
                "name": shot.name,
                "rotation": shot.rotation,
                "translation": shot.translation,
                "objects": [vars(p) for p in shot.objects],
            }
            for shot in shots
        ],
    }
    path = directory / "job.json"
    path.write_text(json.dumps(job) + "\n")
    return path


def missing_assets(assets: Path, hdri: str = DEFAULT_HDRI) -> list[str]:
    return [name for name in (hdri, *GRAVEL) if not (assets / name).exists()]


#: How many times to restart a crashed Blender before giving up. Blender is
#: a large native program driven by a Python module for half an hour at a
#: stretch; it occasionally dies, and the work already on disk should not die
#: with it. Each attempt resumes at the first unrendered shot, so a restart
#: that makes no progress at all is a real failure and stops.
ATTEMPTS = 4


def run(job: Path, out: Path) -> None:
    """Render a job in a throwaway Blender interpreter, resuming if it falls over.

    Stdout is left attached: Cycles reports its own progress, and a
    forty-minute render that prints nothing is indistinguishable from a hung
    one.
    """
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "uv", "run", "--isolated", "--no-project",
        "--python", BPY_PYTHON,
        "--with", BPY_SPEC, "--with", "numpy",
        "python", "-m", "sitegen.cycles",
        str(job), str(out),
    ]
    wanted = len(json.loads(job.read_text())["shots"])
    for attempt in range(ATTEMPTS):
        done = len(list(out.glob("*.ids.npy")))
        result = subprocess.run(command, env={**os.environ, "PYTHONPATH": str(_src())})
        if result.returncode == 0:
            return
        progressed = len(list(out.glob("*.ids.npy")))
        print(
            f"Blender exited {result.returncode} after {progressed} of {wanted} "
            f"frames (attempt {attempt + 1}/{ATTEMPTS})",
            file=sys.stderr,
        )
        if progressed <= done:
            break
    raise RuntimeError(
        f"Blender render failed ({result.returncode}). Command:\n  "
        + " ".join(command)
    )


def _src() -> Path:
    """The import root the Blender interpreter is pointed at."""
    return Path(__file__).resolve().parents[1]


# -- everything below runs inside Blender ----------------------------------


def _material(bpy: Any, key: str, name: str) -> Any:
    """A Principled BSDF from the palette, created once per key."""
    existing = bpy.data.materials.get(key)
    if existing is not None:
        return existing
    colour, roughness, metallic = PALETTE.get(key, PALETTE.get(name, FALLBACK))
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Sheen Weight" in bsdf.inputs and name == "Worker_Vest":
        bsdf.inputs["Sheen Weight"].default_value = 0.3
    return material


def world_hdri(bpy: Any, path: Path, strength: float = 1.0, rotation_deg: float = 0.0) -> None:
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


def ground(
    bpy: Any,
    assets: Path,
    stockpiles: list[tuple[float, float, float]],
    size: float = 400.0,
    tile_m: float = 2.0,
) -> list[Any]:
    """sitegen's terrain: a plane at z = 0 and cones at the angle of repose.

    The same two primitives the LiDAR intersects analytically, so the ground
    the two sensors see is the same ground. Returns the objects, because the
    id pass has to silence them along with everything else.
    """
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

    def image(name: str, colorspace: str) -> Any:
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(assets / name))
        node.image.colorspace_settings.name = colorspace
        node.extension = "REPEAT"
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    diffuse, rough, normal_map = GRAVEL
    links.new(image(diffuse, "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
    links.new(image(rough, "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
    normal = nodes.new("ShaderNodeNormalMap")
    links.new(image(normal_map, "Non-Color").outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
    plane.data.materials.append(material)

    made = [plane]
    dirt = bpy.data.materials.new("pile")
    dirt.use_nodes = True
    pile_bsdf = dirt.node_tree.nodes["Principled BSDF"]
    pile_bsdf.inputs["Base Color"].default_value = (0.14, 0.10, 0.07, 1.0)
    pile_bsdf.inputs["Roughness"].default_value = 0.95
    for x, y, height in stockpiles:
        bpy.ops.mesh.primitive_cone_add(
            radius1=height / math.tan(math.radians(REPOSE_DEG)),
            depth=height,
            location=(x, y, height / 2.0),
            vertices=CONE_SEGMENTS,
        )
        bpy.context.object.data.materials.append(dirt)
        made.append(bpy.context.object)
    return made


def camera(
    bpy: Any,
    mathutils: Any,
    job: dict,
    rotation: Array | None = None,
    translation: Array | None = None,
) -> Any:
    """The exported pinhole, converted to Blender's -z-forward convention.

    Created once per run and re-posed per shot. Creating and removing a camera
    datablock per frame is what the obvious version does, and Blender segfaults
    on it a few dozen frames in -- `scene.camera` keeps pointing at what was
    just freed.
    """
    data = bpy.data.cameras.new("cam")
    obj = bpy.data.objects.new("cam", data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj

    width, height = job["width"], job["height"]
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = SENSOR_MM
    data.lens = job["fx"] * SENSOR_MM / width
    data.shift_x = (width / 2.0 - job["cx"]) / width
    data.shift_y = (job["cy"] - height / 2.0) / width
    assert abs(job["fx"] - job["fy"]) < 1e-6, "square pixels assumed"

    if rotation is not None and translation is not None:
        place(bpy, mathutils, obj, rotation, translation, optical=True)
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    return obj


def place(
    bpy: Any,
    mathutils: Any,
    obj: Any,
    rotation: Array,
    translation: Array,
    optical: bool = False,
) -> None:
    matrix = mathutils.Matrix([list(row) for row in rotation]).to_4x4()
    if optical:
        matrix = matrix @ mathutils.Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    matrix.translation = mathutils.Vector([float(v) for v in translation])
    obj.matrix_world = matrix


def _id_material(bpy: Any, instance: int) -> Any:
    """An emitter whose red channel *is* the instance id, in units of 1/255."""
    key = f"__instance_{instance}"
    existing = bpy.data.materials.get(key)
    if existing is not None:
        return existing
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (instance / 255.0, 0.0, 0.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _override(obj: Any, material: Any | None) -> None:
    """Point every slot at one material, or hand the slots back to the mesh.

    Object-level linking is what makes this possible at all: six grade stakes
    share one mesh datablock, so a per-mesh material could not give them six
    different ids.
    """
    for slot in obj.material_slots:
        if material is None:
            slot.link = "DATA"
        else:
            slot.link = "OBJECT"
            slot.material = material


def engine(bpy: Any, addon_utils: Any, samples: int) -> None:
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


def _worker(job_path: Path, out: Path) -> None:
    """Build the site once, then walk the shots. Runs inside Blender."""
    import bpy  # type: ignore[import-not-found]
    import addon_utils  # type: ignore[import-not-found]
    import mathutils  # type: ignore[import-not-found]

    from .meshes import read_ply

    job = json.loads(job_path.read_text())
    assets = Path(job["assets"])
    library = {
        name: read_ply(job_path.parent / "meshes" / f"{name}.ply")
        for name in job["meshes"]
    }

    bpy.ops.wm.read_factory_settings(use_empty=True)
    engine(bpy, addon_utils, job["samples"])
    world_hdri(bpy, assets / job["hdri"])
    terrain = ground(
        bpy, assets, [tuple(p) for p in job["stockpiles"]], size=job["ground_extent"]
    )

    scene = bpy.context.scene

    datablocks: dict[str, Any] = {}
    for name, source in library.items():
        data = bpy.data.meshes.new(name)
        data.from_pydata(
            source.vertices.tolist(), [], source.triangles.astype(int).tolist()
        )
        for material_name in source.materials:
            data.materials.append(_material(bpy, f"{name}:{material_name}", material_name))
        data.polygons.foreach_set("material_index", source.tri_material.tolist())
        data.update()
        datablocks[name] = data

    # One object per simultaneous use of a mesh. Six stakes share one
    # datablock; the pool grows to six objects and stops.
    pool: dict[str, list[Any]] = {name: [] for name in datablocks}

    lit = scene.world
    dark = bpy.data.worlds.new("id_pass")
    dark.use_nodes = True
    dark.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    out.mkdir(parents=True, exist_ok=True)

    cam = camera(bpy, mathutils, job)
    for index, shot in enumerate(job["shots"], start=1):
        if (out / f"{shot['name']}.ids.npy").exists() and (
            out / f"{shot['name']}.png"
        ).exists():
            continue
        counts: dict[str, int] = {}
        for item in shot["objects"]:
            name = item["mesh"]
            slot = counts.get(name, 0)
            counts[name] = slot + 1
            if slot >= len(pool[name]):
                obj = bpy.data.objects.new(f"{name}.{slot}", datablocks[name])
                scene.collection.objects.link(obj)
                pool[name].append(obj)
            obj = pool[name][slot]
            obj.hide_render = False
            obj.pass_index = item["id"]
            place(bpy, mathutils, obj, item["rotation"], item["translation"])
        for name, objects in pool.items():
            for obj in objects[counts.get(name, 0) :]:
                obj.hide_render = True

        place(bpy, mathutils, cam, shot["rotation"], shot["translation"], optical=True)
        scene.frame_current = index

        scene.world = lit
        settings = scene.render.image_settings
        settings.file_format = "PNG"
        settings.color_mode = "RGB"
        settings.color_management = "FOLLOW_SCENE"
        scene.cycles.samples = job["samples"]
        scene.cycles.max_bounces = 8
        scene.render.filter_size = 1.5
        scene.cycles.use_denoising = True
        scene.render.filepath = str(out / f"{shot['name']}.png")
        bpy.ops.render.render(write_still=True)

        # Every visible surface becomes a pure emitter, terrain included.
        # Leave the gravel shading in and the actors' own emission bounces off
        # it: the ground comes back at a few thousandths, which rounds to
        # instance ids 1 to 3 -- actors that are nowhere near it.
        scene.world = dark
        for obj in terrain:
            _override(obj, _id_material(bpy, 0))
        for objects in pool.values():
            for obj in objects:
                if not obj.hide_render:
                    _override(obj, _id_material(bpy, obj.pass_index))
        settings.file_format = "OPEN_EXR"
        settings.color_depth = "32"
        settings.color_management = "OVERRIDE"
        settings.view_settings.view_transform = "Raw"
        scene.cycles.samples = 1
        scene.cycles.max_bounces = 0
        scene.cycles.use_denoising = False
        scene.render.filter_size = 0.01
        exr = out / f"{shot['name']}.ids.exr"
        scene.render.filepath = str(exr)
        bpy.ops.render.render(write_still=True)
        _decode_ids(bpy, exr, out / f"{shot['name']}.ids.npy", job)
        for obj in (*terrain, *(o for objects in pool.values() for o in objects)):
            _override(obj, None)
        print(f"rendered {shot['name']}", flush=True)


def _decode_ids(bpy: Any, path: Path, destination: Path, job: dict) -> None:
    """Read the emission render back and round it to integers.

    Written to a linear EXR with the view transform forced to Raw, so the red
    channel comes back as exactly `id / 255` and the rounding is decoration
    rather than repair.
    """
    image = bpy.data.images.load(str(path))
    image.colorspace_settings.name = "Non-Color"
    buffer = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(buffer)
    height, width = job["height"], job["width"]
    # Blender hands back rows bottom-up.
    ids = buffer.reshape(height, width, 4)[::-1, :, 0] * 255.0
    np.save(destination, np.rint(ids).astype(np.uint16))
    bpy.data.images.remove(image)
    path.unlink()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
