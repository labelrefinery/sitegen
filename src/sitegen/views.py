"""Pull camera views out of a scene, with the geometry needed to re-render them.

This exists for one experiment: sitegen's own renderer draws actors as
flat-shaded boxes, and an open-vocabulary detector pointed at the result names
the truck and never once finds a worker. To ask whether that is the *renderer's*
fault rather than the scene's, the same views have to be rendered again by
something else -- which means exporting the exact camera the scene used, not an
approximation of it.

Everything here is read back out of the MCAP rather than recomputed from
`Scene`. The recording is the artefact under test; if the export drifted from
it the comparison would be measuring the drift.

Two facts make the instance masks readable. `/ground_truth/actors` writes one
entity per box in the order the renderer walked them, and the mask stores
`index + 1` with 0 for terrain -- so mask id `i` is entity `i - 1` at the same
timestamp. That is what turns a pixel count into "this many pixels of worker_1".
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from mcap.reader import make_reader

from .geometry import Array

#: Below this a "visible" worker is a handful of edge pixels -- too small to
#: ask a detector about, and not what the probe is trying to measure.
MIN_WORKER_PIXELS = 60

#: Two views of the same camera closer than this in time are the same view.
SPACING_S = 2.0


def _matrix_from_quat(x: float, y: float, z: float, w: float) -> Array:
    n = np.hypot(np.hypot(x, y), np.hypot(z, w))
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


@dataclass
class Actor:
    """One ground-truth box, in the order the renderer numbered it."""

    instance: str
    cls: str
    center: Array
    size: Array
    yaw: float


@dataclass
class View:
    """One camera frame, plus everything needed to re-render it elsewhere."""

    camera: str
    t: float
    ns: int
    image: bytes
    mask: np.ndarray
    k: list[float]
    width: int
    height: int
    rotation: Array
    translation: Array
    actors: list[Actor]
    pixels: dict[int, int] = field(default_factory=dict)
    """Mask id -> pixel count, for ids that appear at all."""

    def visible(self, cls: str) -> list[tuple[Actor, int]]:
        """Parts of a class that own pixels here, most visible first."""
        return sorted(
            (
                (self.actors[i - 1], n)
                for i, n in self.pixels.items()
                if 0 < i <= len(self.actors)
                and self.actors[i - 1].cls.split(".")[0] == cls
            ),
            key=lambda p: -p[1],
        )

    def bbox(self, cls: str, instance: str) -> list[int]:
        """Pixel bounds of an actor in this frame, from the instance mask.

        The re-render puts the same actor at the same place, so this is what
        says whether a detection landed on the worker or somewhere else.
        """
        ids = [
            i
            for i, a in enumerate(self.actors, start=1)
            if a.instance == instance and a.cls.split(".")[0] == cls
        ]
        rows, cols = np.where(np.isin(self.mask, ids))
        if not len(rows):
            return []
        return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]

    def instances(self, cls: str) -> list[tuple[str, int]]:
        """The same, summed per actor -- a truck is a cab box plus a bed box."""
        totals: dict[str, int] = {}
        for actor, n in self.visible(cls):
            totals[actor.instance] = totals.get(actor.instance, 0) + n
        return sorted(totals.items(), key=lambda p: -p[1])

    def range_to(self, actor: Actor) -> float:
        return float(np.linalg.norm(actor.center - self.translation))


def read_views(path: Path) -> list[View]:
    """Every camera frame in the recording, paired with its mask and pose."""
    images: dict[tuple[str, int], bytes] = {}
    masks: dict[tuple[str, int], np.ndarray] = {}
    poses: dict[tuple[str, int], tuple[Array, Array]] = {}
    calib: dict[str, tuple[list[float], int, int]] = {}
    actors: dict[int, list[Actor]] = {}
    base_ns: int | None = None

    with open(path, "rb") as f:
        for schema, channel, msg in make_reader(f).iter_messages():
            topic = channel.topic
            if base_ns is None:
                base_ns = msg.log_time
            if topic.startswith("/camera/") and topic.endswith("/image"):
                image = CompressedImage()
                image.ParseFromString(msg.data)
                images[(topic.split("/")[2], msg.log_time)] = image.data
            elif topic.startswith("/ground_truth/camera_instances/"):
                image = CompressedImage()
                image.ParseFromString(msg.data)
                masks[(topic.rsplit("/", 1)[1], msg.log_time)] = np.array(
                    Image.open(io.BytesIO(image.data))
                )
            elif topic.startswith("/camera/") and topic.endswith("/calibration"):
                c = CameraCalibration()
                c.ParseFromString(msg.data)
                calib.setdefault(topic.split("/")[2], (list(c.K), c.width, c.height))
            elif topic == "/tf":
                tf = FrameTransform()
                tf.ParseFromString(msg.data)
                if tf.child_frame_id.startswith("camera_"):
                    q, p = tf.rotation, tf.translation
                    poses[(tf.child_frame_id[7:], msg.log_time)] = (
                        _matrix_from_quat(q.x, q.y, q.z, q.w),
                        np.array([p.x, p.y, p.z]),
                    )
            elif topic == "/ground_truth/actors":
                update = SceneUpdate()
                update.ParseFromString(msg.data)
                actors[msg.log_time] = [
                    Actor(
                        instance=e.id.partition("/")[0],
                        cls=e.id.partition("/")[2],
                        center=np.array([c.pose.position.x, c.pose.position.y, c.pose.position.z]),
                        size=np.array([c.size.x, c.size.y, c.size.z]),
                        yaw=_yaw_from_quat(
                            c.pose.orientation.x, c.pose.orientation.y,
                            c.pose.orientation.z, c.pose.orientation.w,
                        ),
                    )
                    for e in update.entities
                    for c in e.cubes
                ]

    assert base_ns is not None, f"{path} has no messages"
    views: list[View] = []
    for (camera, ns), data in sorted(images.items()):
        mask = masks[(camera, ns)]
        ids, counts = np.unique(mask, return_counts=True)
        k, width, height = calib[camera]
        rotation, translation = poses[(camera, ns)]
        views.append(
            View(
                camera=camera,
                t=(ns - base_ns) / 1e9,
                ns=ns,
                image=data,
                mask=mask,
                k=k,
                width=width,
                height=height,
                rotation=rotation,
                translation=translation,
                actors=actors[ns],
                pixels={int(i): int(n) for i, n in zip(ids, counts) if i != 0},
            )
        )
    return views


def select(views: list[View], count: int = 10) -> list[View]:
    """Every distinct view with a visible worker, then the clearest trucks.

    The worker views are the point of the probe, so they go in first; the rest
    are filled with the best views of the truck, which is the class the
    baseline detector does find and therefore the control that says a new
    render did not simply break everything.

    "Distinct" is doing real work here. The crossing worker is in frame for
    several seconds at 2 Hz, so the naive top-ten is ten near-identical frames
    of one pose, which measures the same view ten times.
    """

    def pixels_of(cls: str):
        return lambda v: sum(n for _, n in v.instances(cls))

    def greedy(candidates: list[View], score, into: list[View]) -> None:
        for v in sorted(candidates, key=lambda v: -score(v)):
            if len(into) >= count:
                return
            if score(v) <= 0:
                return
            if all(o.camera != v.camera or abs(o.t - v.t) >= SPACING_S for o in into):
                into.append(v)

    chosen: list[View] = []
    worker, truck = pixels_of("worker"), pixels_of("haul_truck")
    greedy([v for v in views if worker(v) >= MIN_WORKER_PIXELS], worker, chosen)
    greedy(views, truck, chosen)
    return sorted(chosen, key=lambda v: (v.t, v.camera))


def write_views(views: list[View], out: Path) -> Path:
    """Images plus one JSON manifest of intrinsics, extrinsics and actors."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for v in views:
        name = f"{v.camera}_t{v.t:05.1f}".replace(".", "p")
        (out / f"{name}.jpg").write_bytes(v.image)
        entry = {
            "name": name,
            "camera": v.camera,
            "t": round(v.t, 3),
            "image": f"{name}.jpg",
            "width": v.width,
            "height": v.height,
            "K": v.k,
            # Optical convention, as camera_pose() builds it: +z forward,
            # +x right, +y down. A consumer that assumes -z forward (Blender,
            # OpenGL) has to flip y and z itself.
            "rotation": [list(row) for row in v.rotation],
            "translation": list(v.translation),
            "actors": [],
        }
        for cls in ("worker", "haul_truck", "excavator"):
            for actor, pixels in v.visible(cls):
                entry["actors"].append(
                    {
                        "instance": actor.instance,
                        "class": actor.cls,
                        "pixels": pixels,
                        "range_m": round(v.range_to(actor), 2),
                        "center": [round(float(c), 3) for c in actor.center],
                        "size": [round(float(e), 3) for e in actor.size],
                        "yaw": round(actor.yaw, 4),
                        # The scene's ground is a plane at z = 0 and the piles
                        # are cones; a box sits on whatever is under it, so the
                        # ground height is the box's own underside.
                        "ground_z": round(
                            float(actor.center[2] - actor.size[2] / 2.0), 3
                        ),
                    }
                )
        entry["instances"] = [
            {
                "instance": instance,
                "class": cls,
                "pixels": pixels,
                "bbox": v.bbox(cls, instance),
            }
            for cls in ("worker", "haul_truck", "excavator")
            for instance, pixels in v.instances(cls)
        ]
        manifest.append(entry)

    path = out / "views.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
