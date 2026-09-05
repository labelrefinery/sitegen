"""Export an MCAP scene into the formats the Mojo pipeline stages already read.

The tracker CSV schema is not invented here. `OfflinePoly` reads

    track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf

in global coordinates with `l` along heading `theta`, and `CenterPillars --csv`
writes exactly that. Ground truth is emitted in the same schema so a scorer can
compare like with like, and so a perfect labeler would be byte-comparable to
the oracle.

Sweeps go out as raw float32 rather than CSV: 600 frames of ~8k points is 5M
rows, and a Mojo stage should be reading 16 bytes per point off disk, not
parsing decimal.
"""

from __future__ import annotations

import ast
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.JointStates_pb2 import JointStates
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from mcap.reader import make_reader

from .geometry import Array

CSV_HEADER = [
    "track_id", "cls", "t", "x", "y", "z",
    "w", "l", "h", "vx", "vy", "theta", "conf",
]


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


@dataclass
class TruthBox:
    t: float
    instance: str
    cls: str
    center: Array
    size: Array
    yaw: float


def read_truth(path: Path) -> list[TruthBox]:
    """Decode /ground_truth/actors into flat per-part boxes."""
    out: list[TruthBox] = []
    base_ns: int | None = None
    with open(path, "rb") as f:
        for _, _, msg in make_reader(f).iter_messages(topics=["/ground_truth/actors"]):
            if base_ns is None:
                base_ns = msg.log_time
            t = (msg.log_time - base_ns) / 1e9
            update = SceneUpdate()
            update.ParseFromString(msg.data)
            for entity in update.entities:
                instance, _, cls = entity.id.partition("/")
                for cube in entity.cubes:
                    p, q = cube.pose.position, cube.pose.orientation
                    out.append(
                        TruthBox(
                            t=t,
                            instance=instance,
                            cls=cls,
                            center=np.array([p.x, p.y, p.z]),
                            size=np.array([cube.size.x, cube.size.y, cube.size.z]),
                            yaw=_yaw_from_quat(q.x, q.y, q.z, q.w),
                        )
                    )
    return out


def _corners(center: Array, size: Array, yaw: float) -> Array:
    hx, hy, hz = size / 2.0
    local = np.array(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ r.T + center


def merge_to_objects(boxes: list[TruthBox]) -> list[TruthBox]:
    """Collapse per-part boxes into one enclosing box per instance per frame.

    A cold-start pipeline clusters points; it does not know that a boom and a
    house are one machine. Scoring part-level truth against object-level
    predictions would punish it for a distinction it was never given, so the
    default comparison happens at object level. The parts stay available for
    when a pipeline is good enough to be asked about the bucket specifically.
    """
    grouped: dict[tuple[float, str], list[TruthBox]] = defaultdict(list)
    for b in boxes:
        grouped[(b.t, b.instance)].append(b)

    merged: list[TruthBox] = []
    for (t, instance), parts in grouped.items():
        # Body yaw: the largest part is the body, and its heading is the one a
        # human would draw the box along.
        anchor = max(parts, key=lambda p: float(np.prod(p.size)))
        yaw = anchor.yaw
        c, s = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        pts = np.concatenate([_corners(p.center, p.size, p.yaw) for p in parts])
        local = pts @ rot.T
        lo, hi = local.min(axis=0), local.max(axis=0)
        size = hi - lo
        centre_local = (hi + lo) / 2.0
        back = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        )
        merged.append(
            TruthBox(
                t=t,
                instance=instance,
                cls=anchor.cls.split(".")[0],
                center=back @ centre_local,
                size=size,
                yaw=yaw,
            )
        )
    return merged


def write_tracks_csv(boxes: list[TruthBox], out: Path) -> int:
    """Write boxes in the OfflinePoly schema, with velocity by finite difference."""
    by_instance: dict[str, list[TruthBox]] = defaultdict(list)
    for b in boxes:
        by_instance[b.instance].append(b)

    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for instance, seq in sorted(by_instance.items()):
            seq.sort(key=lambda b: b.t)
            for i, b in enumerate(seq):
                # Central difference where possible; one-sided at the ends.
                lo = seq[max(i - 1, 0)]
                hi = seq[min(i + 1, len(seq) - 1)]
                dt = hi.t - lo.t
                vx, vy = (
                    ((hi.center[0] - lo.center[0]) / dt, (hi.center[1] - lo.center[1]) / dt)
                    if dt > 0
                    else (0.0, 0.0)
                )
                writer.writerow(
                    [
                        instance, b.cls, f"{b.t:.4f}",
                        f"{b.center[0]:.4f}", f"{b.center[1]:.4f}", f"{b.center[2]:.4f}",
                        f"{b.size[1]:.4f}", f"{b.size[0]:.4f}", f"{b.size[2]:.4f}",
                        f"{vx:.4f}", f"{vy:.4f}", f"{b.yaw:.5f}", "1.0",
                    ]
                )
                rows += 1
    return rows


def write_ego_csv(path: Path, out: Path) -> int:
    """`t,x,y` from /gnss, which is what OfflinePoly's --ego expects."""
    base_ns: int | None = None
    lat0 = lon0 = None
    rows = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f, open(out, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerow(["t", "x", "y"])
        for _, _, msg in make_reader(f).iter_messages(topics=["/gnss"]):
            fix = LocationFix()
            fix.ParseFromString(msg.data)
            if base_ns is None:
                base_ns, lat0, lon0 = msg.log_time, fix.latitude, fix.longitude
            assert lat0 is not None and lon0 is not None
            # Local tangent plane; the scene is 120 m across so a flat-earth
            # approximation is exact to well under a millimetre.
            x = (fix.longitude - lon0) * 111_320.0 * float(np.cos(np.radians(lat0)))
            y = (fix.latitude - lat0) * 110_540.0
            writer.writerow([f"{(msg.log_time - base_ns) / 1e9:.4f}", f"{x:.4f}", f"{y:.4f}"])
            rows += 1
    return rows


def write_tf_csv(path: Path, out: Path) -> int:
    """`t,x,y,z,qx,qy,qz,qw` from /tf -- the sensor pose in the map frame.

    Observable, so the pipeline is allowed it. Points arrive in the sensor
    frame and every downstream stage reasons in world coordinates, so this is
    the transform that makes the rest possible.
    """
    base_ns: int | None = None
    rows = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f, open(out, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for _, _, msg in make_reader(f).iter_messages(topics=["/tf"]):
            tf = FrameTransform()
            tf.ParseFromString(msg.data)
            # The camera rig publishes map -> camera_<name> on the same topic.
            # Without this filter a recording made with --camera-hz exports
            # 840 rows for 600 sweeps, every stage pairs sweep i with a camera
            # pose, and round 0 scores F1 0.14 instead of 0.53.
            if tf.child_frame_id != "lidar":
                continue
            if base_ns is None:
                base_ns = msg.log_time
            tr, q = tf.translation, tf.rotation
            writer.writerow(
                [f"{(msg.log_time - base_ns) / 1e9:.4f}",
                 f"{tr.x:.6f}", f"{tr.y:.6f}", f"{tr.z:.6f}",
                 f"{q.x:.8f}", f"{q.y:.8f}", f"{q.z:.8f}", f"{q.w:.8f}"]
            )
            rows += 1
    return rows


def write_joints_csv(path: Path, out: Path) -> int:
    """`t,swing,boom,stick,bucket` from /ego/joint_states.

    This is the free supervision. A pipeline with no labels at all still knows
    exactly where its own machine's every link is, which is enough to mask the
    ego's self-returns out of the cloud before anything else runs.
    """
    base_ns: int | None = None
    names = ["swing", "boom", "stick", "bucket"]
    rows = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f, open(out, "w", newline="") as fo:
        writer = csv.writer(fo)
        writer.writerow(["t", *names])
        for _, _, msg in make_reader(f).iter_messages(topics=["/ego/joint_states"]):
            if base_ns is None:
                base_ns = msg.log_time
            js = JointStates()
            js.ParseFromString(msg.data)
            by_name = {j.name: j.position for j in js.joints}
            writer.writerow(
                [f"{(msg.log_time - base_ns) / 1e9:.4f}", *[f"{by_name.get(n, 0.0):.6f}" for n in names]]
            )
            rows += 1
    return rows


def write_volumes_csv(path: Path, out: Path) -> int:
    """`t` and one column per channel of /ground_truth/volumes.

    Held out, like the topic. This is the timeline a cut/fill or
    progress-monitoring pipeline is scored against: it says how much material
    left the ground, how much is standing in each stockpile, and how much has
    already been driven off the site -- the last of which no survey of the
    site can recover, which is exactly why it is worth publishing.

    The columns are whatever the recording contains, in the order it wrote
    them, so adding a stockpile to the scene adds a column here without
    touching this function.
    """
    base_ns: int | None = None
    rows = 0
    names: list[str] | None = None
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f, open(out, "w", newline="") as fo:
        writer = csv.writer(fo)
        for _, _, msg in make_reader(f).iter_messages(
            topics=["/ground_truth/volumes"]
        ):
            if base_ns is None:
                base_ns = msg.log_time
            record = JointStates()
            record.ParseFromString(msg.data)
            if names is None:
                names = [j.name for j in record.joints]
                writer.writerow(["t", *names])
            writer.writerow(
                [
                    f"{(msg.log_time - base_ns) / 1e9:.4f}",
                    *[f"{j.position:.6f}" for j in record.joints],
                ]
            )
            rows += 1
    return rows


def write_sweeps(path: Path, out_dir: Path) -> int:
    """Raw float32 x,y,z,intensity per frame, plus an index the Mojo side reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ns: int | None = None
    count = 0
    with open(path, "rb") as f, open(out_dir / "index.csv", "w", newline="") as fi:
        index = csv.writer(fi)
        index.writerow(["frame", "t", "n_points", "file"])
        for _, _, msg in make_reader(f).iter_messages(topics=["/lidar/points"]):
            if base_ns is None:
                base_ns = msg.log_time
            cloud = PointCloud()
            cloud.ParseFromString(msg.data)
            name = f"{count:06d}.bin"
            (out_dir / name).write_bytes(cloud.data)
            index.writerow(
                [count, f"{(msg.log_time - base_ns) / 1e9:.4f}", len(cloud.data) // 16, name]
            )
            count += 1
    return count


def unwrap_daft_payload(value: str | bytes) -> bytes:
    """Recover raw message bytes from a `daft.read_mcap` `data` cell.

    Daft hands back the Python `repr` of the bytes in a String column rather
    than the bytes. It round-trips exactly; this is the unwrap, in one place,
    so no consumer has to rediscover it.
    """
    if isinstance(value, bytes):
        return value
    recovered = ast.literal_eval(value)
    if not isinstance(recovered, bytes):
        raise TypeError(f"expected bytes from payload repr, got {type(recovered).__name__}")
    return recovered
