"""Write predicted labels into their own MCAP, for viewing beside the scene.

The scene file stays untouched. Foxglove merges multiple local files into one
playback timeline, so a 200 KB overlay opens next to an 86 MB scene and lines
up frame for frame -- no copying, no risk of a prediction being mistaken later
for part of the recording, and one overlay per pipeline stage on the same
input.

Each prediction set becomes its own `/pred/<name>` topic carrying
`foxglove.SceneUpdate` cuboids, so it can be toggled independently in the 3-D
panel and compared against `/ground_truth/actors`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp
from mcap.reader import make_reader
from mcap_protobuf.writer import Writer

import numpy as np

#: Same scene epoch `writer.generate` uses, so an overlay lines up even when no
#: scene file is handed over to read it from.
DEFAULT_EPOCH_NS = 1_700_000_000_000_000_000

#: Validated categorical slots, in order. One per prediction set.
PALETTE = [
    (0.165, 0.471, 0.839),  # blue
    (0.922, 0.408, 0.204),  # orange
    (0.106, 0.686, 0.478),  # aqua
    (0.929, 0.631, 0.000),  # yellow
    (0.910, 0.482, 0.643),  # magenta
    (0.290, 0.227, 0.655),  # violet
]


def scene_epoch_ns(scene: Path) -> int:
    """First log time in a scene file: the instant `t = 0` refers to."""
    with open(scene, "rb") as f:
        for _, _, message in make_reader(f).iter_messages():
            return message.log_time
    raise ValueError(f"{scene} contains no messages")


def _yaw_quaternion(theta: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=float(np.sin(theta / 2)), w=float(np.cos(theta / 2)))


def read_predictions(path: Path) -> dict[float, list[dict[str, str]]]:
    by_time: dict[float, list[dict[str, str]]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_time[round(float(row["t"]), 4)].append(row)
    return by_time


def write_overlay(
    predictions: list[Path],
    out: Path,
    epoch_ns: int = DEFAULT_EPOCH_NS,
    lifetime_s: float = 0.15,
    show_ids: bool = True,
) -> dict[str, int]:
    """One `/pred/<name>` SceneUpdate topic per prediction CSV."""
    counts: dict[str, int] = {}
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "wb") as f, Writer(f) as writer:
        for i, path in enumerate(predictions):
            topic = f"/pred/{path.stem}"
            r, g, b = PALETTE[i % len(PALETTE)]
            frames = read_predictions(path)
            written = 0

            for t in sorted(frames):
                ns = epoch_ns + int(round(t * 1e9))
                stamp = Timestamp()
                stamp.FromNanoseconds(ns)

                entities = []
                for row in frames[t]:
                    cx, cy, cz = (float(row[k]) for k in ("x", "y", "z"))
                    width, length, height = (float(row[k]) for k in ("w", "l", "h"))
                    theta = float(row["theta"])
                    conf = float(row.get("conf", 1.0) or 1.0)
                    # Wireframe-ish: low alpha so the point cloud stays visible
                    # through the box, brighter for a confident detection.
                    alpha = 0.25 + 0.35 * min(max(conf, 0.0), 1.0)

                    cube = CubePrimitive(
                        pose=Pose(
                            position=Vector3(x=cx, y=cy, z=cz),
                            orientation=_yaw_quaternion(theta),
                        ),
                        # `l` runs along the heading, which is the entity's x axis.
                        size=Vector3(x=length, y=width, z=height),
                        color=Color(r=r, g=g, b=b, a=alpha),
                    )
                    texts = []
                    if show_ids:
                        texts.append(
                            TextPrimitive(
                                pose=Pose(
                                    position=Vector3(x=cx, y=cy, z=cz + height / 2 + 0.4),
                                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                                ),
                                billboard=True,
                                font_size=11.0,
                                scale_invariant=True,
                                color=Color(r=r, g=g, b=b, a=0.95),
                                text=str(row["track_id"]),
                            )
                        )
                    entities.append(
                        SceneEntity(
                            timestamp=stamp,
                            frame_id="map",
                            id=f"{path.stem}/{row['track_id']}",
                            lifetime=Duration(nanos=int(lifetime_s * 1e9)),
                            cubes=[cube],
                            texts=texts,
                        )
                    )

                writer.write_message(
                    topic=topic,
                    message=SceneUpdate(entities=entities),
                    log_time=ns,
                    publish_time=ns,
                )
                written += 1
            counts[topic] = written
    return counts
