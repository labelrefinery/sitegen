"""Wrap GOOSE-Ex frames in sitegen's MCAP contract, so both open in one window.

The point of this is not conversion for its own sake -- it is that
"sitegen looks sparse" is an assertion until you can scrub the two recordings
side by side in the same 3D panel with the same colour map. GOOSE-Ex ships
flat `.bin`/`.label` pairs, so there is no timeline and nothing to drag into
Foxglove; twenty lines of framing fixes that.

The output follows the README's contract exactly:

  * `/lidar/points`         `foxglove.PointCloud`, `x y z intensity`
  * `/ground_truth/points`  `foxglove.PointCloud`, `x y z instance`

so a labeler pointed at a sitegen recording reads a GOOSE-Ex one unchanged.
Intensity is rescaled from Ouster's raw remission to the 0-1 range sitegen
uses, because Foxglove's `Color by` maps the field linearly and a handful of
retroreflective returns at 255 would otherwise flatten everything else to
black. `instance` packs the GOOSE class and instance into the same uint32
slot sitegen uses for its box index -- the ids do not mean the same thing
across the two files, which is the honest situation and is why they are on a
held-out topic in both.

Frames carry their real capture timestamps, so playback runs at the true
5-second labelling interval rather than a synthesised rate.

    uv run python to_mcap.py \\
        --goose ../../../data/goose-ex/gooseEx_3d_val \\
        --scenario alice_scenario02 --limit 40 --out goose_alice.mcap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer

import goose

FLOAT32 = PackedElementField.FLOAT32
UINT32 = PackedElementField.UINT32


def _fields(fourth: str, kind: int) -> list[PackedElementField]:
    return [
        PackedElementField(name="x", offset=0, type=FLOAT32),
        PackedElementField(name="y", offset=4, type=FLOAT32),
        PackedElementField(name="z", offset=8, type=FLOAT32),
        PackedElementField(name=fourth, offset=12, type=kind),
    ]


def _cloud(ns: int, xyz: np.ndarray, fourth: np.ndarray, fields) -> PointCloud:
    stamp = Timestamp()
    stamp.FromNanoseconds(ns)
    packed = np.empty((len(xyz), 4), dtype=np.float32)
    packed[:, :3] = xyz
    # The fourth column is written through whatever dtype the field declares;
    # viewing the float32 buffer as uint32 keeps the bits intact for instance
    # ids while leaving intensity as a genuine float.
    packed[:, 3:].view(fourth.dtype)[:, 0] = fourth
    return PointCloud(
        timestamp=stamp,
        frame_id="lidar",
        pose=Pose(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        point_stride=16,
        fields=fields,
        data=packed.tobytes(),
    )


def convert(root: Path, scenario: str | None, limit: int | None, out: Path) -> int:
    paths = goose.find_frames(root, "alice")
    if scenario:
        paths = [p for p in paths if scenario in str(p)]
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit("no frames matched")

    with out.open("wb") as fh, Writer(fh) as writer:
        for frame in goose.iter_frames(paths):
            # ..._<index>_<capture ns>_pcl.bin -- the timestamp is the field
            # before the suffix, in nanoseconds since the epoch.
            ns = int(frame.path.name.split("_")[-2])
            intensity = np.clip(frame.remission / 255.0, 0.0, 1.0).astype(np.float32)
            writer.write_message(
                topic="/lidar/points",
                message=_cloud(ns, frame.xyz, intensity, _fields("intensity", FLOAT32)),
                log_time=ns,
                publish_time=ns,
            )
            packed_id = (
                frame.semantic.astype(np.uint32) << 16 | frame.instance.astype(np.uint32)
            )
            writer.write_message(
                topic="/ground_truth/points",
                message=_cloud(ns, frame.xyz, packed_id, _fields("instance", UINT32)),
                log_time=ns,
                publish_time=ns,
            )
    return len(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goose", type=Path, required=True)
    ap.add_argument("--scenario", default=None, help="e.g. alice_scenario02")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    n = convert(args.goose, args.scenario, args.limit, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), {n} frames")


if __name__ == "__main__":
    main()
